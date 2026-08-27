# AgentSched — vLLM Batch-Shape GPU Cost Benchmark

동일한 token budget `Σn_i` 하에서 batch를 구성하는 `(n_i, c_i)` 조합이 실제 GPU
execution cost를 어떻게 바꾸는지 측정한다.

- `n_i`: request i가 이번 iteration에서 새로 계산하는 token 수
- `c_i`: prefix cache에서 재사용되는 KV context token 수

**핵심 가설**: `Σn_i`라는 단일 token budget만으로는 heterogeneous batch의 실제
GPU execution cost를 표현할 수 없다.

## 방법

실제 vLLM v1 execution path를 사용한다 (`LLMEngine.step()` 직접 구동,
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`).

1. **Phase A (warm)**: request별 고유 random prefix `c_i` token을 prefill →
   automatic prefix caching으로 KV cache에 적재 (`c_i % 16 == 0` 정렬).
2. **Phase B (measure)**: `prefix_i + n_i` fresh token, `max_tokens=1`로 전체
   batch를 동시에 add → 첫 `engine.step()`이 정확히 `{(n_i, c_i)}` 구성의 한
   iteration이 된다. trial마다 fresh suffix를 새로 생성해 `n_i`가 cache-hit되지
   않게 한다. co-scheduling 여부(`extra_steps==0`)와 실제 `num_cached_tokens`를
   기록해 구성이 의도대로 실현됐는지 검증한다.

측정: wall-clock iteration latency(median of N trials), throughput, NVML GPU
util/power, analytic FLOPs·DRAM bytes 모델(→ achieved TFLOPS, AI, HBM GB/s
하한 추정). 선택적으로 torch.profiler(kernel별 attention/GEMM 시간)와 Nsight
Compute(`dram__bytes_*`, FMA inst count → measured DRAM bytes/FLOPs).

## 실험

| | 고정 | 변화 |
|---|---|---|
| Exp1 fragmentation | Σn=8192, c 균일 (0/4K/16K/64K/128K) | B ∈ {1..128} |
| Exp2 n-heterogeneity | Σn=8192, B=8, mean(n)=1024, c 균일 | CV(n) ∈ {0, .25, .5, 1.0, 1.75} |
| Exp3 c-heterogeneity | n=1024, B=8, Σc=256K, mean(c)=32K | CV(c) ∈ {0, .35, .7, 1.3} |
| Exp4 n-c correlation | 동일 {n},{c} multiset, Σn, Σc, B=8 | pairing: pos / rand / neg |

Exp1의 c=128K × B≥8 등 KV 용량 초과 config는 자동 skip 후 기록된다.

## 실행 (CUDA 머신)

```bash
pip install "vllm>=0.8" pynvml pandas matplotlib
cd AgentSched

# 전체 실험 (H100 80GB, 8B 모델 기준 수십 분)
VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m bench.runner \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --exp exp1 exp2 exp3 exp4 --trials 5 --warmup 2

# kernel-level (torch.profiler, 선택)
python -m bench.profile_kernels --exp exp2 exp3 exp4

# Nsight Compute (선택, 매우 느림 — 대표 config만)
bash bench/ncu.sh "--exp exp4 --model meta-llama/Llama-3.1-8B-Instruct"

# 분석: summary.csv + plots + hypothesis R^2 비교
python -m bench.analyze
```

## 산출물

- `results/raw.jsonl` — config별 전체 metric (README 상단 스키마 필드 전부)
- `results/summary.csv`, `exp1_fragmentation.png`, `exp2.png`, `exp3.png`,
  `exp4_pairing.png`
- `results/hypothesis.txt` — `latency ~ Σn_i` 단독 vs `+ Σn_i(c_i+n_i)` 등
  shape feature 추가 시 R² 비교 (가설 검증)

## 해석 가이드

- **Exp1**: B↑ ⇒ per-request n↓ ⇒ FlashAttention tile utilization↓, c>0이면
  attention FLOPs는 동일하지만 KV read bytes는 B에 비례해 증가 → AI 감소,
  memory-bound 전환점이 roofline plot(3번째 패널)에서 보인다.
- **Exp2**: varlen batch에서 최장 sequence가 kernel wave를 지배 → CV(n)↑ 시
  tile imbalance로 achieved TFLOPS 감소 예상.
- **Exp3**: Σc가 같아도 KV traversal이 긴 request가 attention kernel latency를
  지배 (kernel 시간은 max에 민감, bytes는 sum에 비례).
- **Exp4**: attention FLOPs ∝ Σ n_i·(c_i+n_i) 는 pairing에 따라 달라진다
  (pos ≫ neg). 동일 budget에서 latency 차이가 곧 가설의 직접 증거.
