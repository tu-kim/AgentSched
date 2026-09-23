# AgentSched — vLLM Batch-Shape GPU Cost Benchmark

동일한 token budget `Σn_i` 하에서 batch를 구성하는 `(n_i, c_i)` 조합이 실제 GPU
execution cost를 어떻게 바꾸는지 측정한다.

- `n_i`: request i가 이번 iteration에서 새로 계산하는 token 수
- `c_i`: prefix cache에서 재사용되는 KV context 길이

**핵심 가설**: `Σn_i`라는 단일 token budget만으로는 heterogeneous batch의 실제
GPU execution cost를 표현할 수 없다.

환경: 단일 GPU (A100 80GB 기준), expert parallelism 없음(MoE는 fused-MoE 단일 GPU
경로; EP의 token dispatch 비용은 이 연구에서 제외).

## 1. 왜 attention-dominant 영역인가

iteration 비용은 두 부분으로 나뉜다.

- **GEMM (projection / FFN / MoE expert)**: FLOPs ∝ Σn_i, weight는 iteration당 한 번
  읽음 → 비용이 `Σn_i`에 비례. **token budget이 잘 잡는 부분.**
- **Attention**: FLOPs ∝ Σ n_i(c_i + n_i/2). `Σn_i`가 같아도 `c_i`와 pairing에 따라
  달라짐. **token budget이 놓치는 부분.**

GEMM이 지배적인 영역(짧은 context)에서는 `Σn_i`만 봐도 충분하다. 따라서 실험은
두 단계다.

1. **Exp0**: attention이 GEMM을 넘어서는 context 길이 `c*`를 실측으로 찾는다.
2. **Exp1–4**: `c ≥ c*` 영역에서 `(n_i, c_i)` 분포에 따른 batching 효율 변화를 관측한다.

### prefill과 decode는 반드시 분리한다

이 프레임워크에서 request는 `(n_i, c_i)`이므로 **decode는 `n_i = 1`인 특수 케이스**다.
둘을 섞으면 안 되는 이유:

- **GQA/MQA/MLA는 prefill의 score FLOPs를 거의 바꾸지 않고 decode의 KV read만 바꾼다.**
  합쳐서 재면 이 축의 효과가 통째로 가려진다. (위 표의 decode AI 1.0 → 4.0 → 70.8이
  prefill에서는 전부 동일하다.)
- **attention 비중은 c에 따라, MoE/FFN 비중은 B에 따라** 크게 움직인다. 한 지점의 비율로
  모델을 비교할 수 없으므로 두 축을 각각 훑어야 한다(Exp0/Exp1이 prefill 쪽, Exp5가
  decode 쪽의 2D sweep).
- cost-model 회귀도 phase별로 따로 적합시킨다. 섞으면 phase 지시변수가 batch shape 효과를
  압도한다.

`bench/configs.py`의 `PREFILL_EXPS = (exp0…exp4)`, `DECODE_EXPS = (exp5,)`가 이 경계이고
`analyze.py`가 phase별로 따로 보고한다.

> **측정상 주의**: exp5의 "decode"는 `prefix(c) + 1 fresh token` 프롬프트를 prefix-cache
> hit로 실행한 것이라, scheduler 상태(WAITING→첫 스케줄)는 실제 decode(RUNNING)와 다르다.
> 다만 attention metadata는 `query_len = 1`로 동일하게 분류되므로 **커널 경로는 같다**
> (MLA의 `reorder_batch_threshold = 1`도 이 기준). TBT SLO 같은 스케줄러 효과를 보려면
> 별도의 실행 중 decode 실험이 필요하다.

### 아키텍처 계열 (분석 모델: `bench/arch_compare.py`)

측정 대상은 attention 구조를 한 단계씩 바꾸는 9개 계열이다. `tier`는 단일 A100 80GB에서
측정 가능한지를 뜻한다 — `measurable`(실측), `multi_gpu`(tp≥2 필요), `analytic_only`(분석만).

| # | 계열 | 대표 모델 | tier | params (active) | KV/token | c\*(decode) | c\*(prefill) | decode AI@32K |
|---|---|---|---|---|---|---|---|---|
| 1 | MHA | Llama-2-7B | ✅ | 6.74B | 512 KB | 24.7K | 24.2K | **1.0** |
| 2 | MQA | Falcon-7B | ✅ | 7.22B | 8 KB | 22.8K | 22.3K | **70.8** |
| 3 | GQA | Llama-3-8B | ✅ | 8.03B | 128 KB | 26.6K | 26.1K | **4.0** |
| 4 | local/global SWA | Gemma-3-4B | ✅ | 4.55B (3.21B) | 20 KB | 150.7K | 150.2K | 2.0 |
| 5 | MoE FFN | Mixtral-8x7B | ⚠️ tp≥2 | 46.7B (12.6B) | 128 KB | 48.1K | 47.6K | 4.0 |
| 5′ | MoE FFN (단일 GPU) | Qwen1.5-MoE-A2.7B | ✅ | 14.3B (2.07B) | 192 KB | 21.0K | 20.5K | 1.0 |
| 6 | MLA | DeepSeek-V2-Lite | ✅ | 15.7B (2.24B) | 30 KB | **4.8K** | **9.8K** | 30.2 |
| 7 | sparse (DSA) | DeepSeek-V3.2 | 📐 | 671B (35.7B) | 69 KB | 36.6K | 54.5K | 4.1 |
| 8 | linear hybrid | Qwen3-Next-80B-A3B | ⚠️ | 79.3B (2.84B) | 24 KB | 28.5K | 28.0K | — |
| 9 | 압축+sparse | DeepSeek-V4-Flash | ❌ | — | — | — | — | — |

HF `config.json`으로 검증한 값(계열 9는 사양 미확인 — 아래 참조). `python -m bench.arch_compare
--pairs`가 표·그림·통제 쌍을 재생성한다. `KV/token`은 **한계 증가율**이라 계열 4·8에서는
window·recurrent 층이 빠져 층 수보다 훨씬 작다(Gemma-3: 34층 중 global 5층만 증가).

**계열별로 무엇이 바뀌는가** (decode AI = attention FLOP/byte):

- **1→2→3 (MHA→MQA→GQA)**: prefill의 score FLOPs는 **거의 그대로**, decode의 KV read만
  g=H/H_kv배 줄어든다. bf16 decode AI가 정확히 g로 떨어지는 것이 교과서적 결과이고
  (MHA 1.0, GQA×4 4.0, MQA×71 70.8) 테스트로 고정해 뒀다. 즉 이 축은 **decode 전용 효과**다.
- **4 (SWA)**: local 층의 비용이 window에서 멈춘다. FLOPs와 bytes가 같은 비율로 줄어
  **AI는 그대로**이고 총량만 준다 — capacity/throughput 이득이지 intensity 이득이 아니다.
  c\*가 150K로 6배 밀린다(Gemma-3는 global 5층만 계속 자람).
- **5 (MoE)**: active params가 줄어 GEMM FLOPs가 준다. 다만 expert weight는 token이 하나라도
  가면 전부 읽히므로 **Σn과 무관한 floor**가 생긴다(Qwen1.5-MoE ≈ 25 GB, DeepSeek-V2-Lite
  ≈ 29 GB per iteration → A100에서 12–15 ms). decode에서 B가 커지면 touch되는 expert가 늘어
  FFN 시간이 FLOPs보다 **weight traffic**에 좌우된다 — exp5의 B축이 이걸 본다.
- **6 (MLA)**: 유일하게 **prefill과 decode의 커널 종류가 다르다**. prefill은 prefix-hit된
  latent를 64K chunk 단위로 gather → `kv_b_proj`로 head별 K/V 복원 → FA2 → `merge_attn_states`
  이고, 이 복원 비용은 `2·r·H·(d_nope+d_v)·c`로 **n과 무관하게 c에 비례**한다. decode는
  weight absorption으로 576-wide latent에 직접 score를 내므로 복원이 아예 없고, KV는 1 head로
  줄지만 head당 score FLOPs가 2.8배 커진다 → **메모리 병목에서 연산 병목으로 이동**
  (decode AI 30.2). 그래서 c\*를 하나의 수로 말할 수 없다(decode 4.8K vs prefill 9.8K).
- **7 (DSA)**: indexer가 전 위치를 훑고(O(N), 저강도) 상위 top-k=2048만 sparse MLA로 간다.
  분석 모델 기준 **indexer 비중이 c=4K에서 10.5%, 64K에서 65%, 1M에서 97%**로 커진다 —
  "문맥이 길어지면 indexer가 몫을 가져가는가"가 이 계열의 측정 핵심이고, 그래서 attention
  시간을 indexer / top-k 선택 / sparse attention 셋으로 쪼개야 한다. indexer의 KV는 MQA 1 head
  FP8이라 바이트는 싸고 FLOPs만 자란다.
- **8 (linear hybrid)**: 48층 중 12층만 softmax이므로 **문맥에 비례하는 attention이 1/4만
  남는다**. 나머지 36층은 문맥과 무관한 recurrent state(요청당 약 38 MB, c와 무관)를 갖는다.
  hidden 2048로 작아 GEMM이 얇으니 GPU 활용률이 낮을 수 있고, 비중 해석 시 이를 같이 봐야 한다.
- **9 (CSA/HCA)**: V4-Flash는 arXiv 2606.19348 기준 최신 구조로, 제 지식 시점 이후라
  **사양을 확인하지 못했다**. 분석 모델에 넣지 않았고 preset도 만들지 않았다. 넣으려면
  CSA의 압축률과 HCA의 dense 범위를 원문에서 확정해야 한다.

### 통제된 비교 쌍

실제 모델끼리 비교하면 여러 변수가 동시에 바뀐다(Llama-2→Llama-3은 vocab 32K→128K, FFN
11008→14336까지 같이 바뀌므로 **LM head GEMM을 분리 집계**한다 — `est_flops_lm_head`).
그래서 **한 변수만 바꾼 synthetic config**를 `--load-format dummy`로 함께 돌린다
(timing은 weight 값과 무관; `bench/synthetic_configs/`).

| 쌍 | 바뀌는 것 | KV/token | decode AI@32K |
|---|---|---|---|
| `kv-mha-1.8b` → `kv-gqa4-1.8b` | `num_key_value_heads` 16→4 **만** | 192K → 48K | 1.0 → 4.0 |
| `kv-gqa4-1.8b` → `kv-mqa-1.8b` | `num_key_value_heads` 4→1 **만** | 48K → 12K | 4.0 → 16.0 |
| `kv-mha-1.8b` → `mla-dense-1.8b` | full KV → 576-wide latent | 192K → 27K | 1.0 → 30.2 |
| `kv-mha-1.8b` → `swa-1.8b` | 5:1 local/global window **만** | 192K → 32K | 1.0 → 1.0 |
| `qwen1.5-1.8b` → `qwen1.5-moe-a2.7b` | FFN dense→MoE **만** (attention 동일) | 192K → 192K | 1.0 → 1.0 |
| `mistral-7b` → `mixtral-8x7b` | FFN dense→MoE **만** (tp≥2) | 128K → 128K | 4.0 → 4.0 |
| `deepseek-v3` → `deepseek-v3.2` | DSA indexer 유무 **만** (분석) | 69K → 69K | 241 → 4.1 |

## 2. Attention AI 식 (검증됨)

MHA, bf16, causal 무시, attention kernel이 직접 읽고 쓰는 traffic만 셀 때:

```
AI_attn = Σ n_i(n_i + c_i) / Σ (2n_i + c_i)      [FLOP/byte, 계수 1]
```

분모의 `2n_i` = Q read + O write (n_i) + 새 token K·V read (n_i), `c_i` = cached
K·V read. 분자는 (n, c)에 2차, 분모는 1차이므로 heterogeneity는 오직

```
Σn_i(n_i+c_i) = N·[ m_n²(1+CV_n²) + m_n·m_c·(1 + ρ·CV_n·CV_c) ]
```

의 두 항 — `Σn_i²`(CV_n)와 `Σn_i·c_i`(ρ·CV_n·CV_c) — 을 통해서만 들어온다. CV_c는
단독으로는 AI에 영향이 없다(Exp3가 negative control인 이유). 보정: causal → n_i²
계수 ½; 새 KV write 포함 → 분모 `3n_i + c_i`; GQA(g) → 분모 `n_i + (c_i+n_i)/g`; MLA →
위의 decompression 항과 latent/materialized bytes. `bench/metrics.py`는 보정을 전부
포함한 정확한 값을 쓰고, 이 식에 해당하는 값은 `est_ai_attn`으로 따로 기록한다.

## 3. 방법

실제 vLLM v1 execution path(`LLMEngine.step()` 직접 구동)를 사용한다. "한 번의
`step()` = 정확히 한 scheduler iteration"이 되려면 (vLLM ≥ 0.28):

| 설정 | 이유 |
|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 기본값(1)은 EngineCore가 별도 프로세스에서 자체 loop를 돌고 `step()`은 출력만 꺼냄 |
| `async_scheduling=False` | 기본값(True)은 `step()`이 forward 완료 전에 반환 (출력은 다음 `step()`) |
| `attention_backend="FLASH_ATTN"` | env var `VLLM_ATTENTION_BACKEND`는 제거됨. A100→FA2, H100→FA3. MLA 모델, 또는 pre-Hopper + `--kv-cache-dtype fp8`은 auto |
| `hf_overrides.max_position_embeddings ≥ max(c+n)` | RoPE cos/sin 테이블 크기. `VLLM_ALLOW_LONG_MAX_MODEL_LEN`만 쓰면 OOB |
| `disable_cascade_attn=True`, `block_size=16` | 경로 고정 |

절차:

1. **Phase A (warm)**: request별 고유 random prefix `c_i` token을 prefill → 16-token
   block 단위로 prefix cache에 적재 (`c_i % 16 == 0`).
2. **Phase B (measure)**: `prefix_i + n_i` fresh token, `max_tokens=1`로 전체 batch를
   add → 첫 `engine.step()`이 정확히 `{(n_i, c_i)}` 구성의 한 iteration. trial마다
   suffix를 재생성해 `n_i`가 cache-hit되지 않게 한다.
3. **실현 검증**: `co_scheduled` (첫 step에서 B개 출력, 추가 step 0), `cache_hits_ok`
   (request별 `num_cached_tokens == c_i`)를 기록한다.

측정: iteration latency(median), throughput, NVML util/power, analytic FLOPs·bytes
(→ achieved TFLOPS, `est_ai_attn`, analytic iteration AI), `--kernel-profile` 시
torch.profiler로 attention / GEMM / MoE / KV-cache-write / other kernel 시간.
Kernel 이름(vLLM 0.28): dense FA2는 `flash::flash_fwd_splitkv_kernel`(paged KV라 항상
split-KV 변형), FA3는 `cutlass::device_kernel<flash::…FlashAttnFwdSm90…>`, MLA prefill은
`flash_fwd_kernel` + `merge_attn_states_kernel` + `gather_and_maybe_dequant_cache_page`,
MoE는 `fused_moe_kernel` + `moe_align_block_size_*` + `topk_softmax`/`moe_sum_*`.
(MLA의 latent decompression `kv_b_proj`는 cuBLAS GEMM이라 kernel 이름상 GEMM 버킷에
들어가며, 분석 모델에서만 attention side로 계산된다. 첫 실행 시 `kernel_top`으로
미분류 kernel을 확인할 것.)

## 4. 실험

| | 고정 | 변화 | 목적 |
|---|---|---|---|
| Exp0 dominance | Σn=8192, B ∈ {1, 8, 64, 256} | c ∈ {0, 1K, …, 512K} | **c\* 발견** (attention 시간 비중 ≥ 50%); B=256(n=32)은 MLA의 Σc\_i 항이 가장 크게 보이는 구석 |
| Exp0 budget | B=8, c ∈ {0, 16K} | Σn ∈ {1K, 2K, …, 64K} | `latency ~ Σn` baseline 회귀용 |
| Exp1 fragmentation | Σn=8192, c 균일 | B ∈ {1..512}, c ∈ {0, 4K, …, 512K} | 같은 budget, B·c에 따른 throughput |
| Exp2 n-heterogeneity | Σn=8192, B=8, c ∈ {0, base_c, 2·base_c, 4·base_c} | CV(n) ∈ {0, .18, .38, 1.03, 1.87, 2.60} | Σn_i² 효과 (c=0) / kernel 효과 (c≫n) |
| Exp3 c-heterogeneity | n=1024, B=8, mean(c)=base_c | CV(c) ∈ {0, .29, .61, 1.27, 1.94} | AI-neutral control |
| Exp4 n-c correlation | 동일 {n},{c} multiset, mean(c)=base_c, B=8 | ρ ∈ 도달 가능 범위 7점 (≈ −0.6 … +0.98) | Σn_i·c_i 효과 (multi-turn의 핵심 항) |
| **Exp5 decode** | **n_i = 1** | B ∈ {1,8,32,128,512} × c ∈ {0,1K,4K,16K,64K,256K} | **attention 계열 비교의 본무대**: KV read가 유일한 scaling 항. c축=attention, B축=MoE/FFN |

`base_c`는 기본 `auto`(분석적 c\*(n=1024) 이상의 2의 거듭제곱)이며, Exp0 실측 c\*를
보고 `--base-c`로 override한다. 동일 shape는 한 번만 측정하고 alias별로 기록한다.
`{n},{c}` multiset을 넓힌 대가로 Exp4의 도달 가능 ρ 범위는 좌우 비대칭이다(음의
방향이 더 좁음) — 두 극단 multiset이 완전히 skew-대칭이 아니기 때문이며, 표시되는
config 이름은 목표값이 아니라 실제 도달한 ρ다. 전체 config는 164개(중복 shape 제거 후
약 129개 측정), 이전(약 99개) 대비 넓어진 sweep이다.

## 5. 실행

```bash
pip install "vllm==0.28.*" pynvml pandas matplotlib     # kernel 이름/API는 0.28 기준
cd AgentSched

# 0. GPU 없이: 테스트 + 측정 검증 + 아키텍처 비교 + feasibility
python -m unittest discover -s tests -t .                           # 131 tests, ~2s
python -m bench.selftest                                            # §7, 항상 먼저
python -m bench.arch_compare --pairs --gpu-mem-gib 80                # 계열표 + 통제 쌍
python -m bench.configs --model qwen1.5-1.8b --gpu-mem-gib 80        # base_c auto, gate 0.95

# 1. c* 찾기 (모델별)
python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp0 --kernel-profile
python -m bench.analyze          # results/report.txt 의 "Exp0: c*"

# 2. shape 실험 (base_c ≥ c*)
python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp1 exp2 exp3 exp4 \
    --base-c 16384 --kernel-profile
python -m bench.analyze

# 3. decode (계열 비교의 본무대 — prefill과 따로 집계됨)
python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp5 --kernel-profile

# 아키텍처 계열: 같은 명령을 아래 모델로 반복 (동일 raw.jsonl에 append; analyze가 모델·phase별 분리)
#   1 MHA       --model NousResearch/Llama-2-7b-hf
#   2 MQA       --model tiiuae/falcon-7b            # KV 8KB/token → c를 아주 크게 잡을 수 있음
#   3 GQA       --model NousResearch/Meta-Llama-3-8B
#   4 SWA       --model unsloth/gemma-3-4b-it       # local 29 / global 5, window 1024
#   5 MoE       --model Qwen/Qwen1.5-MoE-A2.7B      # Mixtral은 93GB → tp≥2
#   6 MLA       --model deepseek-ai/DeepSeek-V2-Lite
# 통제 쌍 (한 변수만 다른 synthetic config, 가중치 불필요):
#   --model bench/synthetic_configs/kv-mha-1.8b  --load-format dummy
#   --model bench/synthetic_configs/kv-gqa4-1.8b --load-format dummy
#   --model bench/synthetic_configs/kv-mqa-1.8b  --load-format dummy
#   --model bench/synthetic_configs/swa-1.8b     --load-format dummy
#   --model bench/synthetic_configs/mla-dense-1.8b --load-format dummy
# 옵션: --load-format dummy (random weight, 다운로드 없음), --kv-cache-dtype fp8
```

환경 메모: A100(SM80)에서는 FA2가 사용되고(FA3는 Hopper 전용), MLA 모델에
`attention_backend=FLASH_ATTN`을 강제하면 ValueError가 나므로 runner가 자동으로 auto로 둔다.
`--kv-cache-dtype fp8`은 FA2에 경로가 없어 pre-Hopper에서는 backend를 auto로 넘긴다(Hopper+
권장). `--kernel-profile`은 tp=1 전용(worker가 별도 프로세스면 driver 측 profiler가 kernel을
못 본다). `--device`는 CUDA-visible index이며 NVML 핸들은 UUID로 매칭한다. NVML `gpu_util`/
`mem_util`은 driver의 trailing window(~0.1–1 s) 평균이라 iteration 단위 지표가 아니다(참고용;
per-iteration busy time은 kernel profile을 쓸 것). `--trust-remote-code`는 위 모델 어디에도
필요 없다. DeepSeek-V2-Lite는 yarn(4096×40=163840)이라 `max_position_embeddings` override가
무효이며 c ≤ 128K 범위에서는 필요도 없다.

## 6. 산출물 (`results/`)

- `raw.jsonl` — config별 전체 metric (shape 통계 5개 충분통계량 포함:
  `batch_size, token_budget, sum_c, sum_n_sq, sum_nc`; 모델 메타; 실현 검증 flag)
- `summary.csv` — 위 + 효과 분해 열
- `exp0_dominance.png` — latency, latency/latency(c=0), attention 시간 비중 vs c (모델×B)
- `exp1_fragmentation.png`, `exp2_n_hetero.png`, `exp3_c_hetero.png`, `exp4_pairing.png`
- `exp5_decode.png` — decode latency/throughput vs c 와 vs B (계열 비교용)
- `arch_compare.png/.txt` — 분석 모델의 아키텍처 비교
- `report.txt` — c\*, 고정 budget에서의 latency 산포, cost-model R² 비교, 효과 분해 표

GPU 없이 `analyze.py`/`arch_compare.py` 파이프라인만 시험하려면 `bench/synth_results.py`가
toy latency model로 스키마가 동일한 `raw.jsonl`을 생성한다(측정값 아님):
`python -m bench.synth_results --out results/synth.jsonl --models qwen1.5-1.8b deepseek-v2-lite`.

## 7. 측정 검증 (`python -m bench.selftest`)

attention/MoE 시간이 실제로 제대로 잡히는지는 세 단계로 검증한다. 앞의 두 개는
GPU 없이 항상 실행 가능하고, 세 번째는 실제 실행 중에 자동으로 경고를 낸다.

1. **Kernel 분류 (31개 실제 kernel 이름)**: vLLM 0.28이 A100/H100에서 실제로 띄우는
   kernel 이름들을 `classify()`에 넣어 의도한 버킷으로 가는지 확인한다. substring
   매칭은 순서와 키 선택에 민감해서 조용히 틀리기 쉽다 — FA3 이름에
   `cutlass::device_kernel`이 들어가 GEMM 키보다 먼저 검사해야 하고, MLA KV write
   (`concat_and_cache_mla_kernel`)는 attention 키 `_mla`와 매칭되므로 KVCACHE를 먼저
   봐야 하며, MLA context gather(`gather_and_maybe_dequant_cache_page`)는 `cache`가
   들어가지만 attention으로 가야 한다. dense KV write(`reshape_and_cache_flash_kernel`)가
   안전한 것은 attention 키가 bare `flash`가 아니라 `flash::`/`flash_fwd`이기 때문이다
   (키를 넓히면 모든 KV write가 attention으로 빨려 들어간다). 특히 A100
   dense attention은 vLLM이 항상 block_table을 넘기기 때문에 prefill에서도
   `flash_fwd_kernel`이 아니라 **`flash_fwd_splitkv_kernel`** 이 뜬다 — 이걸 놓치면
   dense attention 시간이 통째로 사라진다.
2. **분석 모델 수치**: `est_flops_attn`, `est_attn_bytes`, MLA decompression,
   MoE active params / expert weight traffic를 손 유도값과 대조한다. 확인된 값:
   MHA attention FLOPs = `4·H·d·L·Σn(c+(n+1)/2)`, README AI 식이 계수 1로 정확히
   성립, MLA decompression 113 MFLOP/cached token (n과 무관, Σc에만 비례),
   Qwen1.5-MoE expert weight 24.91 GB·DeepSeek-V2-Lite 28.79 GB per iteration
   (Σn에 거의 무관한 floor).
3. **실행 중 자동 경고**: `--kernel-profile`이 profile마다 버킷 구성을 점검해
   attention 버킷이 비었거나, MoE 모델인데 moe 버킷이 비었거나, dense 모델인데 moe
   버킷이 찼거나, 미분류 `other`가 25%를 넘으면 경고하고 미분류 kernel 상위 5개를
   출력한다. 매 config의 미분류 목록은 `kernel_unclassified`로 raw.jsonl에도 남는다.

**분류기가 원리적으로 구분할 수 없는 것** (vLLM 0.28 소스에서 한 prefill step의 kernel을
전수 열거해 확인. 측정 해석 시 반드시 감안할 것):

- **MoE 버킷은 MoE 블록의 약 절반만 잡는다.** shared expert와 router gate는 일반 cuBLAS
  GEMM이라 GEMM 버킷으로 간다. Qwen1.5-MoE 기준 shared expert FLOPs(token당 69.2 MFLOP)는
  routed top-4 expert와 **거의 동일**하므로, `kernel_time_moe_us`는 MoE 연산의 절반 수준이다.
  Σn에 비례하는 부분(shared/gate)과 expert weight floor(routed)를 나눠 보려면 이 점이 중요하다.
- **MLA의 `kv_b_proj` decompression은 cuBLAS GEMM**이라 GEMM 버킷에 들어간다.
  DeepSeek-V2-Lite, B=8·n=1024·c=32K 기준 step time의 **약 14%**로 추정되며 Σc에 비례한다.
  같은 조건에서 MLA context FA2가 약 52%다. 추가로 k-concat 복사와 V zero-pad(128→192)도
  Σc에 비례하지만 `other`로 간다. 정확히 분리하려면 `_compute_prefill_context`를
  `record_function`으로 감싸야 한다.
- `act_and_mul`(SwiGLU)은 routed expert·shared expert·dense MLP가 모두 같은 kernel을 쓰므로
  귀속이 불가능하다. `other`를 오염시키지 않도록 **별도 `activation` 버킷**으로 뺐다.
- **`--enforce-eager` 여부로 `other`/`activation`의 구성이 완전히 바뀐다.** 기본값(compiled)에서는
  RMSNorm·RoPE·SiLU가 Inductor Triton kernel(`triton_*_fused_*`)이 되고, eager에서는
  `vllm::rms_norm_kernel` 등 vLLM CUDA kernel이 된다. attention/MoE/KV-write 버킷은 opaque
  custom op 안이라 두 모드에서 동일하다. 모드가 다른 결과끼리 `other`를 비교하면 안 된다.

## 8. 테스트 (`python -m unittest discover -s tests -t .`)

GPU 없이 2초 안에 도는 99개 테스트. `bench/selftest.py`(§7)가 "측정이 물리적으로
맞는가"를 보는 반면, 여기서는 "코드가 의도대로 동작하는가"를 본다.

| 파일 | 커버 범위 | 대표적으로 잡는 것 |
|---|---|---|
| `test_configs.py` | 생성기 불변식 | exp3의 **AI-neutrality**(CV_c를 바꿔도 `sum_n_ctx`/`sum_nc` 불변 — 이게 깨지면 negative control이 아님), exp4가 `{n}`·`{c}` multiset과 Σn·Σc·B를 모두 고정하고 pairing만 바꾸는지, CV 라벨이 실제 CV와 일치하는지, `kv_tokens_needed`가 request별로 올림하는지, shape_key 순서 불변성 |
| `test_metrics.py` | ModelSpec 파싱·유도량 | **preset 하드코딩 값 == config.json 파싱 결과** (손입력 오류 차단), MoE 키 변형(`num_experts`/`n_routed_experts`/`num_local_experts`), MLA c\*의 n 의존성, fp8 KV가 읽기 바이트를 반으로, expert weight가 Σn floor인지, attention 항의 request 가산성 |
| `test_analyze.py` | 분석 파이프라인 | 효과 분해 항등식, 모델·그룹 간 기준행이 섞이지 않는지, c\* 탐지(측정 share / latency 2배 fallback / c=0 행 없음), 전부 skip된 입력, `fit_r2` 성질, synth→analyze end-to-end |
| `test_runner.py` | GPU 비의존 부분 | 분류기 순서 의존성(어떤 것이 실제로 load-bearing인지), 키가 전부 소문자인지, NVML/capacity 조회, plan CLI와 runner의 base_c·capacity gate 일치 |
| `test_families.py` | 아키텍처 계열 | layer stack 분해(SWA 5:1, linear 3:1), decode AI == group size, SWA가 AI를 안 바꾸고 총량만 줄인다는 것, DSA indexer 비중 증가, 통제 쌍이 실제로 한 변수만 다른지, synthetic config가 preset과 일치하는지, tier가 단일 GPU 적재 가능성과 맞는지 |
| `test_selftest.py` | §7 검증을 테스트로 | selftest의 3개 검사를 CI에서도 실패하게 |

## 9. 해석 가이드

**효과 분해** (Exp2–4, homogeneous / ρ≈0 기준 대비):

```
latency_ratio  =  flops_ratio  ×  (1 / tflops_ratio)      ← 항등식 (정확)
                  ───────────      ────────────────
                  분석적 work 증가    kernel efficiency 하락
                  (scheduler가 예측 가능)   (tile imbalance 등, 잔차)

work_ratio = Σn_i(c_i+n_i/2) 비율 = flops_ratio의 attention 부분
             (attention-dominant 영역에서만 flops_ratio ≈ work_ratio)
```

`achieved_tflops = est_flops_total / latency`이므로 위 분해는 **항등식**이다 — 독립적인
검증이 아니라, 관측된 latency 변화를 "분석적으로 예측 가능한 work 증가"와 "설명되지
않는 효율 하락"으로 **귀속(attribution)** 하는 것이다. `flops_ratio`로 설명되는 부분은
scheduler가 `(Σn, Σc, Σn², Σnc, B)`만으로 예측할 수 있고, `tflops_ratio`로 남는 부분이
cost model이 놓치는 잔차다.

**Exp2에서 잔차가 나올 것으로 예상되는 메커니즘**: vLLM의 FA2 varlen 커널은 grid를
`ceil(max_i n_i / 64) × B × H`로 잡고, 자기 request 길이를 넘는 block은 early-exit한다.
즉 **grid는 Σn이 아니라 max(n_i)로 결정**되므로, Σn을 고정한 채 CV(n)을 키우면 max(n_i)가
커지면서 빈 block이 늘어난다. Exp2의 `tflops_ratio` 하락이 이 예측과 맞는지 확인할 것
(맞다면 scheduler cost model에 `max(n_i)` 항을 넣을 근거가 된다).

**Cost-model R²**: `latency ~ Σn` 단독 vs `+ Σn_i(c_i+n_i/2)` vs 충분통계량 5개, 전체
config와 `c ≥ c*` subset 각각. 후자에서 `Σn` 단독 R²가 낮고 shape-aware 모델의 R²가
높으면 가설이 성립한다.

**주의**: 분석적 c\*는 `c + (n+1)/2` (attended context) 기준이라 cached-c 축에서는
B별로 `(n+1)/2`만큼 이동한다(B=1이면 4K). Exp0의 실측 c\*는 cached-c 기준이다.
