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

### 아키텍처별 attention–GEMM 균형 (분석 모델, `bench/arch_compare.py`)

`c*`는 per-request attention-side FLOPs = linear FLOPs가 되는 cached context다.
MHA/GQA에서는 `c* = P_active / (2·L·H·d)` 로 `n`, `B`와 무관한 모델 상수지만,
MLA에서는 `n`에 의존한다 (아래).

| preset | arch | params (active) | KV/token | 80GiB KV 용량 | c\*(n=64) | c\*(n=1024) | c\*(n=8192) |
|---|---|---|---|---|---|---|---|
| Qwen1.5-1.8B | MHA | 1.84B | 192 KB | ≈ 364K | 12.4K | 12.4K | 12.4K |
| Qwen3-1.7B | GQA ×2 | 1.72B | 112 KB | ≈ 625K | 12.3K | 12.3K | 12.3K |
| Qwen2.5-3B | GQA ×8 | 3.09B | 36 KB | ≈ 1.87M | 18.8K | 18.8K | 18.8K |
| Qwen1.5-MoE-A2.7B | MHA + MoE | 14.3B (2.1B) | 192 KB | ≈ 237K | 21.0K | 21.0K | 21.0K |
| mla-dense-1.8b (synthetic) | MLA | 1.76B | 27 KB | ≈ 2.59M | **1.2K** | 5.8K | 7.4K |
| DeepSeek-V2-Lite | MLA + MoE | 15.7B (2.2B) | 30 KB | ≈ 1.41M | **2.1K** | 10.1K | 13.0K |

(HF `config.json`으로 검증한 값; 전부 ungated, vLLM registry 포함. `python -m bench.arch_compare`
가 그림과 함께 재생성. 그 외 preset: Qwen1.5-0.5B, Qwen3-4B, Qwen2.5-1.5B, Llama-3.2-1B/3B,
OLMoE-1B-7B, granite-3b-a800m, Qwen3-30B-A3B(61 GB → KV 90K로 부적합).)

아키텍처 셀 구성 근거: Qwen1.5-MoE-A2.7B의 attention block(L24/H16/d128/hidden 2048)은
Qwen1.5-1.8B와 동일해 **dense→MoE 차이가 FFN만으로 격리**된다. MLA-dense는 vLLM이 native로
지원하는 public checkpoint가 없어(MiniCPM3는 vLLM에서 full K/V cache로 동작, Youtu/TransMLA는
registry 밖) Qwen1.5-1.8B와 L/hidden/heads/FFN을 맞춘 **synthetic DeepseekV2 config +
`--load-format dummy`** 로 채운다(`bench/synthetic_configs/mla-dense-1.8b`; timing은 weight 값과
무관). MLA 수치는 A100의 FA2 경로(V를 192로 zero-pad) 기준이다.

아키텍처 요소별로 균형이 움직이는 방향:

- **GQA (g = H/H_kv)**: attention FLOPs는 그대로, K·V bytes만 1/g. c\*는 거의 변하지
  않지만 attention AI가 g배 올라가고 KV/token이 줄어 같은 GPU에서 훨씬 큰 c에 도달할
  수 있다. 즉 GQA는 "memory capacity 문제"를 완화하지만 "Σn_i가 attention cost를
  놓치는 문제"는 그대로다.
- **MoE**: active params가 줄면 linear FLOPs가 줄어 c\*가 낮아지고(Qwen1.5-MoE처럼 공유
  expert가 크면 반대), expert weight는 token이 하나라도 가면 전부 읽으므로 GEMM side에
  **Σn과 무관한 floor**가 생긴다(Qwen1.5-MoE ≈ 25 GB, DeepSeek-V2-Lite ≈ 29 GB per
  iteration → A100에서 12–15 ms). 단일 GPU에서는 Triton `fused_moe_kernel`이 쓰이고
  tile config가 M=Σn으로 정해지므로 Σn 고정인 Exp1–4 안에서는 상수다(Exp0 budget sweep은
  Σn//E > 128 경계에서 GROUP_SIZE_M이 바뀜). Attention 항은 변하지 않는다.
- **MLA**: KV cache가 latent(`r + d_rope` per token)라 KV/token이 매우 작지만, vLLM의
  prefill 경로는 prefix-hit된 latent를 64K-token chunk 단위로 gather → `kv_b_proj`로
  **head별 K/V로 복원** → FA2(non-causal) → `merge_attn_states` 한다. 이 decompression
  비용은 `2·r·H·(d_nope+d_v)·c` 로 **n과 무관하게 c에 비례**하므로 `n`이 작은
  (multi-turn) request에서는 attention core보다 커진다. 그래서 c\*가 n=64에서 ≈2K,
  n=8192에서 ≈13K로 달라진다 — MLA에서는 "attention-dominant" 경계 자체가 batch shape에
  의존하고, cost model에 `Σc_i` 항이 추가로 필요하다(Exp3가 MLA에서는 AI-neutral이 아님).
  복원된 K/V가 materialize되므로 attention bytes는 MHA와 비슷한 규모다. A100에서는
  FA2가 서로 다른 head dim을 못 다뤄 V가 128→192로 zero-pad된다(`mla_v_padded`).

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

# 0. GPU 없이: 측정 검증 + 아키텍처 비교 + feasibility
python -m bench.selftest                                            # §7, 항상 먼저
python -m bench.arch_compare --gpu-mem-gib 80
python -m bench.configs --model qwen1.5-1.8b --gpu-mem-gib 80        # base_c auto, gate 0.95

# 1. c* 찾기 (모델별)
python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp0 --kernel-profile
python -m bench.analyze          # results/report.txt 의 "Exp0: c*"

# 2. shape 실험 (base_c ≥ c*)
python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp1 exp2 exp3 exp4 \
    --base-c 16384 --kernel-profile
python -m bench.analyze

# 아키텍처 비교: 같은 명령을 아래 모델로 반복 (동일 raw.jsonl에 append; analyze가 모델별 분리)
#   --model Qwen/Qwen3-1.7B                     # GQA ×2
#   --model Qwen/Qwen2.5-3B                     # GQA ×8
#   --model Qwen/Qwen1.5-MoE-A2.7B              # MHA+MoE (max_pos 8192 → hf_overrides 자동)
#   --model deepseek-ai/DeepSeek-V2-Lite        # MLA+MoE (backend auto → TRITON_MLA + FA2 prefill)
#   --model bench/synthetic_configs/mla-dense-1.8b --load-format dummy   # MLA dense (synthetic)
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
   매칭은 순서에 민감해서 조용히 틀리기 쉽다 — FA3 이름에 `cutlass::device_kernel`이
   들어가고(GEMM 키보다 먼저 검사해야 함), KV write kernel 이름에 `flash`가 들어가며
   (`reshape_and_cache_flash_kernel`), MLA context gather 이름에는 `cache`가 들어간다
   (`gather_and_maybe_dequant_cache_page`, attention으로 분류돼야 함). 특히 A100
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

**분류기가 원리적으로 구분할 수 없는 것** (측정 시 감안할 것):

- MLA의 `kv_b_proj` decompression은 cuBLAS GEMM이라 이름만으로는 GEMM 버킷에 들어간다.
  분석 모델에서만 attention side로 계산되므로, MLA 모델에서 측정 attention 비중은
  분석 FLOP 비중보다 낮게 나온다. 분리하려면 `_compute_prefill_context`에
  `record_function`을 감싸야 한다.
- MoE의 shared expert와 router gate는 일반 cuBLAS GEMM이라 GEMM 버킷에 들어간다.
  moe 버킷은 Triton `fused_moe_kernel` 등 routed-expert 경로만 잡는다.
- `act_and_mul_kernel`(SiLU)은 dense MLP와 MoE expert가 공유해서 `other`로 간다.

## 8. 해석 가이드

**효과 분해** (Exp2–4, homogeneous / ρ≈0 기준 대비):

```
latency_ratio  = work_ratio × (1 / tflops_ratio)
                 ────────────   ───────────────
                 AI 효과         kernel efficiency 효과
                 (Σn_i(c_i+n_i/2) 변화, scheduler가 계산 가능)   (tile imbalance 등, 잔차)
```

`work_ratio`로 설명되는 부분은 scheduler가 `(Σn, Σc, Σn², Σnc, B)`만으로 예측할 수
있고, `tflops_ratio`로 남는 부분이 cost model이 놓치는 잔차다.

**Cost-model R²**: `latency ~ Σn` 단독 vs `+ Σn_i(c_i+n_i/2)` vs 충분통계량 5개, 전체
config와 `c ≥ c*` subset 각각. 후자에서 `Σn` 단독 R²가 낮고 shape-aware 모델의 R²가
높으면 가설이 성립한다.

**주의**: 분석적 c\*는 `c + (n+1)/2` (attended context) 기준이라 cached-c 축에서는
B별로 `(n+1)/2`만큼 이동한다(B=1이면 4K). Exp0의 실측 c\*는 cached-c 기준이다.
