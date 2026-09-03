"""GPU-free checks that the attention / MoE measurement is set up correctly.

  python -m bench.selftest

Two things are verified:

1. **Kernel classification** — every kernel name vLLM 0.28 actually launches on
   A100/H100 for the model families we benchmark is mapped to the intended
   bucket by runner.classify(). Names come from a source read of vLLM 0.28.0,
   vllm-flash-attention and cuBLAS/CUTLASS naming families; substring matching
   is order-sensitive (FA3 names contain "cutlass::device_kernel", the KV-write
   kernel contains "flash", the MLA gather contains "cache"), so a reordering
   or a new key silently moving time between buckets is caught here.

2. **Analytic FLOPs / bytes** — the model in metrics.py is cross-checked against
   hand derivations for MHA attention, MLA decompression and MoE expert weight
   traffic. These are the quantities the achieved-TFLOPS and AI columns are
   built from, so an error here misreports every measurement.

What this canNOT check without a GPU: that the profiler actually attributes the
time we think it does. runner.py's --kernel-profile therefore also emits a
per-shape warning when a bucket that must be non-empty is empty, or when the
unclassified "other" bucket is large; check those on the first real run.
"""
import sys

from bench.metrics import MODEL_PRESETS, estimate_flops_bytes
from bench.runner import classify, validate_buckets

# (kernel name as torch.profiler shows it, expected bucket, why it is tricky)
KERNEL_CASES = [
    # ---- dense attention. On A100 vLLM always passes a block_table, so FA2 takes
    # the split-KV kernel even for prefill; matching only "flash_fwd_kernel" would
    # miss ALL dense attention time.
    ("void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false,"
     " cutlass::bfloat16_t>, false, true, false, false, true, false>(flash::Flash_fwd_params)",
     "attention", "A100 dense FA2 (paged KV)"),
    ("void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 8>, 4, false>"
     "(flash::Flash_fwd_params)", "attention", "split-KV combine"),
    ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<192, 128, 64, 8, false, false,"
     " cutlass::bfloat16_t>, true, false>(flash::Flash_fwd_params)",
     "attention", "MLA prefill FA2 (non-paged, head_dim 192)"),
    ("void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<"
     "flash::CollectiveMainloopFwdSm90<...>, flash::CollectiveEpilogueFwd<...>,"
     " flash::VarlenDynamicPersistentTileScheduler<...>>>>(...)",
     "attention", "H100 FA3 — name contains cutlass::device_kernel, must beat the GEMM keys"),
    ("void cutlass::device_kernel<flash::FlashAttnFwdCombine<...>>(...)", "attention", "FA3 combine"),
    ("void pytorch_flash::flash_fwd_kernel<...>(...)", "attention", "torch SDPA fallback"),
    ("void vllm::merge_attn_states_kernel<c10::BFloat16, 128>(...)", "attention", "MLA / cascade LSE merge"),
    ("void vllm::gather_and_maybe_dequant_cache_page<__nv_bfloat16, __nv_bfloat16, 0>(...)",
     "attention", "MLA context gather — contains 'cache' but is not a KV write"),
    ("flash::prepare_varlen_num_blocks_kernel(...)", "attention", "FA3 scheduler prep"),
    ("_fwd_grouped_kernel_stage1", "attention", "Triton MLA decode (should not appear in prefill)"),
    ("_fwd_kernel_stage2", "attention", "Triton MLA decode stage 2"),

    # ---- KV cache writes: contain 'flash'/'mla', must not be counted as attention
    ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16,"
     " (vllm::Fp8KVCacheDataType)0>(...)", "kvcache", "dense KV write — contains 'flash'"),
    ("void vllm::concat_and_cache_mla_kernel<__nv_bfloat16, __nv_bfloat16,"
     " (vllm::Fp8KVCacheDataType)0>(...)", "kvcache", "MLA latent write — contains '_mla'"),

    # ---- MoE
    ("fused_moe_kernel", "moe", "Triton expert GEMM (the bulk of MoE time)"),
    ("void vllm::moe_align_block_size_kernel<int>(...)", "moe", ""),
    ("void vllm::count_and_sort_expert_tokens_kernel<int>(...)", "moe", ""),
    ("void vllm::moe_align_block_size_small_batch_expert_kernel<int>(...)", "moe", ""),
    ("void vllm::moe_sum_kernel<c10::BFloat16, 4>(...)", "moe", ""),
    ("void vllm::topk_softmax_kernel<...>(...)", "moe", "Qwen2-MoE router"),
    ("void tensorrt_llm::kernels::topkGatingSoftmax<...>(...)", "moe", "fused router variant"),

    # ---- GEMM families (cuBLAS / CUTLASS) seen on A100 and H100
    ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn", "gemm", "A100 cuBLAS"),
    ("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32_stage3_warpsize2x2x1", "gemm", "A100 xmma"),
    ("nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN", "gemm", "cuBLAS >= 12.6 nvJet"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_32x5_tn_align8>(...)",
     "gemm", "CUTLASS 2.x"),
    ("void splitKreduce_kernel<...>(...)", "gemm", "cuBLAS split-K reduction"),

    # ---- deliberately unclassified: elementwise / norm / rope / sampling.
    # These are real time but not attributable to attention or MoE by name alone
    # (act_and_mul is used by both dense MLP and MoE experts).
    ("void vllm::rms_norm_kernel<c10::BFloat16>(...)", "other", ""),
    ("void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(...)", "other", ""),
    ("void vllm::rotary_embedding_kernel<c10::BFloat16, true>(...)", "other", ""),
    ("void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(...)",
     "other", "shared by dense MLP and MoE experts — not attributable by name"),
    ("void at::native::elementwise_kernel<128, 2, ...>(...)", "other", "V pad / concat copies"),
    ("triton_poi_fused_mul_sum_0", "other", "inductor fusion (grouped_topk, enforce_eager=False)"),
]


def check_classifier():
    bad = [(name, exp, classify(name), why) for name, exp, why in KERNEL_CASES
           if classify(name) != exp]
    for name, exp, got, why in bad:
        print(f"  FAIL [{exp} -> {got}] {name[:110]}" + (f"\n        ({why})" if why else ""))
    by_bucket = {}
    for name, exp, _ in KERNEL_CASES:
        by_bucket[exp] = by_bucket.get(exp, 0) + 1
    print(f"  {len(KERNEL_CASES) - len(bad)}/{len(KERNEL_CASES)} kernel names classified as intended "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(by_bucket.items()))})")
    return not bad


def check_runtime_validation():
    """The warnings runner.profile_step emits on a real GPU run must fire for the
    failure modes that would silently corrupt the attention / MoE split."""
    B = lambda a=1.0, g=1.0, m=0.0, k=0.1, o=0.1: dict(
        attention=a, gemm=g, moe=m, kvcache=k, other=o)
    cases = [
        ("healthy dense", B(), False, False),
        ("healthy MoE", B(m=2.0), True, False),
        ("attention bucket empty", B(a=0.0), False, True),
        ("MoE model, empty moe bucket", B(), True, True),
        ("dense model, moe bucket non-empty", B(m=1.0), False, True),
        ("other > 25%", B(o=5.0), False, True),
        ("nothing captured", dict(attention=0.0, gemm=0.0, moe=0.0, kvcache=0.0, other=0.0), False, True),
    ]
    ok = True
    for label, buckets, is_moe, want_warn in cases:
        warns = validate_buckets(buckets, is_moe)
        hit = bool(warns) == want_warn
        ok &= hit
        print(f"  {'ok  ' if hit else 'FAIL'} {label:36s} -> {warns[0][:60] if warns else 'no warning'}")
    return ok


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b))


def check_analytic():
    ok = True

    # 1. MHA attention FLOPs = 4·H·d·L · Σ n_i(c_i + (n_i+1)/2)
    s = MODEL_PRESETS["qwen1.5-1.8b"]
    n, c, B = 1024, 32768, 8
    hand = B * 4 * s.n_heads * s.head_dim * s.n_layers * n * (c + (n + 1) / 2)
    got = estimate_flops_bytes(s, [(n, c)] * B)["est_flops_attn"]
    ok &= _close(hand, got, 1e-12)
    print(f"  MHA attention FLOPs @ B=8,n=1024,c=32K: {got:.4e} (hand {hand:.4e}) "
          f"{'ok' if _close(hand, got, 1e-12) else 'FAIL'}")

    # 2. attention bytes: Q read + O write + K,V read + new K,V write, all × L
    e = estimate_flops_bytes(s, [(n, c)] * B)
    hand_bytes = B * s.n_layers * (2 * n * s.n_heads * s.head_dim * 2
                                   + 2 * (c + n) * s.n_kv_heads * s.head_dim * 2
                                   + 2 * n * s.n_kv_heads * s.head_dim * 2)
    ok &= _close(hand_bytes, e["est_attn_bytes"], 1e-12)
    print(f"  MHA attention bytes: {e['est_attn_bytes']/1e9:.2f} GB (hand {hand_bytes/1e9:.2f}) "
          f"{'ok' if _close(hand_bytes, e['est_attn_bytes'], 1e-12) else 'FAIL'}")

    # 3. the README formula: for MHA/bf16, ignoring causal and the new-KV write,
    #    AI_attn == Σ n(n+c) / Σ (2n+c) exactly (coefficient 1)
    pairs = [(1024, 32768), (256, 4096), (4096, 131072)]
    L, H, d = s.n_layers, s.n_heads, s.head_dim
    flops_nc = sum(4 * H * d * L * nn * (nn + cc) for nn, cc in pairs)
    bytes_nowrite = sum(L * (2 * nn * H * d * 2 + 2 * (cc + nn) * H * d * 2) for nn, cc in pairs)
    formula = sum(nn * (nn + cc) for nn, cc in pairs) / sum(2 * nn + cc for nn, cc in pairs)
    ok &= _close(flops_nc / bytes_nowrite, formula, 1e-12)
    print(f"  README AI formula (non-causal, no KV write): {flops_nc/bytes_nowrite:.3f} "
          f"vs Σn(n+c)/Σ(2n+c) = {formula:.3f} "
          f"{'ok' if _close(flops_nc/bytes_nowrite, formula, 1e-12) else 'FAIL'}")

    # 4. MLA: latent KV, decompression ∝ Σc, padded-V attention core
    m = MODEL_PRESETS["deepseek-v2-lite"]
    checks = [
        ("KV bytes/token (latent)", m.kv_bytes_per_token,
         m.n_layers * (m.kv_lora_rank + m.qk_rope_head_dim) * 2),
        ("decompress FLOPs/cached token", m.mla_decompress_flops_per_ctx,
         2 * m.n_layers * m.kv_lora_rank * m.n_heads * (m.qk_nope_head_dim + m.v_head_dim)),
        ("attn FLOPs per (q,ctx) pair", m.attn_flops_per_token_per_ctx,
         2 * m.n_layers * m.n_heads * (192 + 192)),      # V zero-padded 128 -> 192 on FA2
    ]
    for label, got_v, hand_v in checks:
        ok &= _close(got_v, hand_v, 1e-12)
        print(f"  MLA {label}: {got_v:,.0f} (hand {hand_v:,.0f}) "
              f"{'ok' if _close(got_v, hand_v, 1e-12) else 'FAIL'}")
    # decompression must scale with Σc only, not with n
    a = estimate_flops_bytes(m, [(64, 65536)])["est_flops_attn"]
    b = estimate_flops_bytes(m, [(64, 0)])["est_flops_attn"]
    per_ctx = (a - b) / 65536 - m.attn_flops_per_token_per_ctx * 64
    ok &= _close(per_ctx, m.mla_decompress_flops_per_ctx, 1e-9)
    print(f"  MLA decompression is n-independent (∝ Σc): {per_ctx/1e6:.1f} MFLOP/cached token "
          f"{'ok' if _close(per_ctx, m.mla_decompress_flops_per_ctx, 1e-9) else 'FAIL'}")

    # 5. MoE: active params (top_k + shared) and the expert-weight read floor.
    #    Σn=8192 ≫ E/top_k, so every expert receives tokens and its weights are read.
    for key, n_moe_layers in [("qwen1.5-moe-a2.7b", 24), ("deepseek-v2-lite", 26)]:
        sp = MODEL_PRESETS[key]
        hand_active = (sp.n_layers * sp.attn_proj_params
                       + (sp.n_layers - n_moe_layers) * 3 * sp.hidden * sp.inter
                       + n_moe_layers * (sp.top_k * 3 * sp.hidden * sp.moe_inter
                                         + 3 * sp.hidden * sp.shared_inter
                                         + sp.hidden * sp.n_experts))
        ok &= sp.n_moe_layers == n_moe_layers and _close(sp.linear_params_active, hand_active, 1e-12)
        hand_expert_bytes = n_moe_layers * sp.n_experts * 3 * sp.hidden * sp.moe_inter * 2
        ef = estimate_flops_bytes(sp, [(1024, 0)] * 8)
        dense_part = (sp.linear_params_total - n_moe_layers * sp.n_experts * 3 * sp.hidden * sp.moe_inter) * 2
        implied_expert_bytes = ef["est_weight_bytes"] - dense_part - sp.hidden * sp.vocab * 2 - 8192 * sp.hidden * 2
        hit = _close(implied_expert_bytes, hand_expert_bytes, 1e-3)
        ok &= hit
        print(f"  MoE {key}: active {sp.linear_params_active/1e9:.3f}B (hand {hand_active/1e9:.3f}B), "
              f"expert weight read {implied_expert_bytes/1e9:.2f} GB (hand {hand_expert_bytes/1e9:.2f}) "
              f"{'ok' if hit else 'FAIL'}")

    # 6. MoE expert traffic must be ~Σn-independent once every expert is hit,
    #    i.e. it is a floor, not a per-token cost.
    sp = MODEL_PRESETS["qwen1.5-moe-a2.7b"]
    w_small = estimate_flops_bytes(sp, [(1024, 0)])["est_weight_bytes"]
    w_big = estimate_flops_bytes(sp, [(1024, 0)] * 8)["est_weight_bytes"]
    ratio = w_big / w_small
    ok &= ratio < 1.05
    print(f"  MoE weight traffic Σn=8192 vs Σn=1024: ×{ratio:.3f} (floor, not ∝Σn) "
          f"{'ok' if ratio < 1.05 else 'FAIL'}")
    return ok


def main():
    print("1. kernel classification (vLLM 0.28 names)")
    ok1 = check_classifier()
    print("\n2. analytic FLOPs / bytes vs hand derivation")
    ok2 = check_analytic()
    print("\n3. runtime bucket validation (what --kernel-profile warns about)")
    ok3 = check_runtime_validation()
    print(f"\n{'PASS' if ok1 and ok2 and ok3 else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
