"""vLLM batch-shape benchmark driver.

Drives vllm.LLMEngine.step() directly so each measured configuration is a
single scheduler iteration whose (n_i, c_i) composition we control exactly:

  Phase A (warm): per-request unique random prefix of c_i tokens is prefilled
                  once so its KV lands in the prefix cache (c_i % 16 == 0).
  Phase B (meas): requests with prompt = prefix_i + n_i fresh random tokens,
                  max_tokens=1. The scheduler cache-hits c_i tokens and
                  computes exactly n_i new tokens per request.

Engine settings that make "one step() == one iteration" hold (vLLM ≥ 0.28):
  VLLM_ENABLE_V1_MULTIPROCESSING=0  → EngineCore in-process (InprocClient)
  async_scheduling=False            → step() blocks until the forward finishes
  attention_backend=FLASH_ATTN      → FA2 on A100 / FA3 on H100 (MLA models: auto)
  hf_overrides.max_position_embeddings ≥ max(c+n)  → RoPE table large enough

Typical flow (A100 80GB, single GPU, no EP):
  python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp0 --kernel-profile
  python -m bench.analyze                       # → c* in results/report.txt
  python -m bench.runner --model Qwen/Qwen1.5-1.8B --exp exp1 exp2 exp3 exp4 \
         --base-c 16384 --kernel-profile

Requires: vllm >= 0.28, torch, pynvml, transformers.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

# Must be set before vllm is imported/constructed: with the default (1) the
# EngineCore runs in a child process with its own busy loop and step() only
# dequeues outputs, so wall-clock around step() is not one iteration.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from bench.configs import (all_configs, group_by_shape, resolve_base_c, BLOCK_SIZE,
                           GENERATORS)
from bench.metrics import ModelSpec, estimate_flops_bytes, NVMLSampler

MAX_MODEL_LEN_PAD = 256

# torch.profiler kernel classification (vLLM 0.28, verified names). Order matters:
# FA3 names contain "cutlass::device_kernel", the KV-write kernel contains "flash".
#   dense FA2 (paged KV)   : flash::flash_fwd_splitkv_kernel<...>      (A100)
#   dense FA3              : cutlass::device_kernel<flash::...FlashAttnFwdSm90...>  (H100)
#   MLA prefill            : flash::flash_fwd_kernel (non-paged), merge_attn_states_kernel,
#                            gather_and_maybe_dequant_cache_page; kv_b_proj decompression is a
#                            cuBLAS GEMM and lands in "gemm" (analytic model attributes it to attention)
#   KV write               : vllm::reshape_and_cache_flash_kernel / concat_and_cache_mla_kernel
#   MoE                    : Triton fused_moe_kernel + moe_align_block_size / topk / moe_sum kernels
KVCACHE_KEYS = ("reshape_and_cache", "concat_and_cache", "_and_cache")
ATTN_KEYS = ("flashattnfwd", "flash::", "flash_fwd", "pytorch_flash::", "merge_attn_states",
             "prepare_varlen_num_blocks", "gather_and_maybe_dequant", "fwd_grouped_kernel",
             "fwd_kernel_stage", "_mla", "fmha", "attn_fwd", "paged_attention")
MOE_KEYS = ("fused_moe", "moe_align", "moe_sum", "count_and_sort_expert", "topk_softmax",
            "topkgating", "moesoftmax", "moetopk", "grouped_topk", "moe_")
GEMM_KEYS = ("nvjet", "xmma", "gemm", "gemv", "cutlass3x", "kernel2<", "cutlass::kernel<",
             "cutlass_80", "ampere_", "sm80_", "splitkreduce", "cublas", "wgmma", "s16816", "matmul")


def classify(name):
    n = name.lower()
    if any(k in n for k in KVCACHE_KEYS):
        return "kvcache"
    if any(k in n for k in ATTN_KEYS):
        return "attention"
    if any(k in n for k in MOE_KEYS):
        return "moe"
    if any(k in n for k in GEMM_KEYS):
        return "gemm"
    return "other"


def build_engine(args, spec, max_model_len, max_num_seqs, max_num_batched_tokens):
    from vllm import EngineArgs, LLMEngine
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}
    if spec.max_position and max_model_len > spec.max_position:
        # enlarges both the derived max_model_len and the RoPE cos/sin table;
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN alone would leave the table too small
        hf_overrides.setdefault("max_position_embeddings", max_model_len)
        print(f"[engine] max_model_len {max_model_len} > native {spec.max_position}: "
              f"hf_overrides.max_position_embeddings={hf_overrides['max_position_embeddings']}")
    backend = args.attention_backend
    if backend == "auto":
        backend = None if spec.attn_type == "mla" else "FLASH_ATTN"
        if backend and "fp8" in args.kv_cache_dtype.lower():
            import torch
            if torch.cuda.get_device_capability()[0] < 9:
                # FA2 (pre-Hopper) has no fp8 KV path; let vLLM pick a backend that does
                print("[engine] fp8 KV cache on a pre-SM90 GPU: leaving attention backend to auto-select")
                backend = None
    kw = dict(
        model=args.model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        disable_cascade_attn=True,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
        load_format=args.load_format,
        tensor_parallel_size=args.tp,
        skip_tokenizer_init=True,      # we feed token ids; no tokenizer needed
        disable_log_stats=True,
        seed=0,
        trust_remote_code=args.trust_remote_code,
    )
    if backend:
        kw["attention_backend"] = backend
    if args.kv_cache_dtype:
        kw["kv_cache_dtype"] = args.kv_cache_dtype
    if hf_overrides:
        kw["hf_overrides"] = hf_overrides
    engine = LLMEngine.from_engine_args(EngineArgs(**kw))
    cc = engine.vllm_config.cache_config
    assert cc.block_size == BLOCK_SIZE, f"engine block_size {cc.block_size} != {BLOCK_SIZE}"
    try:                                        # the backend vLLM actually resolved
        resolved = engine.vllm_config.attention_config.backend
        resolved = getattr(resolved, "name", None) or str(resolved)
    except Exception:
        resolved = backend or "auto"
    return engine, resolved


def kv_capacity_tokens(engine):
    """Usable KV capacity in tokens (num_gpu_blocks includes one null block)."""
    cc = engine.vllm_config.cache_config
    nb = getattr(cc, "num_gpu_blocks", None)
    return (int(nb) - 1) * int(cc.block_size) if nb else None


def flash_attn_version():
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        return get_flash_attn_version()
    except Exception:
        return None


def rand_tokens(rng, k, vocab):
    return [rng.randrange(10, vocab - 10) for _ in range(k)]


def drain(engine):
    outs = []
    while engine.has_unfinished_requests():
        outs.extend(engine.step())
    return outs


def validate_buckets(buckets, is_moe):
    """Warnings about a profiled step's kernel classification (pure; unit-tested
    in bench/selftest.py). Empty means the split looks plausible for this model."""
    total = sum(buckets.values())
    if total <= 0:
        return ["no CUDA kernels captured"]
    warn = []
    if buckets["attention"] <= 0:
        warn.append("attention bucket is EMPTY — no kernel matched ATTN_KEYS; "
                    "attention time is being counted elsewhere")
    if is_moe and buckets["moe"] <= 0:
        warn.append("MoE model but moe bucket is EMPTY — expert-GEMM time is being counted elsewhere")
    if not is_moe and buckets["moe"] > 0:
        warn.append("dense model but moe bucket is non-empty — a MOE_KEYS substring is over-matching")
    if buckets["other"] > 0.25 * total:
        warn.append(f"unclassified 'other' is {buckets['other']/total:.0%} of kernel time")
    return warn


def profile_step(engine, spec, warn_state):
    """Run one engine.step() under torch.profiler (CUDA activity only); return
    per-class kernel time [µs], the top kernels, and the top unclassified ones.

    Also validates the classification: buckets that must carry time for this
    model (attention always; moe for MoE models) and the size of the "other"
    bucket are checked, so a kernel name this build does not match shows up as a
    warning instead of silently moving time between buckets. `warn_state` is a
    set used to print each distinct warning only once per run.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        engine.step()
        torch.cuda.synchronize()
    buckets = {"attention": 0.0, "gemm": 0.0, "moe": 0.0, "kvcache": 0.0, "other": 0.0}
    per_kernel = {}
    for ev in prof.key_averages():
        us = getattr(ev, "self_device_time_total", 0) or 0
        if us <= 0:
            continue
        buckets[classify(ev.key)] += us
        per_kernel[ev.key] = per_kernel.get(ev.key, 0.0) + us
    total = sum(buckets.values())
    if total <= 0:
        print("[profile] WARNING: no CUDA kernels captured (CUPTI unavailable?); kernel times omitted")
        return None

    unclassified = sorted(((k, v) for k, v in per_kernel.items() if classify(k) == "other"),
                          key=lambda kv: -kv[1])
    for w in validate_buckets(buckets, spec.is_moe):
        if w not in warn_state:
            warn_state.add(w)
            print(f"[profile] WARNING: {w}")
            for k, v in unclassified[:5]:
                print(f"[profile]   unclassified {v/1e3:8.2f} ms  {k[:120]}")
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:15]
    return (buckets,
            [(k[:160], round(v, 1), classify(k)) for k, v in top],
            [(k[:160], round(v, 1)) for k, v in unclassified[:10]])


def run_shape(engine, spec, cfg, args, rng, capacity, sampler, warn_state):
    """Measure one batch shape; returns a result dict (or a skip record)."""
    from vllm import SamplingParams, TokensPrompt
    import torch
    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    vocab = spec.vocab
    s = cfg.stats()

    if capacity and s["kv_tokens_needed"] > capacity * args.capacity_margin:
        return dict(skipped=True, reason=f"kv_tokens_needed {s['kv_tokens_needed']} > "
                                         f"{args.capacity_margin:.2f}×capacity {capacity}")

    assert engine.reset_prefix_cache(), "reset_prefix_cache failed (blocks still held)"

    # ---- Phase A: warm prefix caches
    prefixes = []
    for i, (n, c) in enumerate(cfg.pairs):
        assert c % BLOCK_SIZE == 0
        pref = rand_tokens(rng, c, vocab)
        prefixes.append(pref)
        if c > 0:
            engine.add_request(f"warm-{i}", TokensPrompt(prompt_token_ids=pref), sp)
    drain(engine)

    # ---- Phase B: measured iterations (fresh suffix each trial)
    def submit(tag):
        expected = {}
        for i, (n, c) in enumerate(cfg.pairs):
            rid = f"m-{tag}-{i}"
            toks = prefixes[i] + rand_tokens(rng, n, vocab)
            engine.add_request(rid, TokensPrompt(prompt_token_ids=toks), sp)
            expected[rid] = c
        return expected

    results = []
    for trial in range(args.warmup + args.trials):
        expected = submit(trial)
        torch.cuda.synchronize()
        with sampler as nv:
            t0 = time.perf_counter()
            outs = list(engine.step())          # the measured iteration
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            n_first = len(outs)
            extra_steps = 0
            while engine.has_unfinished_requests():
                outs.extend(engine.step())
                extra_steps += 1
        hits = {o.request_id: o.num_cached_tokens for o in outs}
        hits_ok = all(hits.get(rid) == c for rid, c in expected.items())
        if trial >= args.warmup:
            results.append(dict(
                latency_s=t1 - t0, extra_steps=extra_steps, nvml=nv.summary(t0, t1),
                cached_tokens=sum(v for v in hits.values() if v is not None),
                hits_ok=hits_ok, n_outputs_first_step=n_first,
            ))

    kernels = None
    if args.kernel_profile:
        submit("prof-warm")
        profile_step(engine, spec, warn_state)      # CUPTI init outside the recorded step
        drain(engine)
        submit("prof")
        kernels = profile_step(engine, spec, warn_state)
        drain(engine)

    lat = sorted(r["latency_s"] for r in results)
    med = lat[len(lat) // 2]
    est = estimate_flops_bytes(spec, cfg.pairs)
    nv_avg = lambda k: (sum(r["nvml"][k] for r in results if r["nvml"][k] is not None)
                        / max(1, len(results)))
    rec = dict(
        skipped=False, **s, **est, pairs=cfg.pairs,
        latency_median_s=med, latency_all_s=lat,
        throughput_tok_s=s["token_budget"] / med,
        achieved_tflops=est["est_flops_total"] / med / 1e12,
        est_hbm_gbps=est["est_bytes_analytic"] / med / 1e9,
        gpu_util=nv_avg("gpu_util"), mem_util=nv_avg("mem_util"), power_w=nv_avg("power_w"),
        # config-realisation checks
        co_scheduled=all(r["extra_steps"] == 0 and r["n_outputs_first_step"] == cfg.batch_size
                         for r in results),
        cache_hits_ok=all(r["hits_ok"] for r in results),
        extra_steps_max=max(r["extra_steps"] for r in results),
        cached_tokens=results[-1]["cached_tokens"],
        cached_tokens_expected=s["sum_c"],
    )
    if kernels:
        buckets, top, unclassified = kernels
        rec.update({f"kernel_time_{k}_us": v for k, v in buckets.items()},
                   kernel_top=top, kernel_unclassified=unclassified)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-1.8B",
                    help="HF id; small models keep KV/token low so c can be large")
    ap.add_argument("--exp", nargs="+", default=list(GENERATORS))
    ap.add_argument("--base-c", default="auto",
                    help="centre c for exp2-4; 'auto' = next power of two ≥ analytic c*")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--kernel-profile", action="store_true",
                    help="extra torch.profiler step per shape → attention/GEMM/MoE kernel time")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--capacity-margin", type=float, default=0.95)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--attention-backend", default="auto",
                    help="'auto' = FLASH_ATTN for MHA/GQA, engine default for MLA")
    ap.add_argument("--load-format", default="auto", help="'dummy' = random weights, no download")
    ap.add_argument("--kv-cache-dtype", default="", help="e.g. fp8 (halves KV bytes)")
    ap.add_argument("--hf-overrides", default="", help="JSON dict merged into the HF config")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", default="results/raw.jsonl")
    args = ap.parse_args()
    if args.kernel_profile and args.tp > 1:
        ap.error("--kernel-profile needs tp=1: with TP>1 the model workers are separate "
                 "processes and the driver-side torch.profiler sees none of their kernels")

    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    hf = getattr(hf, "get_text_config", lambda: hf)()
    kv_dtype_bytes = 1 if "fp8" in args.kv_cache_dtype.lower() else 2
    spec = ModelSpec.from_hf_config(hf, name=args.model, kv_dtype_bytes=kv_dtype_bytes)
    base_c = resolve_base_c(args.base_c, spec)
    cstar = {n: spec.attn_crossover_ctx(n) for n in (64, 1024, 8192)}

    print(f"[model] {spec.name}: {spec.arch_label} L={spec.n_layers} H={spec.n_heads} Hkv={spec.n_kv_heads} "
          f"d={spec.head_dim} params={spec.total_params/1e9:.2f}B "
          f"active/token={spec.linear_params_active/1e9:.2f}B KV={spec.kv_bytes_per_token/1024:.0f}KB/token")
    print(f"[model] analytic c*: " + ", ".join(f"n={n}: {v:,.0f}" for n, v in cstar.items())
          + f" → base_c={base_c}")
    if spec.is_moe and args.tp > 1:
        print("[model] NOTE: MoE with TP>1; keep expert parallelism disabled (default) for this study")

    cfgs = all_configs(args.exp, base_c)
    shapes = group_by_shape(cfgs)
    max_ctx = max(max(n + c for n, c in c.pairs) for c in cfgs) + MAX_MODEL_LEN_PAD
    max_seqs = max(c.batch_size for c in cfgs)
    max_bt = max(sum(c.ns) for c in cfgs) + BLOCK_SIZE * max_seqs
    print(f"[engine] max_model_len={max_ctx} max_num_seqs={max_seqs} max_num_batched_tokens={max_bt}")
    engine, backend = build_engine(args, spec, max_ctx, max_seqs, max_bt)
    spec_engine = ModelSpec.from_vllm(engine, kv_dtype_bytes)
    assert (spec_engine.n_layers, spec_engine.n_heads, spec_engine.n_kv_heads) == \
           (spec.n_layers, spec.n_heads, spec.n_kv_heads), "HF config / engine mismatch"

    capacity = kv_capacity_tokens(engine)
    fa_ver = flash_attn_version()
    print(f"[engine] kv_capacity_tokens={capacity if capacity else 'unknown'} "
          f"attention_backend={backend} flash_attn_version={fa_ver}")
    sampler = NVMLSampler(args.device)         # fail fast if NVML is unavailable
    n_fit = sum(1 for c in cfgs if not capacity or c.kv_tokens_needed() <= capacity * args.capacity_margin)
    print(f"[plan] {n_fit}/{len(cfgs)} configs fit the KV cache; {len(shapes)} distinct shapes")

    model_meta = dict(model=args.model, model_arch=spec.arch_label, model_is_mha=spec.is_mha,
                      model_is_moe=spec.is_moe, model_attn_type=spec.attn_type,
                      model_kv_bytes_per_token=spec.kv_bytes_per_token,
                      model_params=spec.total_params, model_active_params=spec.linear_params_active,
                      model_attn_crossover_ctx=cstar[1024], model_attn_crossover_by_n=cstar,
                      base_c=base_c, tp=args.tp, kv_capacity_tokens=capacity,
                      attention_backend=backend, flash_attn_version=fa_ver,
                      enforce_eager=args.enforce_eager, kv_cache_dtype=args.kv_cache_dtype or "auto",
                      load_format=args.load_format)

    rng = random.Random(1234)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    consecutive_errors = []
    warn_state = set()                          # each profiling warning printed once per run
    with outp.open("a") as f:
        for key, aliases in shapes.items():
            cfg = aliases[0]
            t = time.time()
            try:
                res = run_shape(engine, spec, cfg, args, rng, capacity, sampler, warn_state)
                consecutive_errors = []
            except Exception as e:                      # OOM etc. — record and continue
                res = dict(skipped=True, reason=repr(e))
                consecutive_errors.append(type(e).__name__)
                try:
                    drain(engine)
                except Exception:
                    pass
            res.update(wall_s=time.time() - t, **model_meta)
            for alias in aliases:                       # one record per (exp, name)
                rec = dict(res, exp=alias.exp, name=alias.name, group=alias.group)
                if res.get("skipped"):
                    rec.update(alias.stats())
                f.write(json.dumps(rec) + "\n")
            f.flush()
            label = "/".join(f"{a.exp}:{a.name}" for a in aliases)
            if res.get("skipped"):
                print(f"[skip] {label}: {res.get('reason')}")
                if len(consecutive_errors) >= 3 and len(set(consecutive_errors)) == 1:
                    raise SystemExit(f"aborting: {len(consecutive_errors)} consecutive shapes failed with "
                                     f"{consecutive_errors[0]} — looks systematic, not a capacity issue")
                continue
            flag = "" if res["co_scheduled"] and res["cache_hits_ok"] \
                else f"  !! co_sched={res['co_scheduled']} hits_ok={res['cache_hits_ok']} " \
                     f"cached={res['cached_tokens']}/{res['cached_tokens_expected']}"
            kp = ""
            if "kernel_time_attention_us" in res:
                tot = sum(v for k, v in res.items() if k.startswith("kernel_time_") and k.endswith("_us"))
                kp = f" attn={res['kernel_time_attention_us']/tot*100:.0f}%" if tot > 0 else " attn=n/a"
            print(f"[done] {label}: lat={res['latency_median_s']*1e3:.1f}ms "
                  f"thr={res['throughput_tok_s']:,.0f} tok/s TFLOPS={res['achieved_tflops']:.1f}{kp}{flag}")


if __name__ == "__main__":
    main()
