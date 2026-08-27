"""vLLM batch-shape benchmark driver.

Drives vllm.LLMEngine.step() directly so each measured configuration is a
single scheduler iteration whose (n_i, c_i) composition we control exactly:

  Phase A (warm): per-request unique random prefix of c_i tokens is prefilled
                  once so its KV lands in the prefix cache (c_i % 16 == 0).
  Phase B (meas): requests with prompt = prefix_i + n_i fresh random tokens,
                  max_tokens=1. The scheduler cache-hits c_i tokens and
                  computes exactly n_i new tokens per request.

Run on a CUDA machine:
  VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m bench.runner \
      --model meta-llama/Llama-3.1-8B-Instruct --exp exp1 exp2 exp3 exp4

Requires: vllm >= 0.8 (v1 engine), pynvml, torch.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from bench.configs import all_configs, BLOCK_SIZE
from bench.metrics import ModelSpec, estimate_flops_bytes, NVMLSampler

MAX_MODEL_LEN_PAD = 256


def build_engine(args, max_model_len, max_num_seqs, max_num_batched_tokens):
    from vllm import EngineArgs, LLMEngine
    ea = EngineArgs(
        model=args.model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,      # needed for large budgets; budget set high
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    return LLMEngine.from_engine_args(ea)


def rand_tokens(rng, k, vocab):
    # avoid special ids near 0
    return [rng.randrange(10, vocab - 10) for _ in range(k)]


def drain(engine):
    outs = []
    while engine.has_unfinished_requests():
        outs.extend(engine.step())
    return outs


def run_config(engine, spec, cfg, args, rng):
    """Returns a result dict for one BatchConfig, or None if infeasible."""
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    vocab = spec.vocab

    # ---- feasibility: total KV tokens must fit in cache
    total_kv = cfg.stats()["total_kv_tokens"]
    if args.kv_capacity_tokens and total_kv > args.kv_capacity_tokens * 0.95:
        return dict(skipped=True, reason=f"kv_tokens {total_kv} > capacity")

    try:
        engine.reset_prefix_cache()
    except Exception:
        pass

    # ---- Phase A: warm prefix caches (skip c_i == 0)
    prefixes = []
    for i, (n, c) in enumerate(cfg.pairs):
        assert c % BLOCK_SIZE == 0
        pref = rand_tokens(rng, c, vocab)
        prefixes.append(pref)
        if c > 0:
            engine.add_request(f"warm-{cfg.name}-{i}",
                               TokensPrompt(prompt_token_ids=pref), sp)
    drain(engine)

    import torch
    results = []
    for trial in range(args.warmup + args.trials):
        # fresh suffixes each trial so n_i tokens are never cache-hits
        for i, (n, c) in enumerate(cfg.pairs):
            toks = prefixes[i] + rand_tokens(rng, n, vocab)
            engine.add_request(f"m-{cfg.name}-{trial}-{i}",
                               TokensPrompt(prompt_token_ids=toks), sp)
        torch.cuda.synchronize()
        with NVMLSampler(args.device) as nv:
            t0 = time.perf_counter()
            step_outs = engine.step()          # the measured iteration
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            extra_steps = 0
            outs = list(step_outs)
            while engine.has_unfinished_requests():
                outs.extend(engine.step())
                extra_steps += 1
        co_scheduled = extra_steps == 0
        cached = {}
        for o in outs:
            if getattr(o, "num_cached_tokens", None) is not None:
                cached[o.request_id] = o.num_cached_tokens
        if trial >= args.warmup:
            results.append(dict(
                latency_s=t1 - t0, co_scheduled=co_scheduled,
                extra_steps=extra_steps, nvml=nv.summary(t0, t1),
                cached_tokens=sum(cached.values()) if cached else None,
            ))

    lat = sorted(r["latency_s"] for r in results)
    med = lat[len(lat) // 2]
    est = estimate_flops_bytes(spec, cfg.pairs)
    s = cfg.stats()
    n_avg = lambda k: (sum(r["nvml"][k] for r in results if r["nvml"][k] is not None)
                       / max(1, len(results)))
    return dict(
        skipped=False, **s, **est,
        pairs=cfg.pairs,
        latency_median_s=med, latency_all_s=lat,
        throughput_tok_s=s["token_budget"] / med,
        achieved_tflops=est["est_flops_total"] / med / 1e12,
        est_hbm_gbps=est["est_bytes_total"] / med / 1e9,
        gpu_util=n_avg("gpu_util"), mem_util=n_avg("mem_util"), power_w=n_avg("power_w"),
        co_scheduled=all(r["co_scheduled"] for r in results),
        cached_tokens=results[-1]["cached_tokens"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--exp", nargs="+", default=["exp1", "exp2", "exp3", "exp4"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--kv-capacity-tokens", type=int, default=0,
                    help="skip configs whose total KV tokens exceed this (0=try anyway)")
    ap.add_argument("--out", default="results/raw.jsonl")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_V1", "1")

    cfgs = [c for c in all_configs() if c.exp in args.exp]
    max_ctx = max(max(n + c for n, c in c.pairs) for c in cfgs) + MAX_MODEL_LEN_PAD
    max_seqs = max(c.batch_size for c in cfgs)
    max_bt = max(sum(c.ns) for c in cfgs) + BLOCK_SIZE * max_seqs

    print(f"[engine] max_model_len={max_ctx} max_num_seqs={max_seqs} "
          f"max_num_batched_tokens={max_bt}")
    engine = build_engine(args, max_ctx, max_seqs, max_bt)
    spec = ModelSpec.from_vllm(engine)
    # discover KV capacity if user didn't set it
    if not args.kv_capacity_tokens:
        try:
            ncache = engine.model_executor.driver_worker.cache_config.num_gpu_blocks  # type: ignore
            args.kv_capacity_tokens = ncache * BLOCK_SIZE
        except Exception:
            pass
    print(f"[engine] kv_capacity_tokens={args.kv_capacity_tokens or 'unknown'}")

    rng = random.Random(1234)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a") as f:
        for cfg in cfgs:
            t = time.time()
            try:
                res = run_config(engine, spec, cfg, args, rng)
            except Exception as e:  # OOM etc. — record and continue
                res = dict(skipped=True, reason=repr(e), **cfg.stats())
            res.update(exp=cfg.exp, name=cfg.name, model=args.model,
                       wall_s=time.time() - t)
            f.write(json.dumps(res) + "\n")
            f.flush()
            if res.get("skipped"):
                print(f"[skip] {cfg.exp}/{cfg.name}: {res.get('reason')}")
            else:
                print(f"[done] {cfg.exp}/{cfg.name}: "
                      f"lat={res['latency_median_s']*1e3:.1f}ms "
                      f"thr={res['throughput_tok_s']:,.0f} tok/s "
                      f"TFLOPS={res['achieved_tflops']:.1f} "
                      f"co_sched={res['co_scheduled']}")


if __name__ == "__main__":
    main()
