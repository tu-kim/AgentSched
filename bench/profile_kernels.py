"""Kernel-level profiling for selected configs (torch.profiler / CUPTI).

Reruns a subset of configs with torch.profiler around the measured step and
records per-kernel CUDA time, separating attention kernels (FlashAttention)
from GEMMs. Much slower than runner.py — use on the configs you care about.

  python -m bench.profile_kernels --model ... --exp exp2 --match cv
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from bench.configs import all_configs
from bench.metrics import ModelSpec
from bench import runner as R

ATTN_KEYS = ("flash", "fmha", "attn", "mha")
GEMM_KEYS = ("gemm", "cutlass", "matmul", "nvjet", "s16816", "wgmma")


def classify(name):
    n = name.lower()
    if any(k in n for k in ATTN_KEYS):
        return "attention"
    if any(k in n for k in GEMM_KEYS):
        return "gemm"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--exp", nargs="+", default=["exp2", "exp3", "exp4"])
    ap.add_argument("--match", default="", help="substring filter on config name")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--out", default="results/kernels.jsonl")
    args = ap.parse_args()
    args.kv_capacity_tokens = 0
    args.trials, args.warmup = 1, 1

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_V1", "1")
    import torch
    from torch.profiler import profile, ProfilerActivity
    from vllm import SamplingParams, TokensPrompt

    cfgs = [c for c in all_configs()
            if c.exp in args.exp and args.match in c.name]
    max_ctx = max(max(n + c for n, c in c.pairs) for c in cfgs) + R.MAX_MODEL_LEN_PAD
    engine = R.build_engine(args, max_ctx,
                            max(c.batch_size for c in cfgs),
                            max(sum(c.ns) for c in cfgs) + 4096)
    spec = ModelSpec.from_vllm(engine)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(1234)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a") as f:
        for cfg in cfgs:
            try:
                engine.reset_prefix_cache()
            except Exception:
                pass
            prefixes = []
            for i, (n, c) in enumerate(cfg.pairs):
                pref = R.rand_tokens(rng, c, spec.vocab)
                prefixes.append(pref)
                if c:
                    engine.add_request(f"w{i}", TokensPrompt(prompt_token_ids=pref), sp)
            R.drain(engine)

            # warmup pass
            for i, (n, c) in enumerate(cfg.pairs):
                engine.add_request(f"wu{i}", TokensPrompt(
                    prompt_token_ids=prefixes[i] + R.rand_tokens(rng, n, spec.vocab)), sp)
            R.drain(engine)

            for i, (n, c) in enumerate(cfg.pairs):
                engine.add_request(f"p{i}", TokensPrompt(
                    prompt_token_ids=prefixes[i] + R.rand_tokens(rng, n, spec.vocab)), sp)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA],
                         record_shapes=False) as prof:
                engine.step()
                torch.cuda.synchronize()
            R.drain(engine)

            buckets = {"attention": 0.0, "gemm": 0.0, "other": 0.0}
            attn_kernels = {}
            for ev in prof.key_averages():
                dur_us = getattr(ev, "device_time_total", 0) or getattr(ev, "cuda_time_total", 0)
                if dur_us <= 0:
                    continue
                cls = classify(ev.key)
                buckets[cls] += dur_us
                if cls == "attention":
                    attn_kernels[ev.key] = attn_kernels.get(ev.key, 0) + dur_us
            rec = dict(exp=cfg.exp, name=cfg.name, **cfg.stats(),
                       kernel_time_us=buckets,
                       attn_kernels_us=dict(sorted(attn_kernels.items(),
                                                   key=lambda x: -x[1])[:8]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            tot = sum(buckets.values())
            print(f"[prof] {cfg.exp}/{cfg.name}: total={tot/1e3:.1f}ms "
                  f"attn={buckets['attention']/tot*100:.0f}% "
                  f"gemm={buckets['gemm']/tot*100:.0f}%")


if __name__ == "__main__":
    main()
