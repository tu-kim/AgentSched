"""Synthetic results/raw.jsonl with the exact schema runner.py writes, from a toy
latency model (GEMM at 55% of peak, attention roofline at 40% / 80% BW, +10% per
unit CV). For exercising bench.analyze without a GPU — NOT measurements.

  python -m bench.synth_results --out results/synth.jsonl --models qwen1.5-1.8b mla-dense-1.8b
  python -m bench.analyze --raw results/synth.jsonl --outdir results/synth
"""
import argparse
import json
import random
from pathlib import Path

from bench.configs import all_configs, group_by_shape, resolve_base_c
from bench.metrics import MODEL_PRESETS, DEVICE_PEAKS, estimate_flops_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synth.jsonl")
    ap.add_argument("--models", nargs="+", default=["qwen1.5-1.8b"])
    ap.add_argument("--device", default="A100-80GB")
    ap.add_argument("--gpu-mem-gib", type=float, default=80)
    args = ap.parse_args()
    peak, bw = DEVICE_PEAKS[args.device]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for key in args.models:
            spec = MODEL_PRESETS[key]
            cap = spec.kv_capacity_tokens(args.gpu_mem_gib)
            base_c = resolve_base_c("auto", spec)
            rng = random.Random(0)
            cstar = {n: spec.attn_crossover_ctx(n) for n in (64, 1024, 8192)}
            meta = dict(model=spec.name, model_arch=spec.arch_label, model_is_mha=spec.is_mha,
                        model_is_moe=spec.is_moe, model_attn_type=spec.attn_type,
                        model_kv_bytes_per_token=spec.kv_bytes_per_token, model_params=spec.total_params,
                        model_active_params=spec.linear_params_active, model_attn_crossover_ctx=cstar[1024],
                        model_attn_crossover_by_n=cstar, base_c=base_c, tp=1, kv_capacity_tokens=cap,
                        attention_backend="synthetic", flash_attn_version=None, enforce_eager=False,
                        kv_cache_dtype="auto", load_format="synthetic")
            for aliases in group_by_shape(all_configs(base_c=base_c)).values():
                cfg = aliases[0]
                s = cfg.stats()
                if s["kv_tokens_needed"] > cap * 0.95:
                    res = dict(skipped=True, reason=f"kv_tokens_needed {s['kv_tokens_needed']} > 0.95×capacity {cap}")
                else:
                    e = estimate_flops_bytes(spec, cfg.pairs)
                    t_gemm = e["est_flops_linear"] / (0.55 * peak) + e["est_weight_bytes"] / bw
                    t_attn = max(e["est_flops_attn"] / (0.40 * peak), e["est_attn_bytes"] / (0.8 * bw))
                    t_attn *= 1 + 0.10 * (s["cv_n"] + s["cv_c"])
                    lat = (t_gemm + t_attn + 3e-4) * rng.uniform(0.98, 1.02)
                    lats = sorted(lat * rng.uniform(0.97, 1.03) for _ in range(5))
                    res = dict(skipped=False, **s, **e, pairs=cfg.pairs, latency_median_s=lat, latency_all_s=lats,
                               throughput_tok_s=s["token_budget"] / lat,
                               achieved_tflops=e["est_flops_total"] / lat / 1e12,
                               est_hbm_gbps=e["est_bytes_analytic"] / lat / 1e9,
                               gpu_util=rng.uniform(85, 100), mem_util=rng.uniform(30, 90), power_w=rng.uniform(250, 400),
                               co_scheduled=True, cache_hits_ok=True, extra_steps_max=0,
                               cached_tokens=s["sum_c"], cached_tokens_expected=s["sum_c"],
                               kernel_time_attention_us=t_attn * 1e6, kernel_time_gemm_us=t_gemm * 1e6,
                               kernel_time_moe_us=0.0, kernel_time_kvcache_us=50.0, kernel_time_other_us=300.0,
                               kernel_top=[])
                res.update(wall_s=0.0, **meta)
                for a in aliases:
                    rec = dict(res, exp=a.exp, name=a.name, group=a.group)
                    if res.get("skipped"):
                        rec.update(a.stats())
                    out.write(json.dumps(rec) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
