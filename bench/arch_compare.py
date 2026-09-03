"""Analytic comparison of architectures (MHA / GQA / MLA / MoE): how the
attention-vs-GEMM balance moves with (c, n). GPU-free.

  python -m bench.arch_compare --models qwen1.5-1.8b qwen2.5-3b qwen1.5-moe-a2.7b deepseek-v2-lite \
      --gpu-mem-gib 80 --outdir results

Writes arch_compare.png (attention FLOP share and attention AI vs c, per model,
one line per n) and arch_compare.txt (c*(n) table, KV/token, capacity).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.metrics import MODEL_PRESETS, estimate_flops_bytes

DEFAULT = ["qwen1.5-1.8b", "qwen3-1.7b", "qwen2.5-3b", "qwen1.5-moe-a2.7b", "mla-dense-1.8b", "deepseek-v2-lite"]
NS = [64, 256, 1024, 8192]
CS = [0] + [2 ** k for k in range(8, 18)]          # 0, 256 … 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--gpu-mem-gib", type=float, default=80)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    M = len(args.models)
    fig, axes = plt.subplots(2, M, figsize=(4.4 * M, 8), squeeze=False)
    lines = [f"{'model':28s} {'arch':8s} {'params':>7s} {'active':>7s} {'KV/tok':>7s} {'KV cap@80GiB':>13s} "
             + " ".join(f"{'c*(n=' + str(n) + ')':>12s}" for n in NS)]
    for j, key in enumerate(args.models):
        spec = MODEL_PRESETS[key]
        cap = spec.kv_capacity_tokens(args.gpu_mem_gib, args.gpu_mem_util)
        lines.append(f"{spec.name:28s} {spec.arch_label:8s} {spec.total_params/1e9:6.2f}B "
                     f"{spec.linear_params_active/1e9:6.2f}B {spec.kv_bytes_per_token/1024:5.0f}KB "
                     f"{cap:>13,d} " + " ".join(f"{spec.attn_crossover_ctx(n):12,.0f}" for n in NS))
        for n in NS:
            share, ai = [], []
            for c in CS:
                e = estimate_flops_bytes(spec, [(n, c)])
                share.append(e["est_attn_flop_frac"]); ai.append(e["est_ai_attn"])
            x = [max(c, 128) for c in CS]
            axes[0][j].plot(x, share, "o-", ms=3, label=f"n={n}")
            axes[1][j].plot(x, ai, "o-", ms=3, label=f"n={n}")
        axes[0][j].axhline(0.5, color="gray", ls=":")
        axes[0][j].set_title(f"{spec.name.split('/')[-1]}\n{spec.arch_label}, KV {spec.kv_bytes_per_token/1024:.0f} KB/tok")
        axes[0][j].set_ylabel("attention share of FLOPs"); axes[0][j].set_ylim(0, 1)
        axes[1][j].set_ylabel("attention AI [FLOP/byte]"); axes[1][j].set_yscale("log")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log", base=2); ax.set_xlabel("c (cached context; c=0 at 2⁷)")
            ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("Analytic attention-vs-GEMM balance by architecture (single request, prefill path)")
    fig.tight_layout(); fig.savefig(outdir / "arch_compare.png", dpi=150)
    (outdir / "arch_compare.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {outdir}/arch_compare.png, arch_compare.txt")


if __name__ == "__main__":
    main()
