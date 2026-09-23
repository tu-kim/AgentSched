"""Analytic comparison of attention families. GPU-free.

  python -m bench.arch_compare --gpu-mem-gib 80 --outdir results
  python -m bench.arch_compare --pairs                 # controlled pairs only
  python -m bench.arch_compare --models llama2-7b falcon-7b llama3-8b gemma3-4b

Prefill and decode are plotted separately on purpose: GQA/MQA/MLA barely change
prefill score FLOPs and mostly divide the decode KV read, so a single combined
number hides the whole effect.

Writes arch_compare.png (attention FLOP share and attention AI vs context, one
column per model, prefill and decode rows) and arch_compare.txt (the family
table: KV rate, capacity, c* per phase, decode AI).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.metrics import (CONTROLLED_PAIRS, DEVICES, MODEL_PRESETS,
                           estimate_flops_bytes, for_device)

# one representative per taxonomy family that fits a single 80GB GPU
DEFAULT = ["llama2-7b", "falcon-7b", "llama3-8b", "gemma3-4b",
           "qwen1.5-moe-a2.7b", "deepseek-v2-lite"]
PREFILL_NS = [64, 1024, 8192]
CS = [0] + [2 ** k for k in range(7, 20)]          # 0, 128 … 524288


def fmt(v):
    return "inf" if v == math.inf else f"{v:,.0f}"


def table(keys, dev, gpu_mem_gib, util):
    """Family table: what each architecture changes, per phase."""
    lines = [f"{'preset':22s} {'family':11s} {'arch':16s} {'tier':12s} {'params':>8s} "
             f"{'active':>8s} {'KV/tok':>9s} {'KVcap':>12s} {'c*(dec)':>10s} {'c*(pre)':>10s} "
             f"{'AI_dec@32K':>11s}"]
    for key in keys:
        s = for_device(MODEL_PRESETS[key], dev)
        cap = s.kv_capacity_tokens(gpu_mem_gib, util)
        ai = estimate_flops_bytes(s, [(1, 32768)])["est_ai_attn"]
        lines.append(
            f"{key:22s} {s.family:11s} {s.arch_label:16s} {s.tier:12s} "
            f"{s.total_params/1e9:7.2f}B {s.linear_params_active/1e9:7.2f}B "
            f"{s.kv_bytes_per_token/1024:8.1f}K {cap:>12,d} "
            f"{fmt(s.attn_crossover_ctx(1)):>10s} {fmt(s.attn_crossover_ctx(1024)):>10s} "
            f"{ai:11.1f}")
        if s.notes:
            lines.append(f"{'':22s} └─ {s.notes}")
    return lines


def pair_table(dev, gpu_mem_gib, util):
    lines = ["", "Controlled pairs (one structural change each):"]
    for name, (a, b, why) in CONTROLLED_PAIRS.items():
        sa, sb = for_device(MODEL_PRESETS[a], dev), for_device(MODEL_PRESETS[b], dev)
        ai_a = estimate_flops_bytes(sa, [(1, 32768)])["est_ai_attn"]
        ai_b = estimate_flops_bytes(sb, [(1, 32768)])["est_ai_attn"]
        lines.append(f"  {name:22s} {a} → {b}")
        lines.append(f"  {'':22s} {why}")
        lines.append(f"  {'':22s} KV/tok {sa.kv_bytes_per_token/1024:.0f}K → "
                     f"{sb.kv_bytes_per_token/1024:.0f}K   "
                     f"decode AI@32K {ai_a:.1f} → {ai_b:.1f}   "
                     f"active {sa.linear_params_active/1e9:.2f}B → "
                     f"{sb.linear_params_active/1e9:.2f}B   "
                     f"tier {sa.tier}/{sb.tier}")
    return lines


def plot(keys, dev, outdir):
    M = len(keys)
    fig, axes = plt.subplots(3, M, figsize=(4.3 * M, 11.5), squeeze=False)
    for j, key in enumerate(keys):
        s = for_device(MODEL_PRESETS[key], dev)
        x = [max(c, 100) for c in CS]
        # row 0 — prefill attention share
        for n in PREFILL_NS:
            share = [estimate_flops_bytes(s, [(n, c)])["est_attn_flop_frac"] for c in CS]
            axes[0][j].plot(x, share, "o-", ms=3, label=f"prefill n={n}")
        dec = [estimate_flops_bytes(s, [(1, c)])["est_attn_flop_frac"] for c in CS]
        axes[0][j].plot(x, dec, "s--", ms=3, color="k", label="decode n=1")
        axes[0][j].axhline(0.5, color="gray", ls=":")
        axes[0][j].set_ylim(0, 1)
        axes[0][j].set_ylabel("attention share of FLOPs")
        axes[0][j].set_title(f"{s.name.split('/')[-1]}\n{s.family} · {s.arch_label} · "
                             f"KV {s.kv_bytes_per_token/1024:.0f} KB/tok")
        # row 1 — attention AI, prefill vs decode
        for n in PREFILL_NS:
            ai = [estimate_flops_bytes(s, [(n, c)])["est_ai_attn"] for c in CS]
            axes[1][j].plot(x, ai, "o-", ms=3, label=f"prefill n={n}")
        ai_d = [estimate_flops_bytes(s, [(1, c)])["est_ai_attn"] for c in CS]
        axes[1][j].plot(x, ai_d, "s--", ms=3, color="k", label="decode n=1")
        axes[1][j].set_yscale("log")
        axes[1][j].set_ylabel("attention AI [FLOP/byte]")
        # row 2 — per-request KV footprint (where windows and linear layers show up)
        kv = [s.kv_bytes_for(c) / 1e6 for c in CS]
        axes[2][j].plot(x, kv, "o-", ms=3, color="#e45756", label="KV per request")
        axes[2][j].set_yscale("log")
        axes[2][j].set_ylabel("KV footprint per request [MB]")
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("context c (c=0 drawn at 2⁷)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
    fig.suptitle("Attention families: prefill vs decode (analytic, single request)")
    fig.tight_layout()
    fig.savefig(outdir / "arch_compare.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--pairs", action="store_true", help="also print the controlled-pair table")
    ap.add_argument("--all", action="store_true", help="table for every preset")
    ap.add_argument("--gpu", default="A100-80GB", help=", ".join(DEVICES))
    ap.add_argument("--gpu-mem-gib", type=float, default=None,
                    help="override the device profile's memory size")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dev = DEVICES[args.gpu]
    mem = args.gpu_mem_gib if args.gpu_mem_gib else dev.mem_gib
    keys = list(MODEL_PRESETS) if args.all else args.models
    lines = [f"device: {dev.name}  cc{dev.capability[0]}.{dev.capability[1]}  "
             f"{mem:.0f}GiB  {dev.peak_bf16_flops/1e12:.0f} TF/s bf16  "
             f"{dev.hbm_bytes_per_s/1e12:.2f} TB/s  FA{dev.fa_version}"
             + ("  (MLA V is zero-padded on this GPU)" if dev.mla_v_padded else ""), ""]
    lines += table(keys, dev, mem, args.gpu_mem_util)
    if args.pairs or args.all:
        lines += pair_table(dev, mem, args.gpu_mem_util)
    plot(args.models, dev, outdir)
    (outdir / "arch_compare.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {outdir}/arch_compare.png, arch_compare.txt")


if __name__ == "__main__":
    main()
