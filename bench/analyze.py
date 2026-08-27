"""Analysis of raw.jsonl → per-experiment tables + plots + hypothesis test.

  python -m bench.analyze --raw results/raw.jsonl --outdir results
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["exp", "name", "token_budget", "batch_size", "mean_n", "cv_n",
        "mean_c", "cv_c", "corr_nc", "sum_n_times_ctx",
        "est_flops_total", "est_flops_attn", "est_bytes_total",
        "est_arith_intensity", "latency_median_s", "throughput_tok_s",
        "achieved_tflops", "est_hbm_gbps", "gpu_util", "mem_util",
        "power_w", "co_scheduled", "cached_tokens"]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame([r for r in rows if not r.get("skipped")])
    skipped = [(r.get("exp"), r.get("name"), r.get("reason"))
               for r in rows if r.get("skipped")]
    return df[[c for c in COLS if c in df.columns]], skipped


def plot_exp1(df, outdir):
    d = df[df.exp == "exp1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for c, g in d.groupby("mean_c"):
        g = g.sort_values("batch_size")
        axes[0].plot(g.batch_size, g.throughput_tok_s, "o-", label=f"c={int(c)}")
        axes[1].plot(g.batch_size, g.achieved_tflops, "o-", label=f"c={int(c)}")
        axes[2].plot(g.est_arith_intensity, g.achieved_tflops, "o-", label=f"c={int(c)}")
    for ax, (x, y) in zip(axes, [("batch size B (Σn=8192)", "throughput [tok/s]"),
                                 ("batch size B", "achieved TFLOPS"),
                                 ("estimated AI [FLOP/byte]", "achieved TFLOPS (roofline)")]):
        ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(); ax.grid(alpha=.3)
        if "AI" not in x:
            ax.set_xscale("log", base=2)
        else:
            ax.set_xscale("log")
    fig.suptitle("Exp1: token fragmentation @ fixed Σn=8192")
    fig.tight_layout(); fig.savefig(outdir / "exp1_fragmentation.png", dpi=150)


def plot_cv(df, exp, xcol, title, outdir):
    d = df[df.exp == exp]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    key = "mean_c" if exp == "exp2" else None
    groups = d.groupby(key) if key else [(None, d)]
    for c, g in groups:
        g = g.sort_values(xcol)
        lbl = f"c={int(c)}" if c is not None else None
        axes[0].plot(g[xcol], g.throughput_tok_s, "o-", label=lbl)
        axes[1].plot(g[xcol], g.achieved_tflops, "o-", label=lbl)
    axes[0].set_ylabel("throughput [tok/s]"); axes[1].set_ylabel("achieved TFLOPS")
    for ax in axes:
        ax.set_xlabel(xcol); ax.grid(alpha=.3)
        if key:
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(outdir / f"{exp}.png", dpi=150)


def plot_exp4(df, outdir):
    d = df[df.exp == "exp4"].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = d.sort_values("corr_nc")
    ax.bar(d.name, d.latency_median_s * 1e3, color=["#4c78a8", "#f58518", "#e45756"])
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, r.latency_median_s * 1e3, f"Σn(c+n)={r.sum_n_times_ctx/1e6:.0f}M\n"
                f"corr={r.corr_nc:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("iteration latency [ms]")
    ax.set_title("Exp4: n-c pairing (identical {n},{c} multisets, Σn, Σc, B)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(outdir / "exp4_pairing.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    df, skipped = load(args.raw)
    df.to_csv(outdir / "summary.csv", index=False)

    plot_exp1(df, outdir)
    plot_cv(df, "exp2", "cv_n", "Exp2: CV(n) @ Σn=8192, B=8, mean(n)=1024", outdir)
    plot_cv(df, "exp3", "cv_c", "Exp3: CV(c) @ n=1024, B=8, mean(c)=32K", outdir)
    plot_exp4(df, outdir)

    # ---- hypothesis test: does Σn_i predict latency?  add shape features and
    # compare explanatory power (R^2 of latency ~ features).
    import numpy as np
    d = df.dropna(subset=["latency_median_s"])
    y = np.log(d.latency_median_s.values)
    def r2(cols):
        X = np.column_stack([np.log(np.maximum(d[c].values.astype(float), 1e-9))
                             for c in cols] + [np.ones(len(d))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return 1 - resid.var() / y.var()
    lines = [
        "Hypothesis: Σn_i alone cannot explain GPU execution cost.",
        f"  R2( latency ~ token_budget )                       = {r2(['token_budget']):.4f}",
        f"  R2( latency ~ token_budget + Σn_i(c_i+n_i) )       = {r2(['token_budget','sum_n_times_ctx']):.4f}",
        f"  R2( latency ~ + batch_size + est_bytes + est_flops)= {r2(['token_budget','sum_n_times_ctx','batch_size','est_bytes_total','est_flops_total']):.4f}",
        "",
        f"Skipped configs: {len(skipped)}",
        *[f"  {e}/{n}: {r}" for e, n, r in skipped],
    ]
    (outdir / "hypothesis.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {outdir}/summary.csv, plots, hypothesis.txt")


if __name__ == "__main__":
    main()
