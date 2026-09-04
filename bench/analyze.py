"""Analysis of results/raw.jsonl (one or several models).

  python -m bench.analyze --raw results/raw.jsonl [more.jsonl ...] --outdir results [--device A100-80GB]

Produces
  summary.csv               all configs × all metrics (+ derived efficiency columns)
  exp0_dominance.png        latency and attention share vs c  → c*   (one line per model × B)
  exp1_fragmentation.png    throughput / TFLOPS / roofline vs B
  exp2_n_hetero.png         effect decomposition vs CV(n)
  exp3_c_hetero.png         effect decomposition vs CV(c)
  exp4_pairing.png          latency vs ρ(n,c)
  report.txt                c*, fixed-budget spread, cost-model R², decomposition tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.metrics import DEVICE_PEAKS

KERNEL_CLASSES = ["attention", "gemm", "moe", "activation", "kvcache", "other"]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("skipped")]
    skipped = [(r.get("model"), r.get("exp"), r.get("name"), r.get("reason")) for r in rows if r.get("skipped")]
    df = pd.DataFrame(ok)
    if df.empty:
        return df, skipped
    df["model_short"] = df["model"].astype(str).str.split("/").str[-1]
    kcols = [f"kernel_time_{k}_us" for k in KERNEL_CLASSES if f"kernel_time_{k}_us" in df.columns]
    if kcols:
        tot = df[kcols].sum(axis=1, min_count=1)
        df["attn_time_frac"] = df.get("kernel_time_attention_us", np.nan) / tot
        df["gemm_time_frac"] = (df.get("kernel_time_gemm_us", 0).fillna(0)
                                + df.get("kernel_time_moe_us", 0).fillna(0)) / tot
    else:
        df["attn_time_frac"] = np.nan
        df["gemm_time_frac"] = np.nan
    return df, skipped


# ------------------------------------------------------------------ helpers
def homogeneous_reference(df, exp):
    """Per (model, group), the cv=0 / ρ≈0 config the others are compared against."""
    d = df[df.exp == exp]
    key = {"exp2": "cv_n", "exp3": "cv_c", "exp4": "corr_nc"}[exp]
    return {g: gd.loc[gd[key].abs().idxmin()] for g, gd in d.groupby(["model_short", "group"])}


def add_efficiency_columns(df):
    """Decompose each config's latency relative to its homogeneous reference:
       latency_ratio       = lat / lat_ref                   (total effect)
       work_ratio          = Σn(c+n/2)_causal / ref          (attention-work → AI effect)
       tflops_ratio        = achieved_tflops / ref           (kernel-efficiency effect)
       batching_efficiency = throughput / throughput_ref
    """
    for col in ["latency_ratio", "work_ratio", "tflops_ratio", "batching_efficiency"]:
        df[col] = np.nan
    for exp in ["exp2", "exp3", "exp4"]:
        if exp not in set(df.exp):
            continue
        refs = homogeneous_reference(df, exp)
        for i, r in df[df.exp == exp].iterrows():
            ref = refs[(r.model_short, r.group)]
            df.at[i, "latency_ratio"] = r.latency_median_s / ref.latency_median_s
            df.at[i, "work_ratio"] = r.sum_n_ctx_causal / ref.sum_n_ctx_causal
            df.at[i, "tflops_ratio"] = r.achieved_tflops / ref.achieved_tflops
            df.at[i, "batching_efficiency"] = r.throughput_tok_s / ref.throughput_tok_s
    return df


def _sorted_groups(d):
    return sorted(d.groupby(["model_short", "group"]),
                  key=lambda kv: (kv[0][0], int(kv[0][1].split("=")[1])))


# ------------------------------------------------------------------ exp0
def analyze_exp0(df, outdir, lines):
    d = df[(df.exp == "exp0") & df.name.str.startswith("dom_")]
    if d.empty:
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    cstar = {}
    multi = d.model_short.nunique() > 1
    for (m, g), gd in _sorted_groups(d):
        gd = gd.sort_values("mean_c")
        lbl = f"{m} {g}" if multi else g
        x = gd.mean_c.clip(lower=256)          # log axis: c=0 drawn at 2^8
        base = gd[gd.mean_c == 0].latency_median_s
        if base.empty:
            lines.append(f"  [exp0] {lbl}: no c=0 row (skipped?) – cannot normalise")
            continue
        base = base.iloc[0]
        axes[0].plot(x, gd.latency_median_s * 1e3, "o-", label=lbl)
        axes[1].plot(x, gd.latency_median_s / base, "o-", label=lbl)
        measured = gd.attn_time_frac.notna().any()
        if measured:
            axes[2].plot(x, gd.attn_time_frac, "o-", label=f"{lbl} measured")
        axes[2].plot(x, gd.est_attn_flop_frac, "x--", alpha=.6, label=f"{lbl} est. FLOP share")
        if measured:
            hit = gd[gd.attn_time_frac >= 0.5]; how = "measured attention kernel-time share ≥ 50%"
        else:
            hit = gd[gd.latency_median_s >= 2 * base]; how = "latency ≥ 2× the c=0 baseline"
        cstar[(m, g)] = (int(hit.mean_c.iloc[0]) if len(hit) else None, how)
    axes[0].set_ylabel("iteration latency [ms]")
    axes[1].set_ylabel("latency / latency(c=0)")
    axes[1].axhline(2, color="gray", ls=":", label="2× (attn == GEMM)")
    axes[2].set_ylabel("attention share of iteration")
    axes[2].axhline(0.5, color="gray", ls=":")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("c (cached context per request; c=0 drawn at 2⁸)")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("Exp0: where does attention dominate?  (Σn = 8192)")
    fig.tight_layout(); fig.savefig(outdir / "exp0_dominance.png", dpi=150)

    lines.append("== Exp0: attention-dominance crossover c* ==")
    for (m, g), (c, how) in cstar.items():
        lines.append(f"  {m:22s} {g:6s} c* = {str(c) if c is not None else 'not reached in sweep':>22s}   ({how})")
    for m, md in d.groupby("model_short"):
        r = md.iloc[0]
        by_n = r.get("model_attn_crossover_by_n")
        if isinstance(by_n, dict):
            lines.append(f"  analytic FLOP crossover {m} ({r.get('model_arch')}): "
                         + ", ".join(f"n={k}: {v:,.0f}" for k, v in by_n.items())
                         + f"  [KV {r.model_kv_bytes_per_token/1024:.0f} KB/token]")
    return cstar


# ------------------------------------------------------------------ exp1
def plot_exp1(df, outdir, device):
    d = df[df.exp == "exp1"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    multi = d.model_short.nunique() > 1
    for (m, g), gd in _sorted_groups(d):
        gd = gd.sort_values("batch_size")
        lbl = f"{m} {g}" if multi else g
        axes[0].plot(gd.batch_size, gd.throughput_tok_s, "o-", label=lbl)
        axes[1].plot(gd.batch_size, gd.achieved_tflops, "o-", label=lbl)
        axes[2].plot(gd.est_ai_analytic, gd.achieved_tflops, "o-", label=lbl)
    if device in DEVICE_PEAKS:
        pf, bw = DEVICE_PEAKS[device]
        ai = np.logspace(0, 4, 100)
        axes[2].plot(ai, np.minimum(pf, bw * ai) / 1e12, "k--", alpha=.5, label=f"{device} roofline")
    for ax, (x, y) in zip(axes, [("batch size B (Σn=8192)", "throughput [tok/s]"),
                                 ("batch size B", "achieved TFLOPS"),
                                 ("analytic iteration AI [FLOP/byte]", "achieved TFLOPS")]):
        ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_xscale("log", base=2 if "B" in x else 10)
    fig.suptitle("Exp1: token fragmentation @ fixed Σn=8192")
    fig.tight_layout(); fig.savefig(outdir / "exp1_fragmentation.png", dpi=150)


# ------------------------------------------------------------------ exp2 / exp3
def plot_hetero(df, exp, xcol, title, outdir, lines):
    d = df[df.exp == exp]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lines.append(f"== {title} ==")
    lines.append(f"  {'model':22s} {'group':14s} {xcol:>6s} {'lat×':>6s} {'work×':>6s} {'TFLOPS×':>8s} {'eff':>6s}")
    multi = d.model_short.nunique() > 1
    for (m, g), gd in _sorted_groups(d):
        gd = gd.sort_values(xcol)
        lbl = f"{m} {g}" if multi else g
        axes[0].plot(gd[xcol], gd.latency_ratio, "o-", label=f"{lbl} latency×")
        axes[0].plot(gd[xcol], gd.work_ratio, "x--", alpha=.6, label=f"{lbl} attn-work× (AI effect)")
        axes[1].plot(gd[xcol], gd.tflops_ratio, "o-", label=lbl)
        for _, r in gd.iterrows():
            lines.append(f"  {m:22s} {g:14s} {r[xcol]:6.2f} {r.latency_ratio:6.3f} {r.work_ratio:6.3f} "
                         f"{r.tflops_ratio:8.3f} {r.batching_efficiency:6.3f}")
    axes[0].set_ylabel("ratio vs homogeneous"); axes[0].set_title("total effect vs. work (AI) effect")
    axes[1].set_ylabel("achieved TFLOPS / homogeneous"); axes[1].set_title("kernel-efficiency effect")
    for ax in axes:
        ax.set_xlabel(xcol); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.axhline(1, color="gray", ls=":")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(outdir / f"{exp}_{'n' if exp == 'exp2' else 'c'}_hetero.png", dpi=150)


# ------------------------------------------------------------------ exp4
def plot_exp4(df, outdir, lines):
    d = df[df.exp == "exp4"]
    if d.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    lines.append("== Exp4: pairing ==")
    for (m, g), gd in _sorted_groups(d):
        gd = gd.sort_values("corr_nc")
        ax1.plot(gd.corr_nc, gd.latency_median_s * 1e3, "o-", label=f"{m} latency [ms]")
        ax2.plot(gd.corr_nc, gd.sum_n_ctx_causal / 1e6, "x--", alpha=.6, label=f"{m} Σn_i(c_i+n_i/2) [M]")
        for _, r in gd.iterrows():
            lines.append(f"  {m:22s} ρ={r.corr_nc:+.2f}  lat={r.latency_median_s*1e3:7.1f}ms  "
                         f"work={r.sum_n_ctx_causal/1e6:6.0f}M  TFLOPS={r.achieved_tflops:6.1f}  "
                         f"eff(vs ρ≈0)={r.batching_efficiency:.3f}")
    ax1.set_xlabel("ρ(n, c)  (identical {n}, {c} multisets, Σn, Σc, B)"); ax1.set_ylabel("iteration latency [ms]")
    ax2.set_ylabel("attention work [M token²]")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7); ax1.grid(alpha=.3)
    fig.suptitle("Exp4: n-c pairing")
    fig.tight_layout(); fig.savefig(outdir / "exp4_pairing.png", dpi=150)


# ------------------------------------------------------------------ hypothesis
def fit_r2(d, cols):
    y = d.latency_median_s.values
    X = np.column_stack([d[c].values.astype(float) for c in cols] + [np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var(), beta


MODELS = [
    ("Σn only", ["token_budget"]),
    ("Σn + Σc", ["token_budget", "sum_c"]),
    ("Σn + Σn_i(c_i+n_i/2)", ["token_budget", "sum_n_ctx_causal"]),
    ("Σn + Σc + Σn² + Σnc + B", ["token_budget", "sum_c", "sum_n_sq", "sum_nc", "batch_size"]),
    ("est FLOPs + est bytes", ["est_flops_total", "est_bytes_analytic"]),
]


def hypothesis(df, cstar, lines):
    for m, dm in df.groupby("model_short"):
        d = dm.dropna(subset=["latency_median_s"]).drop_duplicates("shape_key")
        lines.append(f"== [{m}] fixed-budget spread (Σn = 8192, distinct shapes) ==")
        fb = d[d.token_budget == 8192]
        if len(fb) > 1:
            lines.append(f"  shapes: {len(fb)}  latency min={fb.latency_median_s.min()*1e3:.1f}ms "
                         f"max={fb.latency_median_s.max()*1e3:.1f}ms  "
                         f"ratio={fb.latency_median_s.max()/fb.latency_median_s.min():.1f}×  "
                         f"CV={fb.latency_median_s.std()/fb.latency_median_s.mean():.2f}")
        lines.append(f"== [{m}] cost-model fit: latency ~ linear in features (+intercept), distinct shapes ==")
        for label, cols in MODELS:
            if len(d) > len(cols) + 2:
                r2, _ = fit_r2(d, cols)
                lines.append(f"  all ({len(d):3d})     R²({label:28s}) = {r2:.4f}")
        cs = [c for (mm, _), (c, _) in cstar.items() if mm == m and c is not None]
        if cs:
            thr = min(cs)
            ad = d[d.mean_c >= thr]
            lines.append(f"  -- attention-dominant subset (mean c ≥ c*={thr}): {len(ad)} shapes")
            for label, cols in MODELS:
                if len(ad) > len(cols) + 2:
                    r2, _ = fit_r2(ad, cols)
                    lines.append(f"  c ≥ c* ({len(ad):3d})  R²({label:28s}) = {r2:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["results/raw.jsonl"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--device", default="A100-80GB", help=f"roofline reference: {list(DEVICE_PEAKS)}")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, skipped = load(args.raw)
    if df.empty:
        print(f"No measured configs. Skipped {len(skipped)}:")
        print("\n".join(f"  {m} {e}/{n}: {r}" for m, e, n, r in skipped))
        return
    df = add_efficiency_columns(df)
    df.drop(columns=[c for c in ["latency_all_s", "pairs", "kernel_top", "kernel_unclassified",
                                 "model_attn_crossover_by_n"]
                     if c in df.columns]).to_csv(outdir / "summary.csv", index=False)

    lines = []
    cstar = analyze_exp0(df, outdir, lines)
    plot_exp1(df, outdir, args.device)
    plot_hetero(df, "exp2", "cv_n", "Exp2: CV(n) @ Σn=8192, B=8", outdir, lines)
    plot_hetero(df, "exp3", "cv_c", "Exp3: CV(c) @ n=1024, B=8 (AI-neutral control)", outdir, lines)
    plot_exp4(df, outdir, lines)
    hypothesis(df, cstar, lines)
    bad = df[~(df.get("co_scheduled", True).astype(bool) & df.get("cache_hits_ok", True).astype(bool))]
    if len(bad):
        lines.append(f"\n!! {len(bad)} configs were NOT realised as designed (co_scheduled / cache_hits_ok False):")
        lines += [f"  {r.model_short} {r.exp}/{r.name}" for _, r in bad.iterrows()]
    lines.append(f"\nSkipped configs: {len(skipped)}")
    lines += [f"  {m} {e}/{n}: {r}" for m, e, n, r in skipped]
    (outdir / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {outdir}/summary.csv, plots, report.txt")


if __name__ == "__main__":
    main()
