"""Experiment configuration generators.

A BatchConfig fully specifies one measured batch iteration:
  pairs = [(n_i, c_i), ...]   n_i = new tokens computed, c_i = cached KV tokens.

All c_i are aligned to BLOCK_SIZE (vLLM prefix cache reuses whole blocks only).

Experiments
  exp0  attention dominance : fixed Σn, sweep c finely to find c* where attention
                              time overtakes GEMM time; plus a small Σn sweep so
                              "latency ~ Σn alone" is a well-posed regression.
  exp1  token fragmentation : fixed Σn split across B in {1..512}, uniform c.
  exp2  n heterogeneity     : fixed Σn, B, c; vary CV(n).
  exp3  c heterogeneity     : fixed n, B, Σc; vary CV(c).   (AI-neutral control)
  exp4  n-c correlation     : same {n_i}, {c_i} multisets; vary pairing ρ(n,c).

exp2-4 are centred on base_c (mean c = base_c), which should be ≥ the c* found by
exp0. `auto` picks the next power of two ≥ the model's analytic c*.
"""
from dataclasses import dataclass, field
from itertools import permutations
import math

BLOCK_SIZE = 16
TOKEN_BUDGET = 8192
K = 1024


def align(c: int) -> int:
    return (c // BLOCK_SIZE) * BLOCK_SIZE


def next_pow2(x, floor=8192):
    p = floor
    while p < x:
        p *= 2
    return p


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def pearson(a, b):
    sa, sb = _std(a), _std(b)
    if sa == 0 or sb == 0:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) * sa * sb)


@dataclass
class BatchConfig:
    exp: str
    name: str
    pairs: list = field(default_factory=list)  # [(n_i, c_i)]
    group: str = ""                            # analysis grouping key (e.g. "c=32768")

    @property
    def batch_size(self):
        return len(self.pairs)

    @property
    def ns(self):
        return [p[0] for p in self.pairs]

    @property
    def cs(self):
        return [p[1] for p in self.pairs]

    @property
    def shape_key(self):
        """Identical shapes (up to request order) are measured once."""
        return "|".join(f"{n},{c}" for n, c in sorted(self.pairs))

    def kv_tokens_needed(self):
        """KV slots allocated by the scheduler: ceil((c+n)/16) blocks per request.
        (max_tokens=1: the sampled token never gets a slot.)"""
        return sum(math.ceil((c + n) / BLOCK_SIZE) * BLOCK_SIZE for n, c in self.pairs)

    def stats(self):
        """Shape statistics. The five sufficient statistics of the attention cost
        model are (batch_size, token_budget, sum_c, sum_n_sq, sum_nc); everything
        else is derived for readability."""
        ns, cs = self.ns, self.cs
        B = len(ns)
        mean_n, mean_c = _mean(ns), _mean(cs)
        cv = lambda xs, m: (_std(xs) / m) if m > 0 else 0.0
        return dict(
            token_budget=sum(ns), batch_size=B,
            sum_c=sum(cs),
            mean_n=mean_n, cv_n=cv(ns, mean_n), max_n=max(ns),
            mean_c=mean_c, cv_c=cv(cs, mean_c), max_c=max(cs),
            corr_nc=pearson(ns, cs),
            sum_n_sq=sum(n * n for n in ns),
            sum_nc=sum(n * c for n, c in self.pairs),
            # Σ n_i (c_i + n_i): non-causal attention work (numerator of the AI formula)
            sum_n_ctx=sum(n * (c + n) for n, c in self.pairs),
            # Σ n_i (c_i + (n_i+1)/2): causal attention work (what the kernel executes)
            sum_n_ctx_causal=sum(n * (c + (n + 1) / 2) for n, c in self.pairs),
            total_kv_tokens=sum(c + n for n, c in self.pairs),
            kv_tokens_needed=self.kv_tokens_needed(),
            shape_key=self.shape_key,
        )


# ---------------------------------------------------------------- Exp 0
def exp0_attention_dominance():
    """(a) Σn=8192 at B ∈ {1, 8, 64, 256}, c swept geometrically up to 512K: locate
    c* where the attention share of iteration time crosses 50%. B=256 (n=32) pushes
    into the small-n / large-B corner where MLA's Σc_i decompression term matters most.
    (b) Σn ∈ {1024..65536} at B=8, c ∈ {0, 16K}: gives Σn variance for the
    'latency ~ Σn alone' baseline regression, up to a large-batch-budget scale."""
    cfgs = []
    for B in [1, 8, 64, 256]:
        n = TOKEN_BUDGET // B
        for c in [0, K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K, 512 * K]:
            cfgs.append(BatchConfig("exp0", f"dom_B{B}_n{n}_c{c}", [(n, c)] * B,
                                    group=f"B={B}"))
    for c in [0, 16 * K]:
        for budget in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            n = budget // 8
            cfgs.append(BatchConfig("exp0", f"budget{budget}_B8_n{n}_c{c}", [(n, c)] * 8,
                                    group=f"budget,c={c}"))
    return cfgs


# ---------------------------------------------------------------- Exp 1
def exp1_fragmentation(c_levels=(0, 4 * K, 16 * K, 64 * K, 128 * K, 256 * K, 512 * K)):
    """Same Σn=8192 split across B in {1..512} (n from 8192 down to 16); sweep
    uniform c up to 512K."""
    cfgs = []
    for c in c_levels:
        c = align(c)
        for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            n = TOKEN_BUDGET // B
            cfgs.append(BatchConfig("exp1", f"B{B}_n{n}_c{c}", [(n, c)] * B,
                                    group=f"c={c}"))
    return cfgs


# ---------------------------------------------------------------- Exp 2
N_VARIANTS = {
    # Σ = 8192, mean = 1024, B = 8; labels are the actual population CV
    "cv0.00": [1024] * 8,
    "cv0.18": [704, 832, 960, 1024, 1024, 1088, 1216, 1344],
    "cv0.38": [512, 512, 768, 896, 1152, 1280, 1536, 1536],
    "cv1.03": [128, 256, 384, 512, 768, 1024, 1536, 3584],
    "cv1.87": [64, 64, 128, 128, 256, 512, 1024, 6016],
    # 7 near-idle continuations (n=16) + one huge new-conversation prefill: the
    # extreme multi-turn shape this benchmark targets
    "cv2.60": [16, 16, 16, 16, 16, 16, 16, 8080],
}


def exp2_n_heterogeneity(base_c):
    """Σn=8192, B=8, uniform c ∈ {0, base_c, 2·base_c, 4·base_c}; vary CV(n).
    At c=0 the AI itself changes with CV(n) (Σn_i² term); at c ≫ n the AI is
    ~constant and the experiment isolates kernel-level effects."""
    for k, v in N_VARIANTS.items():
        assert sum(v) == TOKEN_BUDGET, (k, sum(v))
    cfgs = []
    for c in [0, align(base_c), align(2 * base_c), align(4 * base_c)]:
        for k, ns in N_VARIANTS.items():
            cfgs.append(BatchConfig("exp2", f"{k}_c{c}", [(n, c) for n in ns],
                                    group=f"c={c}"))
    return cfgs


# ---------------------------------------------------------------- Exp 3
C_TEMPLATES = {
    # in units of base_c/32; each sums to 256 (= 8 × 32) so mean(c) = base_c
    "cv0.00": [32, 32, 32, 32, 32, 32, 32, 32],
    "cv0.29": [16, 24, 28, 32, 32, 36, 40, 48],
    "cv0.61": [8, 12, 16, 24, 32, 44, 56, 64],
    "cv1.27": [4, 4, 8, 8, 16, 24, 64, 128],
    # 4K-context continuations alongside a ~192K-context outlier (unit = base_c/32)
    "cv1.94": [2, 2, 2, 2, 4, 8, 44, 192],
}


def _check_base_c(base_c):
    assert base_c % 512 == 0, "base_c must be a multiple of 512 for exact block alignment"


def exp3_c_heterogeneity(base_c):
    """n_i=1024, B=8, Σc = 8·base_c fixed; vary CV(c).
    With uniform n, both Σn_i(c_i+n_i) and Σ(2n_i+c_i) are invariant, so the
    analytic AI is identical across variants: this is the negative control that
    tells whether the scheduler must track the c distribution or only Σn_i·c_i."""
    _check_base_c(base_c)
    unit = base_c // 32
    for k, v in C_TEMPLATES.items():
        assert sum(v) == 256, (k, sum(v))
    return [BatchConfig("exp3", k, [(1024, align(t * unit)) for t in cs],
                        group=f"mean_c={base_c}")
            for k, cs in C_TEMPLATES.items()]


# ---------------------------------------------------------------- Exp 4
EXP4_NS = [32, 64, 128, 256, 512, 1024, 2048, 4128]           # Σ = 8192, 129× spread
EXP4_C_TEMPLATE = [2, 4, 8, 16, 24, 40, 64, 98]               # × base_c/32, Σ = 256 → mean = base_c


def exp4_correlation(base_c, n_points=7):
    """Same {n_i}, {c_i} multisets, Σn, Σc, B; only the pairing differs.
    All 8! pairings are enumerated; the attainable Pearson range [ρ_min, ρ_max]
    is split into n_points evenly spaced targets and the closest pairing is
    taken for each. Configs are named by the ρ actually achieved."""
    _check_base_c(base_c)
    assert sum(EXP4_NS) == TOKEN_BUDGET and sum(EXP4_C_TEMPLATE) == 256
    unit = base_c // 32
    cs = [align(t * unit) for t in EXP4_C_TEMPLATE]
    perms = [(pearson(EXP4_NS, list(p)), p) for p in permutations(cs)]
    rho_min, rho_max = min(r for r, _ in perms), max(r for r, _ in perms)
    cfgs = []
    for i in range(n_points):
        target = rho_min + (rho_max - rho_min) * i / (n_points - 1)
        rho, best = min(perms, key=lambda rp: abs(rp[0] - target))
        cfgs.append(BatchConfig("exp4", f"rho{rho:+.2f}", list(zip(EXP4_NS, best)),
                                group=f"mean_c={base_c}"))
    return cfgs


GENERATORS = {
    "exp0": lambda base_c: exp0_attention_dominance(),
    "exp1": lambda base_c: exp1_fragmentation(),
    "exp2": exp2_n_heterogeneity,
    "exp3": exp3_c_heterogeneity,
    "exp4": exp4_correlation,
}


def all_configs(exps=None, base_c=32 * K):
    exps = exps or list(GENERATORS)
    out = []
    for e in exps:
        out.extend(GENERATORS[e](base_c))
    return out


def group_by_shape(cfgs):
    """{shape_key: [configs with that shape]} preserving first-seen order."""
    groups = {}
    for c in cfgs:
        groups.setdefault(c.shape_key, []).append(c)
    return groups


def resolve_base_c(arg, spec):
    return next_pow2(spec.attn_crossover_ctx()) if str(arg) == "auto" else int(arg)


# ---------------------------------------------------------------- plan CLI
def _main():
    import argparse
    from bench.metrics import MODEL_PRESETS, estimate_flops_bytes
    ap = argparse.ArgumentParser(description="List configs; check feasibility for a model/GPU.")
    ap.add_argument("--exp", nargs="+", default=list(GENERATORS))
    ap.add_argument("--base-c", default="auto", help="int, or 'auto' = next pow2 ≥ analytic c* (as runner)")
    ap.add_argument("--model", default="qwen1.5-1.8b", help=f"preset: {list(MODEL_PRESETS)}")
    ap.add_argument("--gpu-mem-gib", type=float, default=80)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--capacity-margin", type=float, default=0.95, help="same gate as runner")
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    spec = MODEL_PRESETS[args.model]
    base_c = resolve_base_c(args.base_c, spec)
    cap = spec.kv_capacity_tokens(args.gpu_mem_gib, args.gpu_mem_util, tp=args.tp)
    gate = cap * args.capacity_margin
    cstar = {n: spec.attn_crossover_ctx(n) for n in (64, 1024, 8192)}
    print(f"model={spec.name} arch={spec.arch_label} params={spec.total_params/1e9:.2f}B "
          f"kv_bytes/token={spec.kv_bytes_per_token/1024:.0f}KB")
    print(f"analytic c* (attn FLOPs == linear FLOPs): " +
          ", ".join(f"n={n}: {v:,.0f}" for n, v in cstar.items()) + f"  → base_c={base_c}")
    print(f"est. KV capacity on {args.gpu_mem_gib:.0f}GiB×{args.tp} @util {args.gpu_mem_util}: {cap:,d} tokens "
          f"(gate {args.capacity_margin:.2f}× = {gate:,.0f})\n")
    print(f"{'exp':5s} {'name':26s} {'B':>4s} {'cv_n':>5s} {'cv_c':>5s} {'rho':>5s} "
          f"{'kv_tokens':>10s} {'attnFLOP%':>9s} {'AI_attn':>8s} fits")
    cfgs = all_configs(args.exp, base_c)
    n_fit = 0
    for cfg in cfgs:
        s = cfg.stats()
        e = estimate_flops_bytes(spec, cfg.pairs)
        fits = s["kv_tokens_needed"] <= gate
        n_fit += fits
        print(f"{cfg.exp:5s} {cfg.name:26s} {s['batch_size']:4d} {s['cv_n']:5.2f} {s['cv_c']:5.2f} "
              f"{s['corr_nc']:+5.2f} {s['kv_tokens_needed']:>10,d} "
              f"{100*e['est_attn_flop_frac']:8.1f}% {e['est_ai_attn']:8.1f} {'ok' if fits else 'NO'}")
    n_shapes = len(group_by_shape(cfgs))
    print(f"\n{n_fit}/{len(cfgs)} configs fit; {n_shapes} distinct shapes to measure.")


if __name__ == "__main__":
    _main()
