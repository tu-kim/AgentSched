"""Experiment configuration generators.

A BatchConfig fully specifies one measured batch iteration:
  pairs = [(n_i, c_i), ...]   n_i = new tokens computed, c_i = cached KV tokens.

All c_i are aligned to BLOCK_SIZE (vLLM prefix cache reuses whole blocks only).
"""
from dataclasses import dataclass, field
import math

BLOCK_SIZE = 16
TOKEN_BUDGET = 8192


def align(c: int) -> int:
    return (c // BLOCK_SIZE) * BLOCK_SIZE


@dataclass
class BatchConfig:
    exp: str                       # "exp1" | "exp2" | "exp3" | "exp4"
    name: str
    pairs: list = field(default_factory=list)  # [(n_i, c_i)]

    @property
    def batch_size(self):
        return len(self.pairs)

    @property
    def ns(self):
        return [p[0] for p in self.pairs]

    @property
    def cs(self):
        return [p[1] for p in self.pairs]

    def stats(self):
        ns, cs = self.ns, self.cs
        B = len(ns)
        mean_n = sum(ns) / B
        mean_c = sum(cs) / B
        std = lambda xs, m: math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        cv = lambda xs, m: (std(xs, m) / m) if m > 0 else 0.0
        # Pearson corr(n, c)
        sn, sc = std(ns, mean_n), std(cs, mean_c)
        if sn > 0 and sc > 0:
            corr = sum((a - mean_n) * (b - mean_c) for a, b in zip(ns, cs)) / (B * sn * sc)
        else:
            corr = 0.0
        return dict(
            token_budget=sum(ns), batch_size=B,
            mean_n=mean_n, cv_n=cv(ns, mean_n),
            mean_c=mean_c, cv_c=cv(cs, mean_c),
            corr_nc=corr,
            sum_n_times_ctx=sum(n * (c + n) for n, c in self.pairs),
            total_kv_tokens=sum(c + n for n, c in self.pairs),
        )


# ---------------------------------------------------------------- Exp 1
def exp1_fragmentation():
    """Same Sn=8192 split across B in {1..128}; sweep uniform c."""
    cfgs = []
    for c in [0, 4096, 16384, 65536, 131072]:
        c = align(c)
        for B in [1, 2, 4, 8, 16, 32, 64, 128]:
            n = TOKEN_BUDGET // B
            cfgs.append(BatchConfig("exp1", f"B{B}_n{n}_c{c}",
                                    [(n, c)] * B))
    return cfgs


# ---------------------------------------------------------------- Exp 2
def exp2_n_heterogeneity():
    """Sn=8192, B=8, uniform c; vary CV(n) with mean(n)=1024 fixed."""
    variants = {
        "cv0.00": [1024] * 8,
        "cv0.25": [704, 832, 960, 1024, 1024, 1088, 1216, 1344],
        "cv0.50": [512, 512, 768, 896, 1152, 1280, 1536, 1536],
        "cv1.00": [128, 256, 384, 512, 768, 1024, 1536, 3584],
        "cv1.75": [64, 64, 128, 128, 256, 512, 1024, 6016],
    }
    for k, v in variants.items():
        assert sum(v) == TOKEN_BUDGET, (k, sum(v))
    cfgs = []
    for c in [0, 16384, 65536]:
        for k, ns in variants.items():
            cfgs.append(BatchConfig("exp2", f"{k}_c{c}",
                                    [(n, c) for n in ns]))
    return cfgs


# ---------------------------------------------------------------- Exp 3
def exp3_c_heterogeneity():
    """n_i=1024, B=8, Sc = 8*32K fixed; vary CV(c)."""
    K = 1024
    variants = {
        "cv0.00": [32 * K] * 8,
        "cv0.35": [16 * K, 24 * K, 28 * K, 32 * K, 32 * K, 36 * K, 40 * K, 48 * K],
        "cv0.70": [8 * K, 12 * K, 16 * K, 24 * K, 32 * K, 44 * K, 56 * K, 64 * K],
        "cv1.30": [4 * K, 4 * K, 8 * K, 8 * K, 16 * K, 24 * K, 64 * K, 128 * K],
    }
    for k, v in variants.items():
        assert sum(v) == 8 * 32 * K, (k, sum(v))
    return [BatchConfig("exp3", k, [(1024, align(c)) for c in cs])
            for k, cs in variants.items()]


# ---------------------------------------------------------------- Exp 4
def exp4_correlation():
    """Same {n_i}, {c_i} multisets; only the pairing differs."""
    K = 1024
    ns = [64, 128, 256, 512, 1024, 1536, 2048, 2624]      # sum = 8192
    cs = [4 * K, 8 * K, 16 * K, 24 * K, 32 * K, 48 * K, 64 * K, 128 * K]
    assert sum(ns) == TOKEN_BUDGET
    import random
    rng = random.Random(0)
    # pick a shuffle whose pairing correlation is near zero
    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((x - mb) ** 2 for x in b) ** 0.5
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)
    rand_cs = min((rng.sample(cs, len(cs)) for _ in range(500)),
                  key=lambda p: abs(corr(ns, p)))
    variants = {
        "pos": list(zip(ns, cs)),
        "rand": list(zip(ns, rand_cs)),
        "neg": list(zip(ns, reversed(cs))),
    }
    return [BatchConfig("exp4", k, [(n, align(c)) for n, c in ps])
            for k, ps in variants.items()]


def all_configs():
    return exp1_fragmentation() + exp2_n_heterogeneity() + exp3_c_heterogeneity() + exp4_correlation()


if __name__ == "__main__":
    for cfg in all_configs():
        s = cfg.stats()
        print(f"{cfg.exp:5s} {cfg.name:20s} B={s['batch_size']:4d} "
              f"cv_n={s['cv_n']:.2f} cv_c={s['cv_c']:.2f} corr={s['corr_nc']:+.2f} "
              f"kv_tokens={s['total_kv_tokens']:>9,d} "
              f"Sum n*(c+n)={s['sum_n_times_ctx']:>13,d}")
