"""Behaviour of the experiment-configuration generators."""
import unittest

from bench.configs import (BLOCK_SIZE, TOKEN_BUDGET, BatchConfig, C_TEMPLATES, EXP4_C_TEMPLATE,
                           EXP4_NS, N_VARIANTS, align, all_configs, exp0_attention_dominance,
                           exp1_fragmentation, exp2_n_heterogeneity, exp3_c_heterogeneity,
                           exp4_correlation, group_by_shape, next_pow2, pearson, resolve_base_c)
from bench.metrics import MODEL_PRESETS

BASE_C = 16384


class TestHelpers(unittest.TestCase):
    def test_align_floors_to_block(self):
        self.assertEqual(align(0), 0)
        self.assertEqual(align(16), 16)
        self.assertEqual(align(31), 16)
        self.assertEqual(align(4096), 4096)
        for x in (1, 15, 17, 1000, 123456):
            self.assertEqual(align(x) % BLOCK_SIZE, 0)
            self.assertLessEqual(align(x), x)
            self.assertLess(x - align(x), BLOCK_SIZE)

    def test_next_pow2(self):
        self.assertEqual(next_pow2(1), 8192)            # floor
        self.assertEqual(next_pow2(8192), 8192)
        self.assertEqual(next_pow2(8193), 16384)
        self.assertEqual(next_pow2(12352), 16384)       # Qwen1.5-1.8B c*
        self.assertEqual(next_pow2(100, floor=16), 128)

    def test_pearson(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertEqual(pearson([1, 1, 1], [1, 2, 3]), 0.0)   # zero variance, no ZeroDivisionError
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [1, 3, 2, 4]), 0.8)

    def test_resolve_base_c(self):
        spec = MODEL_PRESETS["qwen1.5-1.8b"]
        self.assertEqual(resolve_base_c(4096, spec), 4096)
        self.assertEqual(resolve_base_c("4096", spec), 4096)
        self.assertEqual(resolve_base_c("auto", spec), next_pow2(spec.attn_crossover_ctx()))


class TestBatchConfigStats(unittest.TestCase):
    def test_hand_computed(self):
        cfg = BatchConfig("t", "x", [(100, 1000), (300, 3000)])
        s = cfg.stats()
        self.assertEqual(s["token_budget"], 400)
        self.assertEqual(s["batch_size"], 2)
        self.assertEqual(s["sum_c"], 4000)
        self.assertEqual(s["mean_n"], 200)
        self.assertEqual(s["max_n"], 300)
        self.assertEqual(s["mean_c"], 2000)
        self.assertEqual(s["max_c"], 3000)
        self.assertAlmostEqual(s["cv_n"], 0.5)          # std 100 / mean 200
        self.assertAlmostEqual(s["cv_c"], 0.5)
        self.assertAlmostEqual(s["corr_nc"], 1.0)       # perfectly aligned
        self.assertEqual(s["sum_n_sq"], 100 ** 2 + 300 ** 2)
        self.assertEqual(s["sum_nc"], 100 * 1000 + 300 * 3000)
        self.assertEqual(s["sum_n_ctx"], 100 * 1100 + 300 * 3300)
        self.assertAlmostEqual(s["sum_n_ctx_causal"], 100 * (1000 + 50.5) + 300 * (3000 + 150.5))
        self.assertEqual(s["total_kv_tokens"], 1100 + 3300)

    def test_cv_zero_when_homogeneous(self):
        s = BatchConfig("t", "x", [(1024, 4096)] * 8).stats()
        self.assertEqual(s["cv_n"], 0.0)
        self.assertEqual(s["cv_c"], 0.0)
        self.assertEqual(s["corr_nc"], 0.0)

    def test_cv_is_zero_not_nan_when_mean_zero(self):
        s = BatchConfig("t", "x", [(64, 0)] * 4).stats()   # all c_i = 0
        self.assertEqual(s["cv_c"], 0.0)
        self.assertEqual(s["mean_c"], 0.0)

    def test_kv_tokens_needed_rounds_up_per_request(self):
        # the scheduler allocates whole 16-token blocks per request
        self.assertEqual(BatchConfig("t", "x", [(1, 0)]).kv_tokens_needed(), 16)
        self.assertEqual(BatchConfig("t", "x", [(16, 0)]).kv_tokens_needed(), 16)
        self.assertEqual(BatchConfig("t", "x", [(17, 0)]).kv_tokens_needed(), 32)
        self.assertEqual(BatchConfig("t", "x", [(17, 0)] * 3).kv_tokens_needed(), 96)
        # rounding is per request, not on the total: 3×17 = 51 would round to 64
        self.assertNotEqual(BatchConfig("t", "x", [(17, 0)] * 3).kv_tokens_needed(), 64)

    def test_shape_key_is_order_invariant(self):
        a = BatchConfig("e", "a", [(1, 16), (2, 32)])
        b = BatchConfig("e", "b", [(2, 32), (1, 16)])
        c = BatchConfig("e", "c", [(1, 32), (2, 16)])     # different pairing
        self.assertEqual(a.shape_key, b.shape_key)
        self.assertNotEqual(a.shape_key, c.shape_key)

    def test_group_by_shape_dedups(self):
        cfgs = [BatchConfig("e", "a", [(1, 16)]), BatchConfig("e", "b", [(1, 16)]),
                BatchConfig("e", "c", [(2, 16)])]
        groups = group_by_shape(cfgs)
        self.assertEqual(len(groups), 2)
        self.assertEqual([c.name for c in groups[cfgs[0].shape_key]], ["a", "b"])


class TestGenerators(unittest.TestCase):
    def test_exp0_invariants(self):
        cfgs = exp0_attention_dominance()
        dom = [c for c in cfgs if c.name.startswith("dom_")]
        budget = [c for c in cfgs if c.name.startswith("budget")]
        self.assertTrue(dom and budget)
        for c in dom:
            self.assertEqual(sum(c.ns), TOKEN_BUDGET)     # fixed budget for c* detection
            self.assertEqual(len(set(c.ns)), 1)           # uniform n
            self.assertEqual(len(set(c.cs)), 1)           # uniform c
            self.assertTrue(c.group.startswith("B="))
        self.assertEqual({c.batch_size for c in dom}, {1, 8, 64, 256})
        self.assertIn(512 * 1024, {c.cs[0] for c in dom})  # c range reaches 512K
        for c in budget:
            self.assertEqual(c.batch_size, 8)
        # the budget sweep is what gives Σn variance to the "latency ~ Σn" baseline
        self.assertEqual(sorted({sum(c.ns) for c in budget}),
                         [1024, 2048, 4096, 8192, 16384, 32768, 65536])

    def test_exp1_fixed_budget_and_alignment(self):
        cfgs = exp1_fragmentation()
        for c in cfgs:
            self.assertEqual(sum(c.ns), TOKEN_BUDGET)
            self.assertEqual(len(set(c.ns)), 1)
            self.assertEqual(c.cs[0] % BLOCK_SIZE, 0)
            self.assertEqual(c.group, f"c={c.cs[0]}")
        self.assertEqual(sorted({c.batch_size for c in cfgs}),
                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(min(c.ns[0] for c in cfgs), TOKEN_BUDGET // 512)

    def test_exp2_variants_sum_and_labels_match_cv(self):
        for label, ns in N_VARIANTS.items():
            self.assertEqual(sum(ns), TOKEN_BUDGET, label)
            self.assertEqual(len(ns), 8, label)
            cv = BatchConfig("t", "x", [(n, 0) for n in ns]).stats()["cv_n"]
            self.assertAlmostEqual(cv, float(label[2:]), places=2,
                                   msg=f"{label} labels CV {cv:.4f}")
        cfgs = exp2_n_heterogeneity(BASE_C)
        # mean(n) is held constant so only the shape of {n_i} varies
        self.assertEqual({c.stats()["mean_n"] for c in cfgs}, {TOKEN_BUDGET / 8})
        self.assertEqual(sorted({c.cs[0] for c in cfgs}), [0, BASE_C, 2 * BASE_C, 4 * BASE_C])

    def test_exp3_is_ai_neutral(self):
        """The point of exp3: with n uniform, varying CV(c) leaves both the
        attention-work numerator and the byte denominator unchanged."""
        cfgs = exp3_c_heterogeneity(BASE_C)
        stats = [c.stats() for c in cfgs]
        for s in stats:
            self.assertEqual(s["sum_c"], 8 * BASE_C)
            self.assertEqual(s["mean_c"], BASE_C)
            self.assertEqual(s["token_budget"], 8 * 1024)
            self.assertEqual(s["cv_n"], 0.0)
        self.assertEqual(len({s["sum_n_ctx"] for s in stats}), 1)
        self.assertEqual(len({s["sum_n_ctx_causal"] for s in stats}), 1)
        self.assertEqual(len({s["sum_nc"] for s in stats}), 1)
        # ... while CV(c) genuinely differs
        self.assertEqual(len({round(s["cv_c"], 3) for s in stats}), len(cfgs))
        for label, template in C_TEMPLATES.items():
            self.assertEqual(sum(template), 256, label)
            cv = BatchConfig("t", "x", [(1, t) for t in template]).stats()["cv_c"]
            self.assertAlmostEqual(cv, float(label[2:]), places=2,
                                   msg=f"{label} labels CV {cv:.4f}")

    def test_exp3_alignment_for_every_valid_base_c(self):
        for base_c in (8192, 16384, 32768, 65536):
            for c in exp3_c_heterogeneity(base_c):
                self.assertEqual(c.stats()["sum_c"], 8 * base_c)
                for ci in c.cs:
                    self.assertEqual(ci % BLOCK_SIZE, 0)

    def test_exp3_rejects_unaligned_base_c(self):
        with self.assertRaises(AssertionError):
            exp3_c_heterogeneity(10000)

    def test_exp4_holds_multisets_fixed_and_varies_only_pairing(self):
        cfgs = exp4_correlation(BASE_C)
        self.assertGreaterEqual(len(cfgs), 5)
        ns = [sorted(c.ns) for c in cfgs]
        cs = [sorted(c.cs) for c in cfgs]
        self.assertEqual(len({tuple(x) for x in ns}), 1, "{n_i} multiset must be identical")
        self.assertEqual(len({tuple(x) for x in cs}), 1, "{c_i} multiset must be identical")
        stats = [c.stats() for c in cfgs]
        for key in ("token_budget", "sum_c", "batch_size", "cv_n", "cv_c", "sum_n_sq"):
            self.assertEqual(len({s[key] for s in stats}), 1, f"{key} must not vary")
        rhos = [s["corr_nc"] for s in stats]
        self.assertEqual(rhos, sorted(rhos), "configs should be ordered by increasing ρ")
        self.assertLess(rhos[0], -0.4)
        self.assertGreater(rhos[-1], 0.9)
        # Σ n_i·c_i is what the pairing actually changes — the point of the experiment
        self.assertGreater(stats[-1]["sum_nc"], 2 * stats[0]["sum_nc"])
        for c, s in zip(cfgs, stats):
            self.assertAlmostEqual(float(c.name[3:]), s["corr_nc"], places=2,
                                   msg="config name must report the achieved ρ")

    def test_exp4_templates(self):
        self.assertEqual(sum(EXP4_NS), TOKEN_BUDGET)
        self.assertEqual(sum(EXP4_C_TEMPLATE), 256)
        self.assertEqual(len(EXP4_NS), len(EXP4_C_TEMPLATE))

    def test_all_configs_selects_experiments(self):
        only = all_configs(["exp3"], BASE_C)
        self.assertEqual({c.exp for c in only}, {"exp3"})
        every = all_configs(base_c=BASE_C)
        self.assertEqual({c.exp for c in every}, {"exp0", "exp1", "exp2", "exp3", "exp4"})
        self.assertLess(len(group_by_shape(every)), len(every), "duplicate shapes should exist")

    def test_names_unique_within_experiment(self):
        for exp in ("exp0", "exp1", "exp2", "exp3", "exp4"):
            names = [c.name for c in all_configs([exp], BASE_C)]
            self.assertEqual(len(names), len(set(names)), exp)


if __name__ == "__main__":
    unittest.main()
