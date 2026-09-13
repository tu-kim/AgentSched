"""Behaviour of the analysis pipeline (loading, effect decomposition, c* detection)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bench import analyze


def row(**kw):
    """A result row with every field analyze.py reads; override what a test cares about."""
    base = dict(
        skipped=False, exp="exp2", name="cv0.00_c16384", group="c=16384",
        model="Qwen/Qwen1.5-1.8B", model_arch="MHA", model_kv_bytes_per_token=196608,
        model_attn_crossover_ctx=12352.0,
        model_attn_crossover_by_n={"64": 12352.0, "1024": 12352.0, "8192": 12352.0},
        token_budget=8192, batch_size=8, sum_c=131072, mean_n=1024.0, cv_n=0.0, max_n=1024,
        mean_c=16384.0, cv_c=0.0, max_c=16384, corr_nc=0.0, sum_n_sq=8388608,
        sum_nc=134217728, sum_n_ctx=142606336, sum_n_ctx_causal=142000000.0,
        total_kv_tokens=139264, kv_tokens_needed=139264, shape_key="1024,16384|" * 8,
        est_flops_total=1e14, est_flops_attn=5e13, est_flops_linear=5e13,
        est_attn_flop_frac=0.5, est_bytes_analytic=5e10, est_attn_bytes=4e10,
        est_weight_bytes=1e10, est_act_bytes_approx=1e10, est_bytes_total=6e10,
        est_kv_read_bytes=3e10, est_ai_attn=1000.0, est_ai_analytic=2000.0,
        est_arith_intensity=1666.0,
        latency_median_s=0.1, throughput_tok_s=81920.0, achieved_tflops=1000.0,
        est_hbm_gbps=500.0, gpu_util=95.0, mem_util=60.0, power_w=300.0,
        co_scheduled=True, cache_hits_ok=True, extra_steps_max=0,
        cached_tokens=131072, cached_tokens_expected=131072, wall_s=1.0,
    )
    base.update(kw)
    return base


def write(rows, path):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_splits_measured_and_skipped(self):
        p = write([row(), row(skipped=True, reason="too big", name="x")], self.dir / "r.jsonl")
        df, skipped = analyze.load([p])
        self.assertEqual(len(df), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][3], "too big")

    def test_all_skipped_returns_empty_frame_not_keyerror(self):
        p = write([row(skipped=True, reason="no capacity")], self.dir / "r.jsonl")
        df, skipped = analyze.load([p])
        self.assertTrue(df.empty)
        self.assertEqual(len(skipped), 1)

    def test_model_short_strips_org(self):
        p = write([row()], self.dir / "r.jsonl")
        df, _ = analyze.load([p])
        self.assertEqual(df.model_short.iloc[0], "Qwen1.5-1.8B")

    def test_concatenates_multiple_files(self):
        a = write([row(model="A/one")], self.dir / "a.jsonl")
        b = write([row(model="B/two")], self.dir / "b.jsonl")
        df, _ = analyze.load([a, b])
        self.assertEqual(sorted(df.model_short), ["one", "two"])

    def test_kernel_time_fractions(self):
        p = write([row(kernel_time_attention_us=600.0, kernel_time_gemm_us=200.0,
                       kernel_time_moe_us=100.0, kernel_time_activation_us=50.0,
                       kernel_time_kvcache_us=25.0, kernel_time_other_us=25.0)],
                  self.dir / "r.jsonl")
        df, _ = analyze.load([p])
        self.assertAlmostEqual(df.attn_time_frac.iloc[0], 0.6)
        self.assertAlmostEqual(df.gemm_time_frac.iloc[0], 0.3)   # gemm + moe

    def test_missing_kernel_columns_give_nan_not_crash(self):
        p = write([row()], self.dir / "r.jsonl")
        df, _ = analyze.load([p])
        self.assertTrue(np.isnan(df.attn_time_frac.iloc[0]))


class TestEfficiencyColumns(unittest.TestCase):
    def _df(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = write(rows, Path(d) / "r.jsonl")
            df, _ = analyze.load([p])
        return analyze.add_efficiency_columns(df)

    def test_reference_row_has_unit_ratios(self):
        df = self._df([
            row(name="cv0.00_c16384", cv_n=0.0, latency_median_s=0.10, achieved_tflops=1000.0,
                throughput_tok_s=81920.0, sum_n_ctx_causal=1.0e8),
            row(name="cv1.87_c16384", cv_n=1.87, latency_median_s=0.15, achieved_tflops=800.0,
                throughput_tok_s=54613.0, sum_n_ctx_causal=1.2e8),
        ])
        ref = df[df.cv_n == 0.0].iloc[0]
        self.assertAlmostEqual(ref.latency_ratio, 1.0)
        self.assertAlmostEqual(ref.work_ratio, 1.0)
        self.assertAlmostEqual(ref.tflops_ratio, 1.0)
        self.assertAlmostEqual(ref.batching_efficiency, 1.0)
        het = df[df.cv_n > 0].iloc[0]
        self.assertAlmostEqual(het.latency_ratio, 1.5)
        self.assertAlmostEqual(het.work_ratio, 1.2)
        self.assertAlmostEqual(het.tflops_ratio, 0.8)

    def test_decomposition_identity(self):
        """latency_ratio == flops_ratio / tflops_ratio holds exactly, because
        achieved_tflops is est_flops_total / latency. work_ratio is the
        attention-only part and only approximates flops_ratio."""
        df = self._df([
            row(name="cv0.00_c16384", cv_n=0.0, latency_median_s=0.10,
                sum_n_ctx_causal=1.0e8, est_flops_total=1e14, achieved_tflops=1e14 / 0.10 / 1e12),
            row(name="cv1.03_c16384", cv_n=1.03, latency_median_s=0.13,
                sum_n_ctx_causal=1.1e8, est_flops_total=1.05e14,
                achieved_tflops=1.05e14 / 0.13 / 1e12),
        ])
        for _, r in df.iterrows():
            self.assertAlmostEqual(r.latency_ratio, r.flops_ratio / r.tflops_ratio, places=9)
        het = df[df.cv_n > 0].iloc[0]
        self.assertAlmostEqual(het.flops_ratio, 1.05)
        self.assertAlmostEqual(het.work_ratio, 1.10)
        self.assertAlmostEqual(het.latency_ratio, 1.30)

    def test_groups_and_models_do_not_cross_contaminate(self):
        df = self._df([
            row(name="cv0.00_c0", group="c=0", cv_n=0.0, latency_median_s=0.05),
            row(name="cv1.87_c0", group="c=0", cv_n=1.87, latency_median_s=0.10),
            row(name="cv0.00_c16384", group="c=16384", cv_n=0.0, latency_median_s=0.20),
            row(name="cv1.87_c16384", group="c=16384", cv_n=1.87, latency_median_s=0.30),
            row(name="cv0.00_c0", group="c=0", cv_n=0.0, latency_median_s=0.40,
                model="other/Model"),
            row(name="cv1.87_c0", group="c=0", cv_n=1.87, latency_median_s=0.80,
                model="other/Model"),
        ])
        by = {(r.model_short, r.group, r.cv_n): r.latency_ratio for _, r in df.iterrows()}
        self.assertAlmostEqual(by[("Qwen1.5-1.8B", "c=0", 1.87)], 2.0)
        self.assertAlmostEqual(by[("Qwen1.5-1.8B", "c=16384", 1.87)], 1.5)
        self.assertAlmostEqual(by[("Model", "c=0", 1.87)], 2.0)

    def test_exp4_reference_is_the_rho_closest_to_zero(self):
        df = self._df([
            row(exp="exp4", name="rho-0.62", group="mean_c=16384", corr_nc=-0.62,
                latency_median_s=0.08),
            row(exp="exp4", name="rho-0.09", group="mean_c=16384", corr_nc=-0.09,
                latency_median_s=0.10),
            row(exp="exp4", name="rho+0.98", group="mean_c=16384", corr_nc=0.98,
                latency_median_s=0.20),
        ])
        ref = df[df.corr_nc == -0.09].iloc[0]
        self.assertAlmostEqual(ref.latency_ratio, 1.0)
        self.assertAlmostEqual(df[df.corr_nc == 0.98].iloc[0].latency_ratio, 2.0)

    def test_untouched_experiments_keep_nan(self):
        df = self._df([row(exp="exp1", name="B8_n1024_c16384", group="c=16384")])
        self.assertTrue(np.isnan(df.latency_ratio.iloc[0]))


class TestFitR2(unittest.TestCase):
    def test_perfect_linear_fit(self):
        d = pd.DataFrame({"latency_median_s": [3.0, 5.0, 7.0, 9.0], "x": [1.0, 2.0, 3.0, 4.0]})
        r2, beta = analyze.fit_r2(d, ["x"])
        self.assertAlmostEqual(r2, 1.0, places=9)
        self.assertAlmostEqual(beta[0], 2.0)     # slope
        self.assertAlmostEqual(beta[1], 1.0)     # intercept

    def test_uninformative_feature_gives_zero_r2(self):
        d = pd.DataFrame({"latency_median_s": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 1.0, 1.0, 1.0]})
        r2, _ = analyze.fit_r2(d, ["x"])
        self.assertAlmostEqual(r2, 0.0, places=9)

    def test_adding_features_cannot_reduce_r2(self):
        rng = np.random.default_rng(0)
        n = 40
        a, b = rng.normal(size=n), rng.normal(size=n)
        d = pd.DataFrame({"latency_median_s": 2 * a + 0.5 * b + rng.normal(scale=.1, size=n),
                          "a": a, "b": b})
        self.assertLessEqual(analyze.fit_r2(d, ["a"])[0] - 1e-12, analyze.fit_r2(d, ["a", "b"])[0])


class TestExp0Crossover(unittest.TestCase):
    def _run(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = write(rows, Path(d) / "r.jsonl")
            df, _ = analyze.load([p])
            lines = []
            cstar = analyze.analyze_exp0(df, Path(d), lines)
        return cstar, lines

    def _dom(self, c, frac, lat, group="B=8"):
        return row(exp="exp0", name=f"dom_B8_n1024_c{c}", group=group, mean_c=float(c),
                   latency_median_s=lat, est_attn_flop_frac=frac,
                   kernel_time_attention_us=frac * 1000, kernel_time_gemm_us=(1 - frac) * 1000,
                   kernel_time_moe_us=0.0, kernel_time_activation_us=0.0,
                   kernel_time_kvcache_us=0.0, kernel_time_other_us=0.0)

    def test_uses_first_c_whose_measured_attention_share_reaches_half(self):
        cstar, _ = self._run([self._dom(0, 0.05, 0.10), self._dom(4096, 0.2, 0.12),
                              self._dom(16384, 0.55, 0.20), self._dom(65536, 0.8, 0.50)])
        self.assertEqual(cstar[("Qwen1.5-1.8B", "B=8")][0], 16384)
        self.assertIn("50%", cstar[("Qwen1.5-1.8B", "B=8")][1])

    def test_reports_none_when_never_reached(self):
        cstar, _ = self._run([self._dom(0, 0.01, 0.10), self._dom(4096, 0.05, 0.11)])
        self.assertIsNone(cstar[("Qwen1.5-1.8B", "B=8")][0])

    def test_falls_back_to_latency_doubling_without_kernel_profile(self):
        rows = [row(exp="exp0", name=f"dom_B8_n1024_c{c}", group="B=8", mean_c=float(c),
                    latency_median_s=lat)
                for c, lat in [(0, 0.10), (4096, 0.15), (16384, 0.25), (65536, 0.60)]]
        cstar, _ = self._run(rows)
        self.assertEqual(cstar[("Qwen1.5-1.8B", "B=8")][0], 16384)   # first ≥ 2× the c=0 row
        self.assertIn("2×", cstar[("Qwen1.5-1.8B", "B=8")][1])

    def test_missing_baseline_is_reported_not_crashed(self):
        cstar, lines = self._run([self._dom(4096, 0.2, 0.12), self._dom(16384, 0.55, 0.20)])
        self.assertNotIn(("Qwen1.5-1.8B", "B=8"), cstar)
        self.assertTrue(any("no c=0 row" in l for l in lines))

    def test_budget_rows_are_excluded_from_crossover(self):
        rows = [self._dom(0, 0.05, 0.10), self._dom(16384, 0.55, 0.20),
                row(exp="exp0", name="budget1024_B8_n128_c0", group="budget,c=0", mean_c=0.0)]
        cstar, _ = self._run(rows)
        self.assertEqual(set(cstar), {("Qwen1.5-1.8B", "B=8")})

    def test_no_exp0_rows_returns_empty(self):
        cstar, _ = self._run([row(exp="exp1", name="B8_n1024_c16384", group="c=16384")])
        self.assertEqual(cstar, {})


class TestEndToEnd(unittest.TestCase):
    def test_synth_results_through_analyze(self):
        from bench import synth_results
        import sys
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "synth.jsonl", Path(d) / "out"
            argv = sys.argv
            sys.argv = ["synth_results", "--out", str(raw), "--models",
                        "qwen1.5-1.8b", "qwen1.5-moe-a2.7b"]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    synth_results.main()
                    sys.argv = ["analyze", "--raw", str(raw), "--outdir", str(out)]
                    analyze.main()
            finally:
                sys.argv = argv
            for f in ("summary.csv", "report.txt", "exp0_dominance.png",
                      "exp1_fragmentation.png", "exp2_n_hetero.png",
                      "exp3_c_hetero.png", "exp4_pairing.png"):
                self.assertTrue((out / f).exists(), f)
            summary = pd.read_csv(out / "summary.csv")
            self.assertEqual(set(summary.model_short), {"Qwen1.5-1.8B", "Qwen1.5-MoE-A2.7B"})
            for dropped in ("pairs", "latency_all_s", "kernel_top", "kernel_unclassified"):
                self.assertNotIn(dropped, summary.columns)
            report = (out / "report.txt").read_text()
            self.assertIn("Exp0: attention-dominance crossover", report)
            self.assertIn("R²(Σn only", report)


if __name__ == "__main__":
    unittest.main()
