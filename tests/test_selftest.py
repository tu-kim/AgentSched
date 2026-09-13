"""Run bench/selftest.py's measurement checks as part of the unit-test suite.

selftest.py stays a standalone script (it is what you run on the GPU box before
a measurement session, and its output is meant to be read), but its assertions
should also fail CI, so they are wrapped here.
"""
import contextlib
import io
import unittest

from bench import selftest


class TestMeasurementSelfTest(unittest.TestCase):
    def _quiet(self, fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = fn()
        return ok, buf.getvalue()

    def test_kernel_classification(self):
        ok, out = self._quiet(selftest.check_classifier)
        self.assertTrue(ok, out)

    def test_analytic_model(self):
        ok, out = self._quiet(selftest.check_analytic)
        self.assertTrue(ok, out)

    def test_runtime_bucket_validation(self):
        ok, out = self._quiet(selftest.check_runtime_validation)
        self.assertTrue(ok, out)

    def test_main_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(selftest.main(), 0)

    def test_covers_every_bucket(self):
        buckets = {expected for _, expected, _ in selftest.KERNEL_CASES}
        self.assertEqual(buckets, {"attention", "kvcache", "moe", "activation", "gemm", "other"})


if __name__ == "__main__":
    unittest.main()
