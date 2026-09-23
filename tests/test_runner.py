"""Behaviour of the runner's GPU-independent parts.

The kernel classifier and bucket validation are exercised here against the same
verified kernel-name table that bench/selftest.py uses, plus the ordering
guarantees that a careless edit to the key tuples would break.
"""
import random
import types
import unittest

from bench.runner import (ACT_KEYS, ATTN_KEYS, GEMM_KEYS, KVCACHE_KEYS, MOE_KEYS, classify,
                          flash_attn_version, kv_capacity_tokens, rand_tokens, resolve_base_c,
                          validate_buckets)
from bench.selftest import KERNEL_CASES


class TestClassify(unittest.TestCase):
    def test_verified_kernel_names(self):
        for name, expected, why in KERNEL_CASES:
            self.assertEqual(classify(name), expected, f"{name[:90]} ({why})")

    def test_is_case_insensitive(self):
        self.assertEqual(classify("VOID FLASH::FLASH_FWD_SPLITKV_KERNEL<...>"), "attention")
        self.assertEqual(classify("vllm::MoeSoftmax<256>"), "moe")

    def test_dense_kv_write_is_kvcache(self):
        """The dense KV-write kernel is named ..._cache_flash_kernel. It is safe
        only because the attention keys are 'flash::' / 'flash_fwd' rather than a
        bare 'flash' — broadening them would swallow every KV write."""
        name = "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, ...>(...)"
        self.assertEqual(classify(name), "kvcache")
        self.assertFalse(any(k in name.lower() for k in ATTN_KEYS),
                         "no attention key may match the dense KV-write kernel")

    def test_mla_cache_write_beats_attention(self):
        """concat_and_cache_mla really does match the '_mla' attention key, so
        this one depends on KVCACHE_KEYS being checked first."""
        name = "void vllm::concat_and_cache_mla_kernel<...>(...)"
        self.assertEqual(classify(name), "kvcache")
        self.assertTrue(any(k in name.lower() for k in ATTN_KEYS),
                        "the ordering guarantee is load-bearing here")

    def test_mla_context_gather_is_attention_not_kvcache(self):
        name = "void vllm::gather_and_maybe_dequant_cache_page<__nv_bfloat16, ...>(...)"
        self.assertEqual(classify(name), "attention")
        self.assertFalse(any(k in name.lower() for k in KVCACHE_KEYS),
                         "must not collide with the '_and_cache' key")

    def test_fa3_beats_gemm(self):
        """FA3's demangled name contains cutlass::device_kernel; if the GEMM keys
        were checked first, all H100 attention time would be misfiled."""
        name = ("void cutlass::device_kernel<flash::enable_sm90_or_later<"
                "flash::FlashAttnFwdSm90<...>>>(...)")
        self.assertEqual(classify(name), "attention")

    def test_moe_beats_activation_and_gemm(self):
        self.assertEqual(classify("fused_moe_kernel"), "moe")
        self.assertEqual(classify("void vllm::moe::moe_sum_vec_kernel<...>"), "moe")

    def test_activation_beats_gemm(self):
        # 'silu' must be consulted before the GEMM keys for inductor fusions that
        # happen to mention a matmul-ish op
        self.assertEqual(classify("triton_poi_fused_mul_silu_3"), "activation")
        self.assertEqual(classify("void vllm::act_and_mul_kernel<...>"), "activation")

    def test_unknown_kernel_is_other(self):
        self.assertEqual(classify("some_kernel_nobody_has_seen"), "other")
        self.assertEqual(classify(""), "other")

    def test_key_tuples_are_lowercase(self):
        """classify() lowercases the name, so an upper-case key could never match."""
        for keys in (KVCACHE_KEYS, ATTN_KEYS, MOE_KEYS, ACT_KEYS, GEMM_KEYS):
            for k in keys:
                self.assertEqual(k, k.lower(), k)

    def test_dense_prefill_attention_is_matched(self):
        """On A100 vLLM always passes a block_table, so prefill uses the split-KV
        kernel — matching only 'flash_fwd_kernel' would lose all attention time."""
        self.assertEqual(classify("void flash::flash_fwd_splitkv_kernel<...>(...)"), "attention")


class TestValidateBuckets(unittest.TestCase):
    def full(self, **kw):
        b = dict(attention=1.0, gemm=1.0, moe=0.0, activation=0.1, kvcache=0.1, other=0.1)
        b.update(kw)
        return b

    def test_healthy_dense_and_moe(self):
        self.assertEqual(validate_buckets(self.full(), is_moe=False), [])
        self.assertEqual(validate_buckets(self.full(moe=2.0), is_moe=True), [])

    def test_empty_attention_is_flagged(self):
        self.assertTrue(any("attention" in w for w in
                            validate_buckets(self.full(attention=0.0), is_moe=False)))

    def test_moe_model_without_moe_time_is_flagged(self):
        self.assertTrue(any("MoE model" in w for w in
                            validate_buckets(self.full(), is_moe=True)))

    def test_dense_model_with_moe_time_is_flagged(self):
        self.assertTrue(any("dense model" in w for w in
                            validate_buckets(self.full(moe=1.0), is_moe=False)))

    def test_large_other_is_flagged(self):
        self.assertTrue(any("other" in w for w in
                            validate_buckets(self.full(other=10.0), is_moe=False)))

    def test_activation_does_not_count_towards_other(self):
        self.assertEqual(validate_buckets(self.full(activation=10.0), is_moe=False), [])

    def test_no_kernels_captured(self):
        self.assertEqual(validate_buckets({k: 0.0 for k in self.full()}, is_moe=False),
                         ["no CUDA kernels captured"])

    def test_missing_bucket_keys_are_tolerated(self):
        """A result from an older run may not have every bucket."""
        self.assertEqual(validate_buckets(dict(attention=1.0, gemm=1.0), is_moe=False), [])


class TestTokenHelpers(unittest.TestCase):
    def test_rand_tokens_length_and_range(self):
        rng = random.Random(0)
        toks = rand_tokens(rng, 500, vocab=1000)
        self.assertEqual(len(toks), 500)
        self.assertTrue(all(10 <= t < 990 for t in toks))

    def test_rand_tokens_is_deterministic_per_seed(self):
        self.assertEqual(rand_tokens(random.Random(7), 20, 1000),
                         rand_tokens(random.Random(7), 20, 1000))

    def test_rand_tokens_are_not_all_equal(self):
        """Phase B relies on fresh suffixes never colliding with cached blocks."""
        toks = rand_tokens(random.Random(1), 256, 151936)
        self.assertGreater(len(set(toks)), 200)

    def test_empty_request(self):
        self.assertEqual(rand_tokens(random.Random(0), 0, 1000), [])

    def test_base_c_resolution_is_shared_with_the_planner(self):
        """bench.configs' plan CLI and the runner must resolve --base-c auto the
        same way, or the plan describes a different config set than the run."""
        from bench import configs
        self.assertIs(resolve_base_c, configs.resolve_base_c)


class TestEngineIntrospection(unittest.TestCase):
    def _engine(self, num_gpu_blocks, block_size=16):
        cache = types.SimpleNamespace(num_gpu_blocks=num_gpu_blocks, block_size=block_size)
        return types.SimpleNamespace(vllm_config=types.SimpleNamespace(cache_config=cache))

    def test_capacity_excludes_the_null_block(self):
        self.assertEqual(kv_capacity_tokens(self._engine(1000)), 999 * 16)
        self.assertEqual(kv_capacity_tokens(self._engine(1000, block_size=32)), 999 * 32)

    def test_capacity_unknown_before_profiling(self):
        self.assertIsNone(kv_capacity_tokens(self._engine(None)))
        self.assertIsNone(kv_capacity_tokens(self._engine(0)))

    def test_flash_attn_version_is_reported_or_none(self):
        """Contract: return the FA version vLLM resolved, or None when it cannot
        be determined (no vLLM, no CUDA, ROCm) — never raise. The value is
        recorded in raw.jsonl because it decides the kernel names, whether MLA
        pads V, and whether fp8 KV is available.

        vLLM picks 3 only on SM90 (Hopper) and 4 on SM100 (Blackwell); an A100
        reports 2. A mismatch with the GPU you think you are on is worth
        checking before trusting a measurement run.
        """
        v = flash_attn_version()
        self.assertIn(v, (None, 2, 3, 4), f"unexpected FlashAttention version {v!r}")


class TestFeasibilityGate(unittest.TestCase):
    """The capacity gate the runner applies before touching the GPU must agree
    with what bench.configs' planner reports, or the plan and the run diverge."""

    def test_gate_matches_planner(self):
        from bench.configs import all_configs
        from bench.metrics import MODEL_PRESETS
        spec = MODEL_PRESETS["qwen1.5-1.8b"]
        capacity = spec.kv_capacity_tokens(80)
        margin = 0.95
        cfgs = all_configs(["exp1"], 16384)
        planned = [c for c in cfgs if c.kv_tokens_needed() <= capacity * margin]
        # the runner's own predicate, copied from run_shape
        run = [c for c in cfgs
               if not (capacity and c.stats()["kv_tokens_needed"] > capacity * margin)]
        self.assertEqual([c.name for c in planned], [c.name for c in run])
        self.assertTrue(planned and len(planned) < len(cfgs), "some configs must be skipped")


if __name__ == "__main__":
    unittest.main()
