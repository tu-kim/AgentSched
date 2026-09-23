"""Behaviour of ModelSpec parsing and the analytic FLOPs / bytes model."""
import unittest

from bench.metrics import MODEL_PRESETS, ModelSpec, estimate_flops_bytes

GB = 1e9

# HF config.json fragments (verified against the real repos). Keyed by the
# MODEL_PRESETS entry they should reproduce.
HF_CONFIGS = {
    "qwen1.5-1.8b": dict(
        model_type="qwen2", num_hidden_layers=24, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=5504, vocab_size=151936,
        max_position_embeddings=32768, tie_word_embeddings=False),
    "qwen2.5-3b": dict(
        model_type="qwen2", num_hidden_layers=36, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=2, intermediate_size=11008, vocab_size=151936,
        max_position_embeddings=32768, tie_word_embeddings=True),
    "qwen3-1.7b": dict(
        model_type="qwen3", num_hidden_layers=28, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, intermediate_size=6144, vocab_size=151936,
        max_position_embeddings=40960, tie_word_embeddings=True),
    "llama3.2-1b": dict(
        model_type="llama", num_hidden_layers=16, hidden_size=2048, num_attention_heads=32,
        num_key_value_heads=8, head_dim=64, intermediate_size=8192, vocab_size=128256,
        max_position_embeddings=131072, tie_word_embeddings=True),
    "qwen1.5-moe-a2.7b": dict(
        model_type="qwen2_moe", num_hidden_layers=24, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=5632, vocab_size=151936,
        max_position_embeddings=8192, tie_word_embeddings=False,
        num_experts=60, num_experts_per_tok=4, moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632, decoder_sparse_step=1),
    "olmoe-1b-7b": dict(
        model_type="olmoe", num_hidden_layers=16, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=1024, vocab_size=50304,
        max_position_embeddings=4096, tie_word_embeddings=False,
        num_experts=64, num_experts_per_tok=8),
    "deepseek-v2-lite": dict(
        model_type="deepseek_v2", num_hidden_layers=27, hidden_size=2048, num_attention_heads=16,
        num_key_value_heads=16, intermediate_size=10944, vocab_size=102400,
        max_position_embeddings=163840, tie_word_embeddings=False,
        kv_lora_rank=512, q_lora_rank=None, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, n_routed_experts=64, num_experts_per_tok=6, moe_intermediate_size=1408,
        n_shared_experts=2, first_k_dense_replace=1),
}


class TestFromHFConfig(unittest.TestCase):
    def test_presets_match_parsed_configs(self):
        """The hand-entered MODEL_PRESETS must equal what the runner derives from
        the real config.json, otherwise plans and measurements disagree."""
        for key, hf in HF_CONFIGS.items():
            parsed = ModelSpec.from_hf_config(hf, name=MODEL_PRESETS[key].name)
            preset = MODEL_PRESETS[key]
            for field in ("n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim", "inter",
                          "vocab", "max_position", "tied_embeddings", "mlp_gated",
                          "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                          "v_head_dim", "n_experts", "top_k", "moe_inter", "shared_inter",
                          "n_dense_layers"):
                self.assertEqual(getattr(parsed, field), getattr(preset, field),
                                 f"{key}.{field}")

    def test_head_dim_defaults_to_hidden_over_heads(self):
        s = ModelSpec.from_hf_config(dict(num_hidden_layers=2, hidden_size=512,
                                          num_attention_heads=8, intermediate_size=1,
                                          vocab_size=10))
        self.assertEqual(s.head_dim, 64)

    def test_kv_heads_default_to_mha(self):
        s = ModelSpec.from_hf_config(dict(num_hidden_layers=2, hidden_size=512,
                                          num_attention_heads=8, intermediate_size=1,
                                          vocab_size=10))
        self.assertEqual(s.n_kv_heads, 8)
        self.assertTrue(s.is_mha)

    def test_non_gated_mlp_families(self):
        for mt, gated in [("qwen2", True), ("llama", True), ("gpt_neox", False),
                          ("opt", False), ("bloom", False)]:
            s = ModelSpec.from_hf_config(dict(model_type=mt, num_hidden_layers=1, hidden_size=8,
                                              num_attention_heads=1, intermediate_size=4,
                                              vocab_size=2))
            self.assertEqual(s.mlp_gated, gated, mt)

    def test_moe_key_variants(self):
        common = dict(num_hidden_layers=2, hidden_size=8, num_attention_heads=1,
                      intermediate_size=4, vocab_size=2, num_experts_per_tok=2)
        # Mixtral / OLMoE style
        a = ModelSpec.from_hf_config(dict(common, num_local_experts=8))
        # Qwen-MoE style
        b = ModelSpec.from_hf_config(dict(common, num_experts=8, moe_intermediate_size=3,
                                          shared_expert_intermediate_size=6))
        # DeepSeek style
        c = ModelSpec.from_hf_config(dict(common, n_routed_experts=8, moe_intermediate_size=3,
                                          n_shared_experts=2, first_k_dense_replace=1))
        for s in (a, b, c):
            self.assertTrue(s.is_moe)
            self.assertEqual(s.n_experts, 8)
            self.assertEqual(s.top_k, 2)
        self.assertEqual(a.moe_inter, 4)        # falls back to intermediate_size
        self.assertEqual(b.shared_inter, 6)
        self.assertEqual(c.shared_inter, 2 * 3)  # n_shared_experts × moe_intermediate_size
        self.assertEqual(c.n_dense_layers, 1)
        self.assertEqual(c.n_moe_layers, 1)

    def test_dense_model_is_not_moe(self):
        s = ModelSpec.from_hf_config(HF_CONFIGS["qwen1.5-1.8b"])
        self.assertFalse(s.is_moe)
        self.assertEqual(s.n_moe_layers, 0)
        self.assertEqual(s.linear_params_active, s.linear_params_total)


class TestDerivedProperties(unittest.TestCase):
    def test_arch_labels(self):
        self.assertEqual(MODEL_PRESETS["qwen1.5-1.8b"].arch_label, "MHA")
        self.assertEqual(MODEL_PRESETS["qwen3-1.7b"].arch_label, "GQA")
        self.assertEqual(MODEL_PRESETS["qwen1.5-moe-a2.7b"].arch_label, "MHA+MoE")
        self.assertEqual(MODEL_PRESETS["deepseek-v2-lite"].arch_label, "MLA+MoE")
        self.assertEqual(MODEL_PRESETS["mla-dense-1.8b"].arch_label, "MLA")

    def test_kv_bytes_per_token(self):
        mha = MODEL_PRESETS["qwen1.5-1.8b"]
        self.assertEqual(mha.kv_bytes_per_token, 2 * 24 * 16 * 128 * 2)
        gqa = MODEL_PRESETS["qwen3-1.7b"]
        self.assertEqual(gqa.kv_bytes_per_token, 2 * 28 * 8 * 128 * 2)
        mla = MODEL_PRESETS["deepseek-v2-lite"]
        self.assertEqual(mla.kv_bytes_per_token, 27 * (512 + 64) * 2)

    def test_gqa_shrinks_kv_by_group_size(self):
        base = dict(num_hidden_layers=4, hidden_size=512, num_attention_heads=8,
                    intermediate_size=16, vocab_size=32)
        mha = ModelSpec.from_hf_config(dict(base, num_key_value_heads=8))
        gqa = ModelSpec.from_hf_config(dict(base, num_key_value_heads=2))
        self.assertEqual(gqa.group_size, 4)
        self.assertEqual(mha.kv_bytes_per_token, 4 * gqa.kv_bytes_per_token)

    def test_fp8_kv_halves_cache_and_reads(self):
        bf16 = ModelSpec.from_hf_config(HF_CONFIGS["qwen1.5-1.8b"], kv_dtype_bytes=2)
        fp8 = ModelSpec.from_hf_config(HF_CONFIGS["qwen1.5-1.8b"], kv_dtype_bytes=1)
        self.assertEqual(bf16.kv_bytes_per_token, 2 * fp8.kv_bytes_per_token)
        pairs = [(1024, 32768)] * 4
        self.assertEqual(estimate_flops_bytes(bf16, pairs)["est_kv_read_bytes"],
                         2 * estimate_flops_bytes(fp8, pairs)["est_kv_read_bytes"])
        # weights are unaffected by the KV dtype
        self.assertEqual(bf16.weight_bytes, fp8.weight_bytes)

    def test_mla_head_dims_and_v_padding(self):
        mla = MODEL_PRESETS["deepseek-v2-lite"]
        self.assertEqual(mla.d_qk, 128 + 64)
        self.assertEqual(mla.d_v, 192, "FA2 on SM80 zero-pads V to the QK head dim")
        unpadded = ModelSpec(**{**mla.__dict__, "mla_v_padded": False})
        self.assertEqual(unpadded.d_v, 128)
        self.assertLess(unpadded.attn_flops_per_token_per_ctx, mla.attn_flops_per_token_per_ctx)

    def test_attn_crossover_for_mha_and_gqa_shifts_only_by_the_causal_offset(self):
        """For a uniform full-attention stack the crossover in *attended* context
        is a model constant; in cached context it just slides by (n+1)/2."""
        for key in ("qwen1.5-1.8b", "qwen3-1.7b", "llama2-7b", "falcon-7b"):
            spec = MODEL_PRESETS[key]
            attended = {round(spec.attn_crossover_ctx(n) + (n + 1) / 2)
                        for n in (1, 16, 64, 1024, 8192)}
            self.assertLessEqual(max(attended) - min(attended), 1, key)

    def test_mla_crossover_is_phase_dependent(self):
        """MLA is the one family where the decode and prefill kernels differ in
        kind, so c* cannot be quoted as a single number."""
        mla = MODEL_PRESETS["deepseek-v2-lite"]
        decode, prefill = mla.attn_crossover_ctx(1), mla.attn_crossover_ctx(1024)
        self.assertLess(decode, prefill / 1.5)
        mha = MODEL_PRESETS["qwen1.5-1.8b"]
        self.assertAlmostEqual(mha.attn_crossover_ctx(1) / mha.attn_crossover_ctx(1024),
                               1.0, places=1)

    def test_local_layers_delay_or_remove_the_crossover(self):
        full = MODEL_PRESETS["kv-mha-1.8b"].attn_crossover_ctx(1024)
        swa = MODEL_PRESETS["swa-1.8b"].attn_crossover_ctx(1024)
        self.assertGreater(swa, 3 * full, "windowed layers stop contributing")

    def test_kv_capacity(self):
        spec = MODEL_PRESETS["qwen1.5-1.8b"]
        big, small = spec.kv_capacity_tokens(80), spec.kv_capacity_tokens(40)
        self.assertGreater(big, small)
        self.assertGreater(small, 0)
        # a model whose weights do not fit reports zero capacity, not a negative number
        huge = ModelSpec.from_hf_config(dict(num_hidden_layers=200, hidden_size=8192,
                                             num_attention_heads=64, intermediate_size=28672,
                                             vocab_size=128000))
        self.assertEqual(huge.kv_capacity_tokens(80), 0)
        # tensor parallelism adds capacity (weights are sharded, KV pool is pooled)
        self.assertGreater(spec.kv_capacity_tokens(80, tp=2), 2 * big * 0.9)


class TestEstimateFlopsBytes(unittest.TestCase):
    def setUp(self):
        self.mha = MODEL_PRESETS["qwen1.5-1.8b"]
        self.moe = MODEL_PRESETS["qwen1.5-moe-a2.7b"]
        self.mla = MODEL_PRESETS["deepseek-v2-lite"]

    def test_attention_terms_are_additive_over_requests(self):
        one = estimate_flops_bytes(self.mha, [(1024, 4096)])
        two = estimate_flops_bytes(self.mha, [(1024, 4096), (512, 8192)])
        other = estimate_flops_bytes(self.mha, [(512, 8192)])
        self.assertAlmostEqual(two["est_flops_attn"],
                               one["est_flops_attn"] + other["est_flops_attn"], places=3)
        self.assertAlmostEqual(two["est_attn_bytes"],
                               one["est_attn_bytes"] + other["est_attn_bytes"], places=3)

    def test_linear_flops_scale_with_total_n(self):
        a = estimate_flops_bytes(self.mha, [(1024, 0)])
        b = estimate_flops_bytes(self.mha, [(1024, 0)] * 4)
        # lm_head adds a per-request term, so the ratio is slightly above 4
        self.assertGreater(b["est_flops_linear"] / a["est_flops_linear"], 3.9)
        self.assertLess(b["est_flops_linear"] / a["est_flops_linear"], 4.2)

    def test_attention_share_rises_with_context(self):
        shares = [estimate_flops_bytes(self.mha, [(1024, c)] * 8)["est_attn_flop_frac"]
                  for c in (0, 4096, 32768, 131072)]
        self.assertEqual(shares, sorted(shares))
        self.assertLess(shares[0], 0.15)
        self.assertGreater(shares[-1], 0.85)
        for s in shares:
            self.assertTrue(0.0 <= s <= 1.0)

    def test_bytes_decomposition(self):
        e = estimate_flops_bytes(self.mha, [(1024, 8192)] * 4)
        self.assertAlmostEqual(e["est_bytes_analytic"],
                               e["est_weight_bytes"] + e["est_attn_bytes"], places=3)
        self.assertAlmostEqual(e["est_bytes_total"],
                               e["est_bytes_analytic"] + e["est_act_bytes_approx"], places=3)
        self.assertAlmostEqual(e["est_ai_analytic"],
                               e["est_flops_total"] / e["est_bytes_analytic"], places=6)
        self.assertLess(e["est_arith_intensity"], e["est_ai_analytic"])

    def test_ai_attn_matches_readme_formula_for_mha(self):
        """AI_attn = Σn(n+c) / Σ(2n+c) holds exactly for MHA/bf16 once causal
        masking and the new-KV write are removed from the model."""
        pairs = [(1024, 32768), (256, 4096), (4096, 131072)]
        s, L, H, d = self.mha, self.mha.n_layers, self.mha.n_heads, self.mha.head_dim
        flops = sum(4 * H * d * L * n * (n + c) for n, c in pairs)
        byts = sum(L * (2 * n * H * d * 2 + 2 * (c + n) * H * d * 2) for n, c in pairs)
        formula = sum(n * (n + c) for n, c in pairs) / sum(2 * n + c for n, c in pairs)
        self.assertAlmostEqual(flops / byts, formula, places=6)
        self.assertTrue(s.is_mha)

    def test_moe_expert_traffic_is_a_floor(self):
        """Once every expert receives a token the weight read stops growing with Σn."""
        small = estimate_flops_bytes(self.moe, [(1, 0)])["est_weight_bytes"]
        mid = estimate_flops_bytes(self.moe, [(1024, 0)])["est_weight_bytes"]
        big = estimate_flops_bytes(self.moe, [(1024, 0)] * 8)["est_weight_bytes"]
        self.assertLess(small, mid)                  # one token touches only top_k experts
        self.assertLess(big / mid, 1.05)             # 8× the tokens, ~same weight traffic
        self.assertGreater(big / 1e9, 20)            # ~25 GB of expert weights

    def test_moe_active_params_below_total(self):
        self.assertLess(self.moe.linear_params_active, self.moe.linear_params_total / 3)
        self.assertLess(self.moe.linear_params_active / 1e9, 2.5)
        self.assertGreater(self.moe.total_params / 1e9, 14)

    def test_mla_decompression_scales_with_c_only(self):
        per_ctx = self.mla.mla_decompress_flops_per_ctx
        for n in (16, 1024):
            with_ctx = estimate_flops_bytes(self.mla, [(n, 65536)])["est_flops_attn"]
            without = estimate_flops_bytes(self.mla, [(n, 0)])["est_flops_attn"]
            core = self.mla.attn_flops_per_token_per_ctx * n * 65536
            self.assertAlmostEqual((with_ctx - without - core) / 65536, per_ctx, delta=1.0)

    def test_mla_kv_read_uses_latent_not_materialised_size(self):
        n, c = 1024, 65536
        e = estimate_flops_bytes(self.mla, [(n, c)])
        # the kernel reads the latent for every key it attends to: cached plus new
        self.assertAlmostEqual(e["est_kv_read_bytes"], (c + n) * (512 + 64) * 2 * 27, places=3)
        # in prefill the per-head K/V are additionally materialised from it
        self.assertGreater(e["est_attn_bytes"], 5 * e["est_kv_read_bytes"])

    def test_mla_decode_skips_materialisation(self):
        """Weight absorption means the decode path never expands the latent, so
        its attention bytes are dominated by the latent read itself."""
        dec = estimate_flops_bytes(self.mla, [(1, 65536)])
        self.assertLess(dec["est_attn_bytes"], 1.2 * dec["est_kv_read_bytes"])
        # and per-head score FLOPs grow, pushing decode towards compute
        self.assertGreater(self.mla.attn_flops_per_pair(1),
                           2 * self.mla.attn_flops_per_pair(1024))

    def test_zero_context_is_handled(self):
        e = estimate_flops_bytes(self.mha, [(64, 0)])
        self.assertGreater(e["est_flops_attn"], 0)
        self.assertGreater(e["est_ai_attn"], 0)

    def test_empty_batch_does_not_divide_by_zero(self):
        e = estimate_flops_bytes(self.mha, [])
        self.assertEqual(e["est_flops_attn"], 0.0)
        self.assertEqual(e["est_ai_attn"], 0.0)


if __name__ == "__main__":
    unittest.main()
