"""The attention-family taxonomy: layer stacks, phase behaviour, controlled pairs.

These assert the qualitative facts the model lineup exists to demonstrate, so a
change to the analytic model that silently breaks one of them fails here.
"""
import json
import math
import unittest
from pathlib import Path

from bench.metrics import (CONTROLLED_PAIRS, MODEL_PRESETS, ModelSpec, attended_keys,
                           estimate_flops_bytes)

SYNTH = Path(__file__).resolve().parent.parent / "bench" / "synthetic_configs"


class TestAttendedKeys(unittest.TestCase):
    def test_full_attention_is_causal_triangle(self):
        self.assertEqual(attended_keys(4, 0), 1 + 2 + 3 + 4)
        self.assertEqual(attended_keys(3, 10), 11 + 12 + 13)

    def test_window_caps_the_count(self):
        self.assertEqual(attended_keys(3, 1000, window=100), 300)
        self.assertEqual(attended_keys(4, 0, window=1000), 1 + 2 + 3 + 4)   # below the cap

    def test_window_partial_crossing(self):
        # c=8, n=4, w=10 -> keys 9, 10, 10, 10
        self.assertEqual(attended_keys(4, 8, window=10), 9 + 10 + 10 + 10)

    def test_zero_tokens(self):
        self.assertEqual(attended_keys(0, 1000), 0.0)


class TestLayerStacks(unittest.TestCase):
    def test_uniform_models_are_one_group(self):
        for key in ("llama2-7b", "falcon-7b", "llama3-8b", "deepseek-v2-lite"):
            spec = MODEL_PRESETS[key]
            groups = spec.layer_groups()
            self.assertEqual(len(groups), 1, key)
            self.assertEqual(groups[0][0], spec.n_layers)
            self.assertFalse(spec.is_hybrid, key)

    def test_gemma3_local_global_split(self):
        g = MODEL_PRESETS["gemma3-4b"]
        groups = dict((kind, cnt) for cnt, kind, _ in g.layer_groups())
        self.assertEqual(groups["full"], 34 // 6)          # every 6th layer is global
        self.assertEqual(groups["local"], 34 - 34 // 6)
        self.assertEqual(sum(c for c, _, _ in g.layer_groups()), g.n_layers)
        self.assertTrue(g.is_hybrid)

    def test_qwen3_next_linear_split(self):
        q = MODEL_PRESETS["qwen3-next-80b"]
        groups = dict((kind, cnt) for cnt, kind, _ in q.layer_groups())
        self.assertEqual(groups["full"], 12)               # 48 / interval 4
        self.assertEqual(groups["linear"], 36)             # the 3:1 pattern
        self.assertEqual(q.arch_label, "Linear/GQA+MoE")

    def test_only_growing_layers_set_the_kv_rate(self):
        """A windowed or recurrent layer stops adding bytes per extra token."""
        g = MODEL_PRESETS["gemma3-4b"]
        self.assertEqual(g.n_growing_kv_layers, 34 // 6)
        per_layer = 2 * g.n_kv_heads * g.head_dim * 2
        self.assertEqual(g.kv_bytes_per_token, (34 // 6) * per_layer)
        # a full-attention model of the same shape would be ~6.8× heavier
        self.assertGreater(34 * per_layer, 6 * g.kv_bytes_per_token)

    def test_kv_footprint_saturates_for_windowed_layers(self):
        g = MODEL_PRESETS["swa-1.8b"]
        short, long_ = g.kv_bytes_for(512), g.kv_bytes_for(1 << 20)
        self.assertGreater(long_, short)
        # local layers contribute a constant once past the window
        marginal = (g.kv_bytes_for(1 << 20) - g.kv_bytes_for((1 << 20) - 1024)) / 1024
        self.assertAlmostEqual(marginal, g.kv_bytes_per_token, places=3)

    def test_linear_layers_hold_state_not_context(self):
        q = MODEL_PRESETS["qwen3-next-80b"]
        a, b = q.kv_bytes_for(1024), q.kv_bytes_for(1 << 20)
        self.assertGreater(a, 0, "the recurrent state costs memory even at short context")
        marginal = (b - a) / ((1 << 20) - 1024)
        self.assertAlmostEqual(marginal, q.kv_bytes_per_token, places=3)


class TestPhaseBehaviour(unittest.TestCase):
    def test_decode_ai_equals_the_group_size(self):
        """The textbook result the family sweep should reproduce: in bf16 the
        decode attention AI of MHA/GQA/MQA is just H/H_kv."""
        for key in ("llama2-7b", "llama3-8b", "falcon-7b", "kv-gqa4-1.8b", "kv-mqa-1.8b"):
            spec = MODEL_PRESETS[key]
            ai = estimate_flops_bytes(spec, [(1, 32768)])["est_ai_attn"]
            self.assertAlmostEqual(ai, spec.group_size, delta=0.3 * max(1, spec.group_size * 0.02),
                                   msg=f"{key}: AI {ai:.2f} vs group size {spec.group_size}")

    def test_kv_sharing_divides_decode_traffic(self):
        mha = estimate_flops_bytes(MODEL_PRESETS["kv-mha-1.8b"], [(1, 32768)])
        gqa = estimate_flops_bytes(MODEL_PRESETS["kv-gqa4-1.8b"], [(1, 32768)])
        mqa = estimate_flops_bytes(MODEL_PRESETS["kv-mqa-1.8b"], [(1, 32768)])
        self.assertAlmostEqual(mha["est_kv_read_bytes"] / gqa["est_kv_read_bytes"], 4.0, places=6)
        self.assertAlmostEqual(gqa["est_kv_read_bytes"] / mqa["est_kv_read_bytes"], 4.0, places=6)

    def test_kv_sharing_barely_changes_prefill_score_flops(self):
        """GQA/MQA change what is read, not what is computed — the premise of
        splitting prefill from decode."""
        pairs = [(1024, 32768)] * 4
        a = estimate_flops_bytes(MODEL_PRESETS["kv-mha-1.8b"], pairs)["est_flops_attn"]
        b = estimate_flops_bytes(MODEL_PRESETS["kv-mqa-1.8b"], pairs)["est_flops_attn"]
        self.assertAlmostEqual(a, b, places=3)

    def test_mla_decode_moves_towards_compute(self):
        mla = MODEL_PRESETS["mla-dense-1.8b"]
        mha = MODEL_PRESETS["kv-mha-1.8b"]
        ai_mla = estimate_flops_bytes(mla, [(1, 32768)])["est_ai_attn"]
        ai_mha = estimate_flops_bytes(mha, [(1, 32768)])["est_ai_attn"]
        self.assertGreater(ai_mla, 10 * ai_mha)
        # ... by reading less while computing more per head
        self.assertLess(estimate_flops_bytes(mla, [(1, 32768)])["est_kv_read_bytes"],
                        estimate_flops_bytes(mha, [(1, 32768)])["est_kv_read_bytes"])

    def test_windowing_caps_work_without_changing_intensity(self):
        """SWA is a capacity/throughput win, not an arithmetic-intensity win:
        it removes reads and FLOPs in the same proportion."""
        full, swa = MODEL_PRESETS["kv-mha-1.8b"], MODEL_PRESETS["swa-1.8b"]
        f = estimate_flops_bytes(full, [(1, 131072)])
        s = estimate_flops_bytes(swa, [(1, 131072)])
        self.assertLess(s["est_flops_attn"], f["est_flops_attn"] / 4)
        self.assertLess(s["est_kv_read_bytes"], f["est_kv_read_bytes"] / 4)
        self.assertAlmostEqual(s["est_ai_attn"], f["est_ai_attn"], delta=0.05)

    def test_local_layers_stop_scaling_with_context(self):
        swa = MODEL_PRESETS["swa-1.8b"]
        per_pair = swa.attn_flops_per_pair(1)
        n_global = swa.n_layers // swa.window_pattern
        # far past the window, only the global layers still grow
        d = swa.attn_flops(1, 1 << 20) - swa.attn_flops(1, (1 << 20) - 1)
        self.assertAlmostEqual(d, n_global * per_pair, delta=per_pair * 0.01)


class TestSparseAndMoE(unittest.TestCase):
    def test_dsa_caps_attention_and_shifts_cost_into_the_indexer(self):
        """The measurement question for the DSA family: does the indexer take
        over as the context grows?"""
        v3, v32 = MODEL_PRESETS["deepseek-v3"], MODEL_PRESETS["deepseek-v3.2"]
        shares = []
        for c in (4096, 65536, 1 << 20):
            total = v32.attn_flops(1, c)
            indexer = v32.n_layers * v32.indexer_flops_per_pair * attended_keys(1, c)
            shares.append(indexer / total)
            self.assertLess(total, v3.attn_flops(1, c))      # sparse is always cheaper
        self.assertEqual(shares, sorted(shares))
        self.assertLess(shares[0], 0.2)
        self.assertGreater(shares[-1], 0.9)

    def test_dsa_indexer_cache_is_mqa_shaped(self):
        """The indexer keeps one small FP8 head, not indexer_heads of them —
        otherwise it would cost more bytes than the MLA latent it filters."""
        v32 = MODEL_PRESETS["deepseek-v3.2"]
        idx_bytes = v32.indexer_head_dim * v32.indexer_dtype_bytes
        latent_bytes = (v32.kv_lora_rank + v32.qk_rope_head_dim) * v32.kv_dtype_bytes
        self.assertLess(idx_bytes, latent_bytes)

    def test_moe_pair_isolates_the_ffn(self):
        a, b, _ = CONTROLLED_PAIRS["ffn-dense-vs-moe"]
        sa, sb = MODEL_PRESETS[a], MODEL_PRESETS[b]
        for field in ("n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim"):
            self.assertEqual(getattr(sa, field), getattr(sb, field), field)
        self.assertEqual(sa.kv_bytes_per_token, sb.kv_bytes_per_token)
        self.assertFalse(sa.is_moe)
        self.assertTrue(sb.is_moe)
        # attention work is identical; only the FFN side moves
        pairs = [(1024, 16384)] * 8
        self.assertAlmostEqual(estimate_flops_bytes(sa, pairs)["est_flops_attn"],
                               estimate_flops_bytes(sb, pairs)["est_flops_attn"], places=3)

    def test_mistral_mixtral_pair_shares_attention(self):
        a, b, _ = CONTROLLED_PAIRS["ffn-dense-vs-moe-7b"]
        sa, sb = MODEL_PRESETS[a], MODEL_PRESETS[b]
        for field in ("n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim", "vocab"):
            self.assertEqual(getattr(sa, field), getattr(sb, field), field)
        self.assertEqual(sb.tier, "multi_gpu", "93GB bf16 cannot be measured on one 80GB GPU")

    def test_moe_expert_reads_grow_with_batch_not_with_flops(self):
        """Decode: more requests ⇒ more distinct experts touched, so FFN time is
        driven by weight traffic rather than by FLOPs."""
        moe = MODEL_PRESETS["qwen1.5-moe-a2.7b"]
        one = estimate_flops_bytes(moe, [(1, 4096)])
        many = estimate_flops_bytes(moe, [(1, 4096)] * 512)
        flops_ratio = many["est_flops_body"] / one["est_flops_body"]
        bytes_ratio = many["est_weight_bytes"] / one["est_weight_bytes"]
        self.assertAlmostEqual(flops_ratio, 512, delta=1)
        self.assertGreater(bytes_ratio, 1.5)      # far more experts are touched
        self.assertLess(bytes_ratio, flops_ratio)  # but it saturates


class TestPresetHygiene(unittest.TestCase):
    def test_every_preset_declares_a_family_and_tier(self):
        for key, s in MODEL_PRESETS.items():
            self.assertTrue(s.family, key)
            self.assertIn(s.tier, ("measurable", "multi_gpu", "analytic_only"), key)

    def test_tier_matches_single_gpu_feasibility(self):
        for key, s in MODEL_PRESETS.items():
            if s.tier == "measurable":
                self.assertTrue(s.fits_on(80), f"{key} is marked measurable but does not fit 80GB")
                self.assertGreater(s.kv_capacity_tokens(80), 50_000, key)
            else:
                self.assertFalse(s.fits_on(80), f"{key} fits 80GB — tier should be 'measurable'")

    def test_controlled_pairs_reference_real_presets(self):
        for name, (a, b, why) in CONTROLLED_PAIRS.items():
            self.assertIn(a, MODEL_PRESETS, name)
            self.assertIn(b, MODEL_PRESETS, name)
            self.assertTrue(why)

    def test_synthetic_configs_match_their_presets(self):
        """Each synthetic config.json must parse into exactly the preset that
        describes it, or the plan and the run diverge."""
        for key in ("kv-mha-1.8b", "kv-gqa4-1.8b", "kv-mqa-1.8b", "swa-1.8b", "mla-dense-1.8b"):
            cfg = json.loads((SYNTH / key / "config.json").read_text())
            parsed = ModelSpec.from_hf_config(cfg, name=MODEL_PRESETS[key].name)
            preset = MODEL_PRESETS[key]
            for field in ("n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim", "inter",
                          "vocab", "max_position", "tied_embeddings", "mlp_gated",
                          "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
                          "window", "window_pattern", "n_experts", "top_k"):
                self.assertEqual(getattr(parsed, field), getattr(preset, field), f"{key}.{field}")

    def test_kv_sharing_configs_differ_only_in_kv_heads(self):
        base = json.loads((SYNTH / "kv-mha-1.8b" / "config.json").read_text())
        for key in ("kv-gqa4-1.8b", "kv-mqa-1.8b"):
            other = json.loads((SYNTH / key / "config.json").read_text())
            diff = {k for k in set(base) | set(other) if base.get(k) != other.get(k)}
            self.assertEqual(diff, {"num_key_value_heads", "_comment"}, key)

    def test_falcon_parses_without_an_intermediate_size(self):
        """Falcon's config has no intermediate_size and no explicit KV heads."""
        spec = ModelSpec.from_hf_config(dict(
            model_type="falcon", num_hidden_layers=32, hidden_size=4544,
            num_attention_heads=71, multi_query=True, vocab_size=65024, parallel_attn=True))
        self.assertEqual(spec.n_kv_heads, 1)
        self.assertEqual(spec.attn_type, "mqa")
        self.assertEqual(spec.inter, 4 * 4544)
        self.assertFalse(spec.mlp_gated)
        self.assertTrue(spec.parallel_block)

    def test_crossover_can_be_infinite(self):
        """A stack with no growing attention layer never becomes attention-bound."""
        spec = ModelSpec.from_hf_config(dict(
            model_type="mistral", num_hidden_layers=8, hidden_size=1024,
            num_attention_heads=8, num_key_value_heads=8, intermediate_size=4096,
            vocab_size=1000, sliding_window=256))
        self.assertEqual(spec.layer_groups(), [(8, "local", 256)])
        self.assertEqual(spec.attn_crossover_ctx(1), math.inf)


if __name__ == "__main__":
    unittest.main()


class TestDeviceProfiles(unittest.TestCase):
    """The benchmark must run on A100 (SM80/FA2) and H100 (SM90/FA3) alike, and
    the analytic model has to follow whichever kernels that GPU will use."""

    def test_fa_version_follows_compute_capability(self):
        from bench.metrics import DEVICES
        self.assertEqual(DEVICES["A100-80GB"].fa_version, 2)
        self.assertEqual(DEVICES["H100-SXM"].fa_version, 3)
        self.assertEqual(DEVICES["H200"].fa_version, 3)
        self.assertEqual(DEVICES["B200"].fa_version, 4)

    def test_only_pre_hopper_pads_mla_v(self):
        from bench.metrics import DEVICES
        self.assertTrue(DEVICES["A100-80GB"].mla_v_padded)
        self.assertFalse(DEVICES["H100-SXM"].mla_v_padded)
        self.assertFalse(DEVICES["A100-80GB"].supports_fp8_kv_with_flash_attn)
        self.assertTrue(DEVICES["H100-SXM"].supports_fp8_kv_with_flash_attn)

    def test_for_device_adjusts_mla_and_leaves_others_alone(self):
        from bench.metrics import DEVICES, for_device
        a100, h100 = DEVICES["A100-80GB"], DEVICES["H100-SXM"]
        mla = MODEL_PRESETS["deepseek-v2-lite"]
        on_a100, on_h100 = for_device(mla, a100), for_device(mla, h100)
        self.assertTrue(on_a100.mla_v_padded)
        self.assertFalse(on_h100.mla_v_padded)
        # padding V from 128 to 192 is arithmetic on zeros: ~1.2× the FLOPs per pair
        self.assertAlmostEqual(on_a100.attn_flops_per_pair(1024)
                               / on_h100.attn_flops_per_pair(1024), 384 / 320, places=6)
        # non-MLA models are unaffected and returned unchanged
        for key in ("llama2-7b", "falcon-7b", "gemma3-4b", "qwen1.5-moe-a2.7b"):
            self.assertIs(for_device(MODEL_PRESETS[key], h100), MODEL_PRESETS[key])

    def test_decode_path_is_device_independent_for_mla(self):
        """Weight absorption never materialises V, so padding cannot apply."""
        from bench.metrics import DEVICES, for_device
        mla = MODEL_PRESETS["deepseek-v2-lite"]
        a = for_device(mla, DEVICES["A100-80GB"]).attn_flops_per_pair(1)
        h = for_device(mla, DEVICES["H100-SXM"]).attn_flops_per_pair(1)
        self.assertEqual(a, h)

    def test_device_peaks_stay_in_sync_with_the_profiles(self):
        from bench.metrics import DEVICES, DEVICE_PEAKS
        self.assertEqual(set(DEVICES), set(DEVICE_PEAKS))
        for k, d in DEVICES.items():
            self.assertEqual(DEVICE_PEAKS[k], (d.peak_bf16_flops, d.hbm_bytes_per_s))

    def test_detect_device_falls_back_without_cuda(self):
        from bench.metrics import DEVICES, detect_device
        self.assertEqual(detect_device("H100-SXM").name, "H100-SXM")
        self.assertIn(detect_device().name, {d.name for d in DEVICES.values()} | {"unknown"})
