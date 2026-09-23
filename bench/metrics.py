"""Analytic FLOPs / bytes model across attention families, model presets, NVML.

A model is described as a stack of layer groups, because the families we compare
no longer share one attention shape:

  full    every query attends to the whole context      (MHA / GQA / MQA / MLA)
  local   attention is capped at `window` keys          (Gemma-3 local layers)
  linear  recurrent state, cost per token is constant   (Qwen3-Next DeltaNet)
  dsa     cheap indexer over the context + sparse       (DeepSeek-V3.2)
          attention over the selected top-k positions

Attention core per full layer, per request i (verified in README §AI):
  MHA/GQA FLOPs = 2·H·(d_qk + d_v) · A(n_i, c_i)   with A = Σ_j min(c+j, ctx cap)
          bytes = Q read + O write + K,V read (cached + new) + new K,V write
  MLA     additionally decompresses the cached latent through kv_b_proj, which
          costs 2·r·H·(d_nope+d_v) per *cached* token and is independent of n.

For MHA, bf16, ignoring causal masking and the new-KV write this reduces exactly
to  AI_attn = Σ n_i(n_i+c_i) / Σ (2n_i+c_i)  FLOP/byte.

Linear (GEMM) side: FLOPs per token = 2 × ACTIVE params (MoE: top_k experts plus
shared); the LM head is reported separately because it scales with the number of
sampled tokens, not with Σn, and its size differs sharply across vocabularies.
"""
import math
import threading
import time
from dataclasses import dataclass, field, replace


def attended_keys(n, c, window=0):
    """Σ_{j=1..n} min(c + j, window) — keys a causal batch of n new tokens reads
    on top of c cached ones. window=0 means no cap (full attention)."""
    if n <= 0:
        return 0.0
    if not window:
        return n * c + n * (n + 1) / 2.0
    k = max(0, min(n, window - c))          # tokens still inside the window
    return (k * c + k * (k + 1) / 2.0) + (n - k) * window


@dataclass
class ModelSpec:
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    inter: int                     # dense-MLP intermediate size
    vocab: int
    dtype_bytes: int = 2           # weights / activations
    kv_dtype_bytes: int = 2        # KV cache (fp8 → 1)
    name: str = ""
    max_position: int = 0
    tied_embeddings: bool = False
    mlp_gated: bool = True         # SwiGLU (3 matrices) vs GELU/ReLU (2)
    # ---- MLA (DeepSeek-V2/V3 style); attn_type == "mla" iff kv_lora_rank > 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    # vLLM's MLA prefill on SM80 (A100) runs FA2, which cannot mix head dims, so V is
    # zero-padded from v_head_dim to qk_head_dim; FA3 on SM90 does not pad.
    mla_v_padded: bool = True
    # ---- MoE; is_moe iff n_experts > 0
    n_experts: int = 0             # routed experts
    top_k: int = 0
    moe_inter: int = 0             # per-expert intermediate size
    shared_inter: int = 0          # total intermediate of always-active shared expert(s)
    n_dense_layers: int = 0        # leading dense layers (first_k_dense_replace)
    # ---- sliding window. window_pattern = every k-th layer is global (Gemma-3);
    #      0 with a window set means every layer is windowed (Mistral style).
    window: int = 0
    window_pattern: int = 0
    # ---- linear-attention hybrid: every full_attn_interval-th layer is softmax
    full_attn_interval: int = 0
    linear_k_heads: int = 0
    linear_v_heads: int = 0
    linear_k_head_dim: int = 0
    linear_v_head_dim: int = 0
    linear_conv_kernel: int = 0
    # ---- sparse attention (DeepSeek DSA): a lightweight MQA indexer scores every
    #      position and the top-k are handed to the sparse attention. indexer_heads
    #      is the QUERY head count (drives FLOPs); its KV cache is MQA, i.e. one
    #      head of indexer_head_dim, which is what makes the indexer cheap in bytes.
    dsa_topk: int = 0
    indexer_heads: int = 0
    indexer_head_dim: int = 0
    indexer_dtype_bytes: int = 1   # the indexer runs in FP8
    # ---- annotations that change nothing analytically but matter when comparing
    parallel_block: bool = False   # Falcon: attention and MLP run in parallel
    family: str = ""               # taxonomy label, see MODEL_PRESETS
    tier: str = "measurable"       # measurable | multi_gpu | analytic_only
    notes: str = ""

    # ------------------------------------------------------------ typing
    @property
    def attn_type(self):
        if self.kv_lora_rank:
            return "mla"
        if self.n_kv_heads == 1:
            return "mqa"
        return "gqa" if self.n_kv_heads < self.n_heads else "mha"

    @property
    def is_mha(self):
        return self.attn_type == "mha"

    @property
    def is_moe(self):
        return self.n_experts > 0

    @property
    def is_hybrid(self):
        """Layers do not all have the same attention shape."""
        return len(self.layer_groups()) > 1

    @property
    def group_size(self):
        return self.n_heads // self.n_kv_heads

    @property
    def arch_label(self):
        base = self.attn_type.upper()
        if self.dsa_topk:
            base += "+DSA"
        if self.full_attn_interval:
            base = "Linear/" + base
        elif self.window:
            base += "+SWA"
        return base + ("+MoE" if self.is_moe else "")

    # ------------------------------------------------------------ layer stack
    def layer_groups(self):
        """[(count, kind, param)] summing to n_layers; param is the window (local)
        or the top-k budget (dsa)."""
        L = self.n_layers
        if self.full_attn_interval:
            n_full = L // self.full_attn_interval
            return [(n_full, "full", 0), (L - n_full, "linear", 0)]
        if self.dsa_topk:
            return [(L, "dsa", self.dsa_topk)]
        if self.window and self.window_pattern:
            n_global = L // self.window_pattern
            return [(n_global, "full", 0), (L - n_global, "local", self.window)]
        if self.window:
            return [(L, "local", self.window)]
        return [(L, "full", 0)]

    @property
    def n_growing_kv_layers(self):
        """Layers whose KV footprint keeps growing with context."""
        return sum(cnt for cnt, kind, _ in self.layer_groups() if kind in ("full", "dsa"))

    # ------------------------------------------------------------ per-layer params
    @property
    def d_qk(self):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) if self.attn_type == "mla" else self.head_dim

    @property
    def d_v(self):
        """V head dim as the attention kernel sees it (MLA: padded on FA2)."""
        if self.attn_type == "mla":
            return self.d_qk if self.mla_v_padded else self.v_head_dim
        return self.head_dim

    @property
    def attn_proj_params(self):
        """QKVO projection weights of one softmax-attention layer."""
        hid, H = self.hidden, self.n_heads
        if self.attn_type == "mla":
            r, rope = self.kv_lora_rank, self.qk_rope_head_dim
            q = (hid * self.q_lora_rank + self.q_lora_rank * H * self.d_qk) if self.q_lora_rank \
                else hid * H * self.d_qk
            kv_a = hid * (r + rope)
            kv_b = r * H * (self.qk_nope_head_dim + self.v_head_dim)
            o = H * self.v_head_dim * hid
            return q + kv_a + kv_b + o
        d, Hkv = self.head_dim, self.n_kv_heads
        return hid * H * d + 2 * hid * Hkv * d + H * d * hid

    @property
    def linear_attn_proj_params(self):
        """Q/K/V/O projections of one Gated-DeltaNet layer (gates and conv ignored)."""
        if not self.full_attn_interval:
            return 0
        hid = self.hidden
        qk = self.linear_k_heads * self.linear_k_head_dim
        v = self.linear_v_heads * self.linear_v_head_dim
        return hid * qk * 2 + hid * v + v * hid

    @property
    def linear_state_bytes(self):
        """Recurrent state one DeltaNet layer keeps per request — constant in context."""
        if not self.full_attn_interval:
            return 0
        state = self.linear_v_heads * self.linear_k_head_dim * self.linear_v_head_dim
        conv = self.linear_conv_kernel * (2 * self.linear_k_heads * self.linear_k_head_dim
                                          + self.linear_v_heads * self.linear_v_head_dim)
        return (state + conv) * self.dtype_bytes

    @property
    def linear_flops_per_token_per_layer(self):
        """Delta-rule state update plus readout, ~4 MACs per state element."""
        if not self.full_attn_interval:
            return 0.0
        return 4.0 * self.linear_v_heads * self.linear_k_head_dim * self.linear_v_head_dim

    @property
    def indexer_flops_per_pair(self):
        """DSA indexer score per (query, key) pair, one layer."""
        return 2.0 * self.indexer_heads * self.indexer_head_dim if self.dsa_topk else 0.0

    def _mlp(self, inter):
        return (3 if self.mlp_gated else 2) * self.hidden * inter

    @property
    def n_moe_layers(self):
        return (self.n_layers - self.n_dense_layers) if self.is_moe else 0

    @property
    def mlp_active_params_moe_layer(self):
        return self.top_k * self._mlp(self.moe_inter) + self._mlp(self.shared_inter) \
            + self.hidden * self.n_experts

    @property
    def mlp_total_params_moe_layer(self):
        return self.n_experts * self._mlp(self.moe_inter) + self._mlp(self.shared_inter) \
            + self.hidden * self.n_experts

    def _attn_params_by_kind(self):
        full = sum(cnt for cnt, kind, _ in self.layer_groups() if kind != "linear")
        lin = sum(cnt for cnt, kind, _ in self.layer_groups() if kind == "linear")
        return full * self.attn_proj_params + lin * self.linear_attn_proj_params

    @property
    def linear_params_active(self):
        """Params touched per token, excluding the LM head (sum over layers)."""
        dense_layers = self.n_layers - self.n_moe_layers
        return (self._attn_params_by_kind() + dense_layers * self._mlp(self.inter)
                + self.n_moe_layers * self.mlp_active_params_moe_layer)

    @property
    def linear_params_total(self):
        dense_layers = self.n_layers - self.n_moe_layers
        return (self._attn_params_by_kind() + dense_layers * self._mlp(self.inter)
                + self.n_moe_layers * self.mlp_total_params_moe_layer)

    @property
    def lm_head_params(self):
        return self.hidden * self.vocab

    @property
    def embedding_params(self):
        return self.hidden * self.vocab * (1 if self.tied_embeddings else 2)

    @property
    def total_params(self):
        return self.linear_params_total + self.embedding_params

    @property
    def weight_bytes(self):
        """Resident weight memory (capacity accounting)."""
        return self.total_params * self.dtype_bytes

    # ------------------------------------------------------------ per-token rates
    @property
    def _kv_bytes_per_layer_per_token(self):
        if self.attn_type == "mla":
            return (self.kv_lora_rank + self.qk_rope_head_dim) * self.kv_dtype_bytes
        return 2 * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    @property
    def kv_bytes_per_token(self):
        """Marginal KV bytes per extra context token. Local and linear layers stop
        growing, so for a hybrid model this is far below n_layers × per-layer."""
        return self.n_growing_kv_layers * self._kv_bytes_per_layer_per_token

    def kv_bytes_for(self, ctx):
        """Total KV footprint of one request holding `ctx` tokens of context."""
        total = 0.0
        for cnt, kind, param in self.layer_groups():
            if kind == "linear":
                total += cnt * self.linear_state_bytes
            elif kind == "local":
                total += cnt * min(ctx, param) * self._kv_bytes_per_layer_per_token
            else:
                total += cnt * ctx * self._kv_bytes_per_layer_per_token
        return total

    @property
    def linear_flops_per_token(self):
        """Body GEMM FLOPs per token (LM head excluded — it scales with B)."""
        return 2.0 * self.linear_params_active

    @property
    def attn_flops_per_token_per_ctx(self):
        """Attention-core FLOPs per (query token × context token) summed over the
        layers that still scale with context. Meaningless for linear/local-heavy
        models on its own — use attn_flops()."""
        per_layer = 2.0 * self.n_heads * (self.d_qk + self.d_v)
        return per_layer * self.n_growing_kv_layers

    @property
    def mla_decompress_flops_per_ctx(self):
        """MLA only: kv_b_proj applied to each cached latent token, all layers."""
        if self.attn_type != "mla":
            return 0.0
        return 2.0 * self.n_layers * self.kv_lora_rank * self.n_heads * \
            (self.qk_nope_head_dim + self.v_head_dim)

    def is_decode(self, n):
        """vLLM routes query_len <= 1 to the decode path (MLA's
        reorder_batch_threshold is 1), which for MLA is weight-absorbed."""
        return n <= 1

    def attn_flops_per_pair(self, n):
        """Attention-core FLOPs per (query, key) pair for one layer.

        MLA decode absorbs W_UK/W_UV into the projections, so the score is taken
        against the 576-wide latent directly: no per-head K/V is materialised, the
        KV read collapses to one head, but per-head score FLOPs grow ~2.8×.
        """
        if self.attn_type == "mla" and self.is_decode(n):
            latent = self.kv_lora_rank + self.qk_rope_head_dim
            return 2.0 * self.n_heads * (latent + self.kv_lora_rank)
        return 2.0 * self.n_heads * (self.d_qk + self.d_v)

    def attn_flops(self, n, c):
        """Attention-side FLOPs for one request, summed over the layer stack."""
        per_pair = self.attn_flops_per_pair(n)
        total = 0.0
        for cnt, kind, param in self.layer_groups():
            if kind == "linear":
                total += cnt * self.linear_flops_per_token_per_layer * n
            elif kind == "dsa":
                total += cnt * (self.indexer_flops_per_pair * attended_keys(n, c)
                                + per_pair * attended_keys(n, c, param))
            else:
                total += cnt * per_pair * attended_keys(n, c, param if kind == "local" else 0)
        if self.attn_type == "mla" and not self.is_decode(n):
            # prefill re-materialises the cached latent through kv_b_proj, once per
            # cached token attended to; the absorbed decode path never does this
            seen = min(c, self.dsa_topk) if self.dsa_topk else c
            total += self.mla_decompress_flops_per_ctx * seen
        return total

    def attn_crossover_ctx(self, n=1024, c_max=1 << 22):
        """Cached context c* where a request's attention-side FLOPs reach its body
        GEMM FLOPs. Returns inf when the layer stack caps attention (windowed or
        linear layers) so that it never catches up."""
        target = self.linear_flops_per_token * n
        if self.attn_flops(n, c_max) < target:
            return math.inf
        lo, hi = 0, c_max
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.attn_flops(n, mid) < target:
                lo = mid
            else:
                hi = mid
        return float(hi)

    def kv_capacity_tokens(self, gpu_mem_gib, util=0.90, tp=1, overhead_gib=2.0):
        """Rough KV capacity in *context tokens of a full-attention layer stack*.
        For hybrid models use kv_capacity_requests(), which accounts for the
        windowed and recurrent layers."""
        per_gpu = util * gpu_mem_gib * 2**30 - self.weight_bytes / tp - overhead_gib * 2**30
        rate = self.kv_bytes_per_token
        return int(max(0.0, per_gpu) * tp / rate) if rate else 0

    def kv_capacity_requests(self, ctx, gpu_mem_gib, util=0.90, tp=1, overhead_gib=2.0):
        """How many requests holding `ctx` context tokens fit alongside the weights."""
        per_gpu = util * gpu_mem_gib * 2**30 - self.weight_bytes / tp - overhead_gib * 2**30
        each = self.kv_bytes_for(ctx)
        return int(max(0.0, per_gpu) * tp / each) if each else 0

    def fits_on(self, gpu_mem_gib, tp=1, util=0.90):
        return self.weight_bytes / tp < util * gpu_mem_gib * 2**30

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_hf_config(cls, hf, name="", kv_dtype_bytes=2, **overrides):
        g = lambda k, default=None: (hf.get(k, default) if isinstance(hf, dict)
                                     else getattr(hf, k, default))
        H = g("num_attention_heads") or g("n_head")
        hid = g("hidden_size")
        model_type = g("model_type", "") or ""
        kv_lora = g("kv_lora_rank", 0) or 0
        head_dim = g("head_dim") or (hid // H)
        gated = model_type not in ("gpt_neox", "opt", "bloom", "phi", "gpt2", "gptj", "falcon")
        # KV heads: explicit key, MQA flag, or MHA by default
        n_kv = g("num_key_value_heads") or g("n_head_kv") or g("num_kv_heads")
        if not n_kv:
            n_kv = 1 if g("multi_query", False) else H
        # FFN width: several families omit intermediate_size
        inter = g("intermediate_size") or g("ffn_dim") or g("ffn_hidden_size") or 0
        if not inter:
            inter = 4 * hid                 # Falcon / GPT-NeoX convention
        # MoE key variants across families
        n_exp = g("n_routed_experts") or g("num_experts") or g("num_local_experts") or 0
        top_k = g("num_experts_per_tok", 0) or 0
        moe_inter = g("moe_intermediate_size") or (inter if n_exp else 0) or 0
        shared_inter = 0
        if g("shared_expert_intermediate_size"):
            shared_inter = g("shared_expert_intermediate_size")
        elif g("n_shared_experts"):
            shared_inter = g("n_shared_experts") * moe_inter
        spec = cls(
            n_layers=g("num_hidden_layers") or g("n_layer"), hidden=hid, n_heads=H,
            n_kv_heads=n_kv, head_dim=head_dim, inter=inter, vocab=g("vocab_size"),
            kv_dtype_bytes=kv_dtype_bytes,
            name=name or (g("_name_or_path") or ""),
            max_position=g("max_position_embeddings", 0) or 0,
            tied_embeddings=bool(g("tie_word_embeddings", False)),
            mlp_gated=gated,
            kv_lora_rank=kv_lora, q_lora_rank=g("q_lora_rank", 0) or 0,
            qk_rope_head_dim=g("qk_rope_head_dim", 0) or 0,
            qk_nope_head_dim=g("qk_nope_head_dim", 0) or 0,
            v_head_dim=g("v_head_dim", 0) or 0,
            n_experts=n_exp, top_k=top_k, moe_inter=moe_inter, shared_inter=shared_inter,
            n_dense_layers=g("first_k_dense_replace", 0) or 0,
            window=g("sliding_window", 0) or 0,
            window_pattern=g("sliding_window_pattern", 0) or 0,
            full_attn_interval=g("full_attention_interval", 0) or 0,
            linear_k_heads=g("linear_num_key_heads", 0) or 0,
            linear_v_heads=g("linear_num_value_heads", 0) or 0,
            linear_k_head_dim=g("linear_key_head_dim", 0) or 0,
            linear_v_head_dim=g("linear_value_head_dim", 0) or 0,
            linear_conv_kernel=g("linear_conv_kernel_dim", 0) or 0,
            parallel_block=bool(g("parallel_attn", False)),
        )
        return replace(spec, **overrides) if overrides else spec

    @classmethod
    def from_vllm(cls, llm_engine, kv_dtype_bytes=2):
        mc = llm_engine.model_config
        return cls.from_hf_config(mc.hf_text_config, name=mc.model, kv_dtype_bytes=kv_dtype_bytes)


def _spec(L, hid, H, Hkv, d, inter, vocab, **kw):
    return ModelSpec(L, hid, H, Hkv, d, inter, vocab, **kw)


# ---------------------------------------------------------------------------
# Presets. `family` follows the architecture taxonomy; `tier` says whether the
# model can be measured on one 80GB GPU (measurable), needs several (multi_gpu),
# or is only modelled analytically (analytic_only).
# Values are from the HF config.json of each repo unless marked synthetic.
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    # ---- 1. MHA -----------------------------------------------------------
    "llama2-7b": _spec(32, 4096, 32, 32, 128, 11008, 32000, name="NousResearch/Llama-2-7b-hf",
                       max_position=4096, family="1-MHA"),
    "qwen1.5-1.8b": _spec(24, 2048, 16, 16, 128, 5504, 151936, name="Qwen/Qwen1.5-1.8B",
                          max_position=32768, family="1-MHA"),
    "qwen1.5-0.5b": _spec(24, 1024, 16, 16, 64, 2816, 151936, name="Qwen/Qwen1.5-0.5B",
                          max_position=32768, tied_embeddings=True, family="1-MHA"),
    # ---- 2. MQA -----------------------------------------------------------
    "falcon-7b": _spec(32, 4544, 71, 1, 64, 4 * 4544, 65024, name="tiiuae/falcon-7b",
                       max_position=2048, mlp_gated=False, parallel_block=True, family="2-MQA",
                       notes="attention and MLP run in parallel; non-gated MLP (4·hidden)"),
    # ---- 3. GQA -----------------------------------------------------------
    "llama3-8b": _spec(32, 4096, 32, 8, 128, 14336, 128256, name="NousResearch/Meta-Llama-3-8B",
                       max_position=8192, family="3-GQA",
                       notes="vs Llama-2-7B also changes vocab 32K→128K and FFN 11008→14336; "
                             "compare with the LM head split out"),
    "qwen3-1.7b": _spec(28, 2048, 16, 8, 128, 6144, 151936, name="Qwen/Qwen3-1.7B",
                        max_position=40960, tied_embeddings=True, family="3-GQA"),
    "qwen2.5-3b": _spec(36, 2048, 16, 2, 128, 11008, 151936, name="Qwen/Qwen2.5-3B",
                        max_position=32768, tied_embeddings=True, family="3-GQA"),
    "mistral-7b": _spec(32, 4096, 32, 8, 128, 14336, 32000, name="mistralai/Mistral-7B-v0.1",
                        max_position=32768, family="3-GQA",
                        notes="config declares sliding_window=4096; left off here so the pair "
                              "with Mixtral isolates the FFN change only"),
    # ---- 4. local/global sliding window -----------------------------------
    "gemma3-4b": _spec(34, 2560, 8, 4, 256, 10240, 262208, name="unsloth/gemma-3-4b-it",
                       max_position=131072, window=1024, window_pattern=6, family="4-SWA",
                       notes="5 global layers (every 6th), 29 local at window 1024; "
                             "head_dim 256 is not hidden/heads"),
    # ---- 5. MoE FFN --------------------------------------------------------
    "mixtral-8x7b": _spec(32, 4096, 32, 8, 128, 14336, 32000, name="mistralai/Mixtral-8x7B-v0.1",
                          max_position=32768, n_experts=8, top_k=2, moe_inter=14336,
                          family="5-MoE", tier="multi_gpu",
                          notes="attention identical to Mistral-7B — the FFN-only pair; "
                                "93GB bf16 needs tp≥2"),
    "qwen1.5-moe-a2.7b": _spec(24, 2048, 16, 16, 128, 5632, 151936, name="Qwen/Qwen1.5-MoE-A2.7B",
                               max_position=8192, n_experts=60, top_k=4, moe_inter=1408,
                               shared_inter=5632, family="5-MoE",
                               notes="attention block identical to Qwen1.5-1.8B — the "
                                     "single-GPU stand-in for the FFN-only pair"),
    "olmoe-1b-7b": _spec(16, 2048, 16, 16, 128, 1024, 50304, name="allenai/OLMoE-1B-7B-0924",
                         max_position=4096, n_experts=64, top_k=8, moe_inter=1024, family="5-MoE"),
    # ---- 6. MLA ------------------------------------------------------------
    #      head_dim is unused under MLA (d_qk/d_v come from the MLA dims); it is set to
    #      hidden/heads so the preset equals what from_hf_config() derives.
    "deepseek-v2-lite": _spec(27, 2048, 16, 16, 128, 10944, 102400,
                              name="deepseek-ai/DeepSeek-V2-Lite", max_position=163840,
                              kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128,
                              v_head_dim=128, n_experts=64, top_k=6, moe_inter=1408,
                              shared_inter=2 * 1408, n_dense_layers=1, family="6-MLA"),
    "deepseek-v3": _spec(61, 7168, 128, 128, 128, 18432, 129280, name="deepseek-ai/DeepSeek-V3",
                         max_position=163840, kv_lora_rank=512, q_lora_rank=1536,
                         qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
                         n_experts=256, top_k=8, moe_inter=2048, shared_inter=2048,
                         n_dense_layers=3, family="6-MLA", tier="analytic_only",
                         notes="671B total / 37B active — reference point only"),
    # ---- 7. sparse attention (DSA) ----------------------------------------
    "deepseek-v3.2": _spec(61, 7168, 128, 128, 128, 18432, 129280,
                           name="deepseek-ai/DeepSeek-V3.2-Exp", max_position=163840,
                           kv_lora_rank=512, q_lora_rank=1536, qk_rope_head_dim=64,
                           qk_nope_head_dim=128, v_head_dim=128, n_experts=256, top_k=8,
                           moe_inter=2048, shared_inter=2048, n_dense_layers=3,
                           dsa_topk=2048, indexer_heads=64, indexer_head_dim=128,
                           family="7-DSA", tier="analytic_only",
                           notes="V3.1↔V3.2 is the DSA-only pair; indexer dims are the "
                                 "published lightweight-MQA shape and need confirming"),
    # ---- 8. linear-attention hybrid ---------------------------------------
    "qwen3-next-80b": _spec(48, 2048, 16, 2, 256, 5120, 151936,
                            name="Qwen/Qwen3-Next-80B-A3B-Instruct", max_position=262144,
                            n_experts=512, top_k=10, moe_inter=512, shared_inter=512,
                            full_attn_interval=4, linear_k_heads=16, linear_v_heads=32,
                            linear_k_head_dim=128, linear_v_head_dim=128, linear_conv_kernel=4,
                            family="8-Linear", tier="multi_gpu",
                            notes="12 softmax layers + 36 Gated-DeltaNet (3:1); 160GB bf16"),
    # ---- synthetic: isolate one structural change at a time ---------------
    #      bench/synthetic_configs/*, run with --load-format dummy
    "kv-mha-1.8b": _spec(24, 2048, 16, 16, 128, 5504, 151936, name="synthetic/kv-mha-1.8b",
                         max_position=32768, family="A-KVshare"),
    "kv-gqa4-1.8b": _spec(24, 2048, 16, 4, 128, 5504, 151936, name="synthetic/kv-gqa4-1.8b",
                          max_position=32768, family="A-KVshare"),
    "kv-mqa-1.8b": _spec(24, 2048, 16, 1, 128, 5504, 151936, name="synthetic/kv-mqa-1.8b",
                         max_position=32768, family="A-KVshare"),
    "mla-dense-1.8b": _spec(24, 2048, 16, 16, 128, 5504, 151936, name="synthetic/mla-dense-1.8b",
                            max_position=32768, kv_lora_rank=512, qk_rope_head_dim=64,
                            qk_nope_head_dim=128, v_head_dim=128, family="A-KVshare"),
    "swa-1.8b": _spec(24, 2048, 16, 16, 128, 5504, 151936, name="synthetic/swa-1.8b",
                      max_position=32768, window=1024, window_pattern=6, family="A-KVshare",
                      notes="Gemma-3's 5:1 local/global pattern grafted onto the MHA baseline"),
    # ---- other GQA references ---------------------------------------------
    "llama3.1-8b": _spec(32, 4096, 32, 8, 128, 14336, 128256, name="meta-llama/Llama-3.1-8B",
                         max_position=131072, family="3-GQA"),
    "llama3.2-1b": _spec(16, 2048, 32, 8, 64, 8192, 128256, name="unsloth/Llama-3.2-1B",
                         max_position=131072, tied_embeddings=True, family="3-GQA"),
    "granite-3b-a800m": _spec(32, 1536, 24, 8, 64, 512, 49155,
                              name="ibm-granite/granite-3.1-3b-a800m-instruct",
                              max_position=131072, tied_embeddings=True, n_experts=40, top_k=8,
                              moe_inter=512, family="5-MoE"),
}

# Pairs that differ in exactly one structural dimension.
CONTROLLED_PAIRS = {
    "ffn-dense-vs-moe": ("qwen1.5-1.8b", "qwen1.5-moe-a2.7b",
                         "identical attention block (L24/H16/d128/hidden 2048); FFN dense→MoE"),
    "ffn-dense-vs-moe-7b": ("mistral-7b", "mixtral-8x7b",
                            "identical attention (GQA 32/8); FFN dense→8×top-2 MoE. tp≥2"),
    "kv-mha-vs-gqa": ("kv-mha-1.8b", "kv-gqa4-1.8b", "only num_key_value_heads 16→4"),
    "kv-gqa-vs-mqa": ("kv-gqa4-1.8b", "kv-mqa-1.8b", "only num_key_value_heads 4→1"),
    "kv-mha-vs-mla": ("kv-mha-1.8b", "mla-dense-1.8b", "full KV cache → 576-wide latent"),
    "attn-full-vs-swa": ("kv-mha-1.8b", "swa-1.8b", "only the 5:1 local/global window pattern"),
    "real-mha-vs-gqa": ("llama2-7b", "llama3-8b",
                        "same family but vocab and FFN also change — split the LM head out"),
    "dsa-off-vs-on": ("deepseek-v3", "deepseek-v3.2", "only the sparse-attention indexer"),
}

# (peak dense bf16 FLOP/s, HBM byte/s) for roofline reference lines
DEVICE_PEAKS = {
    "A100-80GB": (312e12, 2.039e12),       # SXM
    "A100-80GB-PCIe": (312e12, 1.935e12),
    "H100-SXM": (989e12, 3.35e12),
    "H100-PCIe": (756e12, 2.0e12),
    "H200": (989e12, 4.8e12),
}


def estimate_flops_bytes(spec: ModelSpec, pairs):
    """Per-iteration estimates for a batch of (n_i, c_i).

    The LM head is reported separately (est_flops_lm_head): it scales with the
    number of sampled requests and its size varies 4× across the vocabularies in
    the preset table, so folding it into the body GEMM makes cross-family
    comparisons misleading.

    Byte totals:
      est_bytes_analytic = weight traffic + attention traffic   (README model)
      est_bytes_total    = analytic + est_act_bytes_approx      (adds a rough
                           activation-traffic heuristic; use analytic for roofline)
    """
    H, s, s_kv = spec.n_heads, spec.dtype_bytes, spec.kv_dtype_bytes
    hid = spec.hidden
    B = len(pairs)
    total_n = sum(n for n, _ in pairs)
    groups = spec.layer_groups()

    flops_attn = kv_read = attn_bytes = 0.0
    for n, c in pairs:
        flops_attn += spec.attn_flops(n, c)
        for cnt, kind, param in groups:
            if kind == "linear":
                attn_bytes += cnt * 2 * spec.linear_state_bytes       # read + write the state
                continue
            held = min(c + n, param) if kind == "local" else c + n
            if spec.attn_type == "mla":
                r, rope = spec.kv_lora_rank, spec.qk_rope_head_dim
                seen = min(held, spec.dsa_topk) if spec.dsa_topk else held
                lat_r = seen * (r + rope) * s_kv
                lat_w = n * (r + rope) * s_kv
                q_o = n * H * (spec.d_qk + spec.v_head_dim) * s
                kv_read += cnt * lat_r
                if spec.is_decode(n):
                    # absorbed decode: the latent IS the KV — nothing is materialised
                    attn_bytes += cnt * (lat_r + lat_w + q_o)
                else:
                    # prefill materialises per-head K (d_qk) and V (d_v, padded on FA2)
                    kv_full = 2 * seen * H * (spec.d_qk + spec.d_v) * s
                    attn_bytes += cnt * (lat_r + lat_w + kv_full + q_o)
            else:
                q_o = 2.0 * n * H * spec.head_dim * s
                kv_r = 2.0 * held * spec.n_kv_heads * spec.head_dim * s_kv
                kv_w = 2.0 * n * spec.n_kv_heads * spec.head_dim * s_kv
                kv_read += cnt * kv_r
                attn_bytes += cnt * (q_o + kv_r + kv_w)
            if spec.dsa_topk:
                # the indexer's own cache is MQA: one head, FP8 — O(context) bytes
                # but a small constant, while its FLOPs still grow with the context
                attn_bytes += cnt * (c + n) * spec.indexer_head_dim * spec.indexer_dtype_bytes

    flops_body = spec.linear_flops_per_token * total_n
    flops_lm_head = 2.0 * spec.lm_head_params * B
    # weight traffic: dense weights once; MoE experts that receive ≥1 token (expected
    # count under uniform routing); lm_head streamed; embedding rows gathered
    if spec.is_moe and spec.n_experts:
        frac = 1.0 - (1.0 - spec.top_k / spec.n_experts) ** total_n
        expert_params = spec.n_moe_layers * spec.n_experts * spec._mlp(spec.moe_inter)
        weight_traffic = (spec.linear_params_total - expert_params + frac * expert_params) * s
    else:
        weight_traffic = spec.linear_params_total * s
    weight_traffic += spec.lm_head_params * s + total_n * hid * s
    # rough activation HBM traffic outside attention (norms, rotary, GEMM in/out, SiLU)
    dense_layers = spec.n_layers - spec.n_moe_layers
    inter_active_sum = dense_layers * spec.inter + spec.n_moe_layers * (
        spec.top_k * spec.moe_inter + spec.shared_inter)
    act_bytes = ((10 * hid + (4 * H + 6 * spec.n_kv_heads) * spec.head_dim) * spec.n_layers
                 + 6 * inter_active_sum) * s * total_n

    flops_linear = flops_body + flops_lm_head
    flops_total = flops_linear + flops_attn
    bytes_analytic = weight_traffic + attn_bytes
    ratio = lambda a, b: a / b if b else 0.0          # an empty batch is 0 work, not a crash
    return dict(
        est_flops_body=flops_body,
        est_flops_lm_head=flops_lm_head,
        est_flops_linear=flops_linear,
        est_flops_attn=flops_attn,
        est_flops_total=flops_total,
        est_attn_flop_frac=ratio(flops_attn, flops_total),
        est_lm_head_flop_frac=ratio(flops_lm_head, flops_total),
        est_weight_bytes=weight_traffic,
        est_kv_read_bytes=kv_read,
        est_attn_bytes=attn_bytes,
        est_act_bytes_approx=act_bytes,
        est_bytes_analytic=bytes_analytic,
        est_bytes_total=bytes_analytic + act_bytes,
        est_ai_attn=ratio(flops_attn, attn_bytes),
        est_ai_analytic=ratio(flops_total, bytes_analytic),
        est_arith_intensity=ratio(flops_total, bytes_analytic + act_bytes),
    )


class NVMLSampler:
    """Background sampler for GPU power / utilisation, reusable across `with` blocks.

    `cuda_index` is the CUDA-visible device index (what the engine uses); the NVML
    handle is resolved through the device UUID so CUDA_VISIBLE_DEVICES is honoured.
    Timestamps use time.perf_counter() so callers can window with the same clock.

    Note: nvmlDeviceGetUtilizationRates is the driver's trailing-window average
    (~100 ms–1 s), so gpu_util/mem_util are NOT per-iteration quantities; power is
    near-instantaneous. Use --kernel-profile kernel times for per-iteration busy time.
    """

    def __init__(self, cuda_index=0, interval_s=0.005):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = None
        try:
            import torch
            uuid = str(torch.cuda.get_device_properties(cuda_index).uuid)
            self.h = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
        except Exception:
            self.h = pynvml.nvmlDeviceGetHandleByIndex(cuda_index)
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            u = self.nv.nvmlDeviceGetUtilizationRates(self.h)
            p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            self.samples.append((time.perf_counter(), u.gpu, u.memory, p))
            time.sleep(self.interval)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self, t0=None, t1=None):
        """Averages over samples in [t0, t1] (perf_counter seconds); None if no sample fell inside."""
        s = [x for x in self.samples
             if (t0 is None or x[0] >= t0) and (t1 is None or x[0] <= t1)]
        if not s:
            return dict(gpu_util=None, mem_util=None, power_w=None)
        avg = lambda i: sum(x[i] for x in s) / len(s)
        return dict(gpu_util=avg(1), mem_util=avg(2), power_w=avg(3))
