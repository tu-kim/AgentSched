"""Analytic FLOPs / bytes model for MHA / GQA / MLA / MoE, model presets, NVML.

Attention core accounting (per layer, per request i; verified in README §AI):
  MHA/GQA  FLOPs = 4·H·d · n_i·(c_i + (n_i+1)/2)                    (causal QKᵀ + PV)
           bytes = 2·n_i·H·d·s          Q read + O write
                 + 2·(c_i+n_i)·H_kv·d·s_kv   K, V read (cached + new)
                 + 2·n_i·H_kv·d·s_kv        new K, V written to the paged cache
  MLA      FLOPs = 2·H·(d_qk + d_v) · n_i·(c_i + (n_i+1)/2)          attention core
                 + 2·r·H·(d_nope + d_v) · c_i    decompress cached latent (kv_b_proj GEMM,
                                                 vLLM prefill path with prefix hits)
           bytes = latent read c_i·(r + d_rope)·s_kv + latent write n_i·(r + d_rope)·s_kv
                 + decompressed K,V materialised: 2·2·(c_i+n_i)·H·(d_nope+d_v)·s  (write + read)
                 + Q read n_i·H·d_qk·s + O write n_i·H·d_v·s
For MHA, bf16, ignoring causal and the new-KV write this reduces exactly to
  AI_attn = Σ n_i(n_i+c_i) / Σ (2n_i+c_i)  FLOP/byte.

Linear (GEMM) side: FLOPs per token = 2 × ACTIVE params (MoE: top_k experts + shared);
weight traffic per iteration = all touched weights once (MoE: every expert that receives
≥1 token, ≈ all of them for Σn ≫ E/k).
"""
import math
import threading
import time
from dataclasses import dataclass


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

    # ------------------------------------------------------------ typing
    @property
    def attn_type(self):
        if self.kv_lora_rank:
            return "mla"
        return "gqa" if self.n_kv_heads < self.n_heads else "mha"

    @property
    def is_mha(self):
        return self.attn_type == "mha"

    @property
    def is_moe(self):
        return self.n_experts > 0

    @property
    def group_size(self):
        return self.n_heads // self.n_kv_heads

    @property
    def arch_label(self):
        return f"{self.attn_type.upper()}{'+MoE' if self.is_moe else ''}"

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

    def _mlp(self, inter):
        return (3 if self.mlp_gated else 2) * self.hidden * inter

    @property
    def n_moe_layers(self):
        return (self.n_layers - self.n_dense_layers) if self.is_moe else 0

    @property
    def mlp_active_params_moe_layer(self):
        return self.top_k * self._mlp(self.moe_inter) + self._mlp(self.shared_inter) + self.hidden * self.n_experts

    @property
    def mlp_total_params_moe_layer(self):
        return self.n_experts * self._mlp(self.moe_inter) + self._mlp(self.shared_inter) + self.hidden * self.n_experts

    @property
    def linear_params_active(self):
        """Params touched per token (sum over layers)."""
        dense_layers = self.n_layers - self.n_moe_layers
        return (self.n_layers * self.attn_proj_params + dense_layers * self._mlp(self.inter)
                + self.n_moe_layers * self.mlp_active_params_moe_layer)

    @property
    def linear_params_total(self):
        dense_layers = self.n_layers - self.n_moe_layers
        return (self.n_layers * self.attn_proj_params + dense_layers * self._mlp(self.inter)
                + self.n_moe_layers * self.mlp_total_params_moe_layer)

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
    def kv_bytes_per_token(self):
        if self.attn_type == "mla":
            return self.n_layers * (self.kv_lora_rank + self.qk_rope_head_dim) * self.kv_dtype_bytes
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    @property
    def linear_flops_per_token(self):
        return 2.0 * self.linear_params_active

    @property
    def attn_flops_per_token_per_ctx(self):
        """Attention-core FLOPs per (query token × context token), all layers."""
        return 2.0 * self.n_layers * self.n_heads * (self.d_qk + self.d_v)

    @property
    def mla_decompress_flops_per_ctx(self):
        """MLA only: kv_b_proj applied to each cached latent token, all layers."""
        if self.attn_type != "mla":
            return 0.0
        return 2.0 * self.n_layers * self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)

    def attn_crossover_ctx(self, n=1024):
        """Cached context c* at which a request's attention-side FLOPs
        (n·c·A + c·D) equal its linear FLOPs (n·P). Independent of n for
        MHA/GQA (D = 0); n-dependent for MLA because decompression is ∝ c only."""
        A, D, P = self.attn_flops_per_token_per_ctx, self.mla_decompress_flops_per_ctx, self.linear_flops_per_token
        return n * P / (n * A + D)

    def kv_capacity_tokens(self, gpu_mem_gib, util=0.90, tp=1, overhead_gib=2.0):
        """Rough KV capacity: (util·mem − weights/tp − overhead) per GPU × tp. Memory in GiB."""
        per_gpu = util * gpu_mem_gib * 2**30 - self.weight_bytes / tp - overhead_gib * 2**30
        return int(max(0.0, per_gpu) * tp / self.kv_bytes_per_token)

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_hf_config(cls, hf, name="", kv_dtype_bytes=2):
        g = lambda k, default=None: (hf.get(k, default) if isinstance(hf, dict)
                                     else getattr(hf, k, default))
        H = g("num_attention_heads")
        hid = g("hidden_size")
        model_type = g("model_type", "") or ""
        kv_lora = g("kv_lora_rank", 0) or 0
        head_dim = g("head_dim") or (hid // H)
        # MoE: DeepSeek / Qwen-MoE / Mixtral / OLMoE / Granite key variants
        n_exp = g("n_routed_experts") or g("num_experts") or g("num_local_experts") or 0
        top_k = g("num_experts_per_tok", 0) or 0
        moe_inter = g("moe_intermediate_size") or (g("intermediate_size") if n_exp else 0) or 0
        shared_inter = 0
        if g("shared_expert_intermediate_size"):
            shared_inter = g("shared_expert_intermediate_size")
        elif g("n_shared_experts"):
            shared_inter = g("n_shared_experts") * moe_inter
        n_dense = g("first_k_dense_replace", 0) or 0
        return cls(
            n_layers=g("num_hidden_layers"), hidden=hid, n_heads=H,
            n_kv_heads=g("num_key_value_heads") or H, head_dim=head_dim,
            inter=g("intermediate_size") or 0, vocab=g("vocab_size"),
            kv_dtype_bytes=kv_dtype_bytes,
            name=name or (g("_name_or_path") or ""),
            max_position=g("max_position_embeddings", 0) or 0,
            tied_embeddings=bool(g("tie_word_embeddings", False)),
            mlp_gated=model_type not in ("gpt_neox", "opt", "bloom", "phi", "gpt2", "gptj"),
            kv_lora_rank=kv_lora, q_lora_rank=g("q_lora_rank", 0) or 0,
            qk_rope_head_dim=g("qk_rope_head_dim", 0) or 0, qk_nope_head_dim=g("qk_nope_head_dim", 0) or 0,
            v_head_dim=g("v_head_dim", 0) or 0,
            n_experts=n_exp, top_k=top_k, moe_inter=moe_inter, shared_inter=shared_inter,
            n_dense_layers=n_dense,
        )

    @classmethod
    def from_vllm(cls, llm_engine, kv_dtype_bytes=2):
        mc = llm_engine.model_config
        return cls.from_hf_config(mc.hf_text_config, name=mc.model, kv_dtype_bytes=kv_dtype_bytes)


def _spec(L, hid, H, Hkv, d, inter, vocab, **kw):
    return ModelSpec(L, hid, H, Hkv, d, inter, vocab, **kw)


# Values verified against HF config.json (2026-09); all ungated, all in vLLM's registry.
MODEL_PRESETS = {
    # ---- MHA dense
    "qwen1.5-0.5b": _spec(24, 1024, 16, 16, 64, 2816, 151936, name="Qwen/Qwen1.5-0.5B", max_position=32768, tied_embeddings=True),
    "qwen1.5-1.8b": _spec(24, 2048, 16, 16, 128, 5504, 151936, name="Qwen/Qwen1.5-1.8B", max_position=32768),
    "qwen1.5-7b":   _spec(32, 4096, 32, 32, 128, 11008, 151936, name="Qwen/Qwen1.5-7B", max_position=32768),
    "llama2-7b":    _spec(32, 4096, 32, 32, 128, 11008, 32000, name="NousResearch/Llama-2-7b-hf", max_position=4096),
    # ---- GQA dense
    "qwen3-1.7b":   _spec(28, 2048, 16, 8, 128, 6144, 151936, name="Qwen/Qwen3-1.7B", max_position=40960, tied_embeddings=True),
    "qwen3-4b":     _spec(36, 2560, 32, 8, 128, 9728, 151936, name="Qwen/Qwen3-4B", max_position=40960, tied_embeddings=True),
    "qwen2.5-1.5b": _spec(28, 1536, 12, 2, 128, 8960, 151936, name="Qwen/Qwen2.5-1.5B", max_position=131072, tied_embeddings=True),
    "qwen2.5-3b":   _spec(36, 2048, 16, 2, 128, 11008, 151936, name="Qwen/Qwen2.5-3B", max_position=32768, tied_embeddings=True),
    "llama3.2-1b":  _spec(16, 2048, 32, 8, 64, 8192, 128256, name="unsloth/Llama-3.2-1B", max_position=131072, tied_embeddings=True),
    "llama3.2-3b":  _spec(28, 3072, 24, 8, 128, 8192, 128256, name="unsloth/Llama-3.2-3B", max_position=131072, tied_embeddings=True),
    "llama3.1-8b":  _spec(32, 4096, 32, 8, 128, 14336, 128256, name="meta-llama/Llama-3.1-8B", max_position=131072),
    # ---- MHA + MoE (attention block identical to qwen1.5-1.8b → isolates the FFN)
    "qwen1.5-moe-a2.7b": _spec(24, 2048, 16, 16, 128, 5632, 151936, name="Qwen/Qwen1.5-MoE-A2.7B", max_position=8192,
                               n_experts=60, top_k=4, moe_inter=1408, shared_inter=5632),
    "olmoe-1b-7b":  _spec(16, 2048, 16, 16, 128, 1024, 50304, name="allenai/OLMoE-1B-7B-0924", max_position=4096,
                          n_experts=64, top_k=8, moe_inter=1024),
    # ---- GQA + MoE
    "granite-3b-a800m": _spec(32, 1536, 24, 8, 64, 512, 49155, name="ibm-granite/granite-3.1-3b-a800m-instruct",
                              max_position=131072, tied_embeddings=True, n_experts=40, top_k=8, moe_inter=512),
    "qwen3-30b-a3b": _spec(48, 2048, 32, 4, 128, 6144, 151936, name="Qwen/Qwen3-30B-A3B", max_position=40960,
                           n_experts=128, top_k=8, moe_inter=768),          # 61 GB bf16: ~90K KV tokens on 80GB
    # ---- MLA + MoE (only registry-native MLA model that fits one 80GB GPU)
    "deepseek-v2-lite": _spec(27, 2048, 16, 16, 192, 10944, 102400, name="deepseek-ai/DeepSeek-V2-Lite", max_position=163840,
                              kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
                              n_experts=64, top_k=6, moe_inter=1408, shared_inter=2 * 1408, n_dense_layers=1),
    # ---- MLA dense: no public checkpoint runs natively in vLLM; synthetic DeepseekV2 config
    #      matched to qwen1.5-1.8b (bench/synthetic_configs/mla-dense-1.8b, --load-format dummy)
    "mla-dense-1.8b": _spec(24, 2048, 16, 16, 192, 5504, 151936, name="synthetic/mla-dense-1.8b", max_position=32768,
                            kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128),
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

    Returns FLOPs (linear / attention), three byte totals and two AIs:
      est_bytes_analytic = weight traffic + attention traffic   (README model)
      est_bytes_total    = analytic + est_act_bytes_approx      (adds a rough
                           activation-traffic heuristic; use analytic for roofline)
      est_ai_attn        = attention FLOPs / attention bytes    (the README formula)
    """
    L, H, Hkv, d = spec.n_layers, spec.n_heads, spec.n_kv_heads, spec.head_dim
    s, s_kv = spec.dtype_bytes, spec.kv_dtype_bytes
    hid = spec.hidden
    B = len(pairs)
    total_n = sum(n for n, _ in pairs)
    A = spec.attn_flops_per_token_per_ctx            # all layers
    D = spec.mla_decompress_flops_per_ctx            # all layers (0 unless MLA)

    flops_attn = kv_read = attn_bytes = 0.0
    for n, c in pairs:
        ctx = c + (n + 1) / 2.0
        flops_attn += n * ctx * A + c * D
        if spec.attn_type == "mla":
            r, rope = spec.kv_lora_rank, spec.qk_rope_head_dim
            lat_r = c * (r + rope) * s_kv * L
            lat_w = n * (r + rope) * s_kv * L
            # decompressed K (d_qk) and V (d_v, padded on FA2) materialised then read by FA
            kv_full = 2 * (c + n) * H * (spec.d_qk + spec.d_v) * s * L
            q_o = n * H * (spec.d_qk + spec.v_head_dim) * s * L
            kv_read += lat_r
            attn_bytes += lat_r + lat_w + kv_full + q_o
        else:
            q_o = 2.0 * n * H * d * s * L
            kv_r = 2.0 * (c + n) * Hkv * d * s_kv * L
            kv_w = 2.0 * n * Hkv * d * s_kv * L
            kv_read += kv_r
            attn_bytes += q_o + kv_r + kv_w

    flops_linear = spec.linear_flops_per_token * total_n + 2.0 * hid * spec.vocab * B
    # weight traffic: dense weights once; MoE experts that receive ≥1 token (expected
    # count under uniform routing); lm_head streamed; embedding rows gathered
    if spec.is_moe and spec.n_experts:
        frac = 1.0 - (1.0 - spec.top_k / spec.n_experts) ** total_n
        expert_params = spec.n_moe_layers * spec.n_experts * spec._mlp(spec.moe_inter)
        weight_traffic = (spec.linear_params_total - expert_params + frac * expert_params) * s
    else:
        weight_traffic = spec.linear_params_total * s
    weight_traffic += hid * spec.vocab * s + total_n * hid * s
    # rough activation HBM traffic outside attention (norms, rotary, GEMM in/out, SiLU)
    dense_layers = L - spec.n_moe_layers
    inter_active_sum = dense_layers * spec.inter + spec.n_moe_layers * (spec.top_k * spec.moe_inter + spec.shared_inter)
    act_bytes = ((10 * hid + (4 * H + 6 * Hkv) * d) * L + 6 * inter_active_sum) * s * total_n

    flops_total = flops_linear + flops_attn
    bytes_analytic = weight_traffic + attn_bytes
    return dict(
        est_flops_linear=flops_linear,
        est_flops_attn=flops_attn,
        est_flops_total=flops_total,
        est_attn_flop_frac=flops_attn / flops_total,
        est_weight_bytes=weight_traffic,
        est_kv_read_bytes=kv_read,
        est_attn_bytes=attn_bytes,
        est_act_bytes_approx=act_bytes,
        est_bytes_analytic=bytes_analytic,
        est_bytes_total=bytes_analytic + act_bytes,
        est_ai_attn=flops_attn / attn_bytes if attn_bytes else 0.0,
        est_ai_analytic=flops_total / bytes_analytic,
        est_arith_intensity=flops_total / (bytes_analytic + act_bytes),
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
