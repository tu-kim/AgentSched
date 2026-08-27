"""Analytic FLOPs / bytes model + NVML sampling."""
import threading
import time
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Filled from the vLLM model config at runtime."""
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    inter: int          # FFN intermediate size
    vocab: int
    dtype_bytes: int = 2

    @classmethod
    def from_vllm(cls, llm_engine):
        hf = llm_engine.model_config.hf_text_config
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        return cls(
            n_layers=hf.num_hidden_layers,
            hidden=hf.hidden_size,
            n_heads=hf.num_attention_heads,
            n_kv_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=head_dim,
            inter=hf.intermediate_size,
            vocab=hf.vocab_size,
        )


def estimate_flops_bytes(spec: ModelSpec, pairs):
    """Per-iteration estimates for a batch of (n_i, c_i).

    FLOPs:
      linear (qkv/o/ffn, gated):  per-token 2 * P_linear params-equivalent
      attention score+value:      4 * n * (c + n) * n_heads * head_dim  per layer
                                  (causal over new tokens: n*(c + (n+1)/2), we use
                                  the causal-corrected form)
    DRAM bytes (lower bound):
      weights read once per iteration (batch-shared),
      KV cache read: (c+n) * n_kv_heads * head_dim * 2(K,V) * dtype per layer per seq,
      activations ~ ignored (small vs the above).
    """
    L, H, Hkv, d = spec.n_layers, spec.n_heads, spec.n_kv_heads, spec.head_dim
    hid, inter = spec.hidden, spec.inter
    # per-token linear FLOPs (2*params): qkv + out + gate/up/down
    p_lin = hid * (H * d) + 2 * hid * (Hkv * d) + (H * d) * hid + 3 * hid * inter
    total_n = sum(n for n, _ in pairs)
    flops_linear = 2.0 * p_lin * total_n * L + 2.0 * hid * spec.vocab * len(pairs)
    flops_attn = 0.0
    kv_bytes = 0.0
    for n, c in pairs:
        eff_ctx = c + (n + 1) / 2.0           # causal average context per new token
        flops_attn += 4.0 * n * eff_ctx * H * d * L
        kv_bytes += (c + n) * Hkv * d * 2 * spec.dtype_bytes * L
    weight_bytes = (p_lin * L + hid * spec.vocab) * spec.dtype_bytes
    total_flops = flops_linear + flops_attn
    total_bytes = weight_bytes + kv_bytes
    return dict(
        est_flops_linear=flops_linear,
        est_flops_attn=flops_attn,
        est_flops_total=total_flops,
        est_weight_bytes=weight_bytes,
        est_kv_bytes=kv_bytes,
        est_bytes_total=total_bytes,
        est_arith_intensity=total_flops / total_bytes,
    )


class NVMLSampler:
    """Background sampler for GPU util / power / mem-bw utilization."""

    def __init__(self, device_index=0, interval_s=0.005):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            u = self.nv.nvmlDeviceGetUtilizationRates(self.h)
            p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            self.samples.append((time.time(), u.gpu, u.memory, p))
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
        s = [x for x in self.samples
             if (t0 is None or x[0] >= t0) and (t1 is None or x[0] <= t1)]
        if not s:
            s = self.samples
        if not s:
            return dict(gpu_util=None, mem_util=None, power_w=None)
        avg = lambda i: sum(x[i] for x in s) / len(s)
        return dict(gpu_util=avg(1), mem_util=avg(2), power_w=avg(3))
