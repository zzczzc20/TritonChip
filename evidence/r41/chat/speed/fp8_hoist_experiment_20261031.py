"""Experimental, shape-gated block-FP8 quantisation hoisting on TNCA."""
import torch
import triton
import triton.language as tl


@triton.jit
def _precise_block_quant(X, Y, S, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    i = p * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.float32)
    scale = tl.div_rn(tl.max(tl.abs(x), axis=0), 448.0)
    q = tl.div_rn(x, tl.maximum(scale, 1e-12)).to(tl.float8e4nv)
    tl.store(Y + i, q)
    tl.store(S + p, scale)


def make_matmul(fallback, prequantized):
    """Return a shape-only selector; unsupported cases keep the given route."""
    def matmul(x, weight, scales, block_size=None, output_dtype=None,
               activation_scale=None):
        k = x.shape[-1]
        m = x.numel() // k
        n = weight.shape[0]
        eligible = (activation_scale is None and tuple(block_size or ()) == (128, 128)
                    and x.is_contiguous() and weight.ndim == 2
                    and weight.dtype == torch.float8_e4m3fn
                    and scales.dtype == torch.float32 and k % 128 == 0
                    and m >= 8 and n >= 4096 and k >= 2048)
        if not eligible:
            return fallback(x, weight, scales, block_size, output_dtype,
                            activation_scale=activation_scale)
        flat = x.reshape(m, k)
        q = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
        a_scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
        _precise_block_quant[(m * k // 128,)](flat, q, a_scales, 128)
        out = prequantized(q, weight, a_scales, scales, [128, 128],
                           output_dtype or x.dtype)
        return out.reshape(*x.shape[:-1], n)
    matmul._tnca_hoist_experiment = True
    return matmul
