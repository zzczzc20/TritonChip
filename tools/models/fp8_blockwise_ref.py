"""Independent reference of fine-grained (block-wise) W8A8 FP8 linear math, as computed by the Hub
Triton kernel `kernels-community/finegrained-fp8` (v4, kernel commit fe2bb66) that transformers'
`FP8Linear` calls (integrations/finegrained_fp8.py: FP8Linear.forward -> fp8_linear ->
finegrained_fp8_linear -> kernel.matmul_2d(input, weight, weight_scale_inv, block_size, input.dtype,
activation_scale=...)).

Nothing here imports or calls the Hub kernel (only `validate --cuda` does, to check this file against it).

The format is keyed on the tensors alone (the kernel's own dispatch, matmul_2d):
  * weight  B  [N, K] float8_e4m3fn, scales Bs [ceil(N/bn), ceil(K/bk)] (fp32 / bf16 / ue8m0 exponent bytes)
  * block_size [bn, bk]           -> block-wise; dynamic activations unless a static scale is given
  * block_size None or [N, K]     -> tensor-wide weight scale, per-row dynamic activations
  * MX / FP4 (group-32 ue8m0)     -> not implemented here (NotImplementedError)

Block-wise dynamic path (w8a8_block_dynamic_fp8_matmul_kernel + fp8_act_quant_inline), per output tile and
for every K tile k of width bk:
    a_raw = A[:, k-tile].float()                      (zero beyond K)
    a_s   = max(|a_raw|, axis=1) / 448.0              fp32, one scale per row and K tile
    a     = (a_raw / max(a_s, 1e-12)).to(e4m3)        round to nearest even
    acc  += tl.dot(a, b_k) * a_s[:, None] * b_s[None, :]     fp32; b_s = Bs[n // bn, k]
    C     = acc.to(output_dtype)                      output_dtype = A.dtype (transformers passes input.dtype)
Bias is added afterwards in the output dtype by transformers (`output.add_(bias)`).

Choices that the source leaves to the hardware, made explicit here (`dot`, `fma`):
  * dot="exact": each K-tile dot of e4m3 x e4m3 values is computed exactly (float64: every e4m3 value is a
    multiple of 2^-9, so a 128-term sum is a multiple of 2^-18 below 2^25 -- exact in 53 bits whatever the
    order) and rounded once to fp32.  Order-independent, deterministic on every host.  Needs float64.
  * dot="fp32": a plain fp32 matmul of the e4m3 values per K tile (BLAS summation order).  No float64; this
    is what a device without float64 runs.
  * fma=True: `acc + t * b_s` contracted into one rounding, as Triton/LLVM fuse it on CUDA (emulated in
    float64, so a double rounding is possible in ~2^-29 of the cases).  Default False (IEEE, separate ops).
  * scale_div="ieee": a_s = amax / 448 with IEEE division (torch's `/`, the source's literal meaning).
    scale_div="recip": a_s = amax * fl32(1/448), which is what Triton emits for `/ 448.0` on CUDA (measured
    bit-identical on GB10 sm_121 for 100% of groups; it differs from IEEE in ~1 ulp for about half of them).
    The element division x / a_s is IEEE in both (measured identical to the CUDA kernel), and the cast to
    e4m3 is round-to-nearest-even (identical to torch's cast on CUDA as well).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0                 # torch.finfo(torch.float8_e4m3fn).max; the kernel's literal 448.0
ACT_SCALE_FLOOR = 1e-12         # the kernel's tl.maximum(a_s, 1e-12)
UE8M0_DTYPES = tuple(d for d in (getattr(torch, "float8_e8m0fnu", None), torch.uint8) if d is not None)
DOT_MODES = ("exact", "fp32")
SCALE_DIVS = ("ieee", "recip")
_RECIP_448 = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)   # fl32(1/448)


# ---- scales --------------------------------------------------------------------------------------
def decode_scale(scale: torch.Tensor) -> torch.Tensor:
    """Weight scales to fp32 the way the kernel decodes them: ue8m0 exponent bytes -> 2^(e-127) (built as
    the fp32 bit pattern e << 23); fp32 / bf16 / fp16 scales are converted exactly."""
    if scale.dtype in UE8M0_DTYPES:
        bits = scale.contiguous().view(torch.uint8).to(torch.int32) << 23
        return bits.view(torch.float32)
    return scale.to(torch.float32)


def act_quant_dynamic(x: torch.Tensor, block_k: int, scale_div: str = "ieee") -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-row, per-K-group activation quantization (fp8_act_quant_inline / _fp8_act_quant_kernel).

    x [..., K] any float dtype -> (q [..., Kp] float8_e4m3fn, s [..., Kp // block_k] fp32), Kp = K rounded up to
    a multiple of block_k with zeros (the kernel's masked load `other=0.0`).  s is NOT floored: a zero group has
    s = 0 (and q = 0), exactly as the kernel stores and uses it."""
    xf = x.to(torch.float32)
    pad = (-xf.shape[-1]) % block_k
    if pad:
        xf = F.pad(xf, (0, pad))
    g = xf.unflatten(-1, (-1, block_k))
    amax = g.abs().amax(dim=-1)
    if scale_div == "ieee":
        s = amax / FP8_MAX
    elif scale_div == "recip":
        s = amax * _RECIP_448.to(amax.device)
    else:
        raise ValueError(f"scale_div must be one of {SCALE_DIVS}, got {scale_div!r}")
    q = (g / s.clamp_min(ACT_SCALE_FLOOR).unsqueeze(-1)).to(FP8_DTYPE)
    return q.flatten(-2), s


def act_quant_static(x: torch.Tensor, scale: torch.Tensor, block_k: int) -> torch.Tensor:
    """Static per-tensor activation quantization (w8a8_block_static_fp8_matmul_kernel: (a / a_s).to(e4m3)).
    The kernel does not clamp; Triton's fp32->e4m3 conversion on NVIDIA saturates (cvt.rn.satfinite), while
    torch's cast gives NaN above 464, so the saturation is made explicit with a clamp."""
    xf = x.to(torch.float32)
    pad = (-xf.shape[-1]) % block_k
    if pad:
        xf = F.pad(xf, (0, pad))
    return (xf / scale.reshape(()).to(torch.float32)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


# ---- weights ----------------------------------------------------------------------------------------
def dequant_weight(weight: torch.Tensor, scale: torch.Tensor, block_size, out_dtype=torch.bfloat16) -> torch.Tensor:
    """W[n, k] * Bs[n // bn, k // bk] in fp32, then cast: transformers' Fp8Dequantize._dequantize_one math
    (q.float() * s.float()).to(dtype) for 2-D FP8 weights.  (Fp8Dequantize derives the block from the scale
    grid, rows // scale_rows; it equals bn whenever N and K are multiples of the block, as in every checkpoint
    used here.)"""
    n, k = weight.shape
    bn, bk = _block(block_size, weight)
    s = decode_scale(scale)
    s = s.reshape(1, 1) if s.numel() == 1 else s
    s_full = s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return (weight.to(torch.float32) * s_full).to(out_dtype)


def _block(block_size, weight):
    n, k = weight.shape[-2:]
    if block_size is None:
        return n, k
    return int(block_size[0]), int(block_size[1])


def is_tensor_wide(block_size, weight) -> bool:
    return block_size is None or (int(block_size[0]) == weight.shape[-2] and int(block_size[1]) == weight.shape[-1])


def is_mx(weight, scale) -> bool:
    return weight.dtype == torch.int8 or (scale.dtype in UE8M0_DTYPES and scale.ndim == weight.ndim
                                           and scale.shape[-1] == weight.shape[-1] // 32)


# ---- the K-tile dot ---------------------------------------------------------------------------------
def _tile_dot(a_q: torch.Tensor, b_q: torch.Tensor, dot: str) -> torch.Tensor:
    """a_q [M, bk] e4m3, b_q [N, bk] e4m3 -> [M, N] fp32 (exact-then-rounded, or plain fp32 matmul)."""
    if dot == "exact":
        return (a_q.to(torch.float64) @ b_q.to(torch.float64).t()).to(torch.float32)
    if dot == "fp32":
        return a_q.to(torch.float32) @ b_q.to(torch.float32).t()
    raise ValueError(f"dot must be one of {DOT_MODES}, got {dot!r}")


def _tile_dots(a_q: torch.Tensor, b_q: torch.Tensor, dot: str) -> torch.Tensor:
    """All K tiles at once: a_q [KB, M, bk], b_q [KB, N, bk] (e4m3) -> [KB, M, N] fp32.  Same values as
    _tile_dot per tile (exact: order-independent; fp32: one fp32 matmul per tile)."""
    if dot == "exact":
        return torch.bmm(a_q.to(torch.float64), b_q.to(torch.float64).transpose(1, 2)).to(torch.float32)
    if dot == "fp32":
        return torch.bmm(a_q.to(torch.float32), b_q.to(torch.float32).transpose(1, 2))
    raise ValueError(f"dot must be one of {DOT_MODES}, got {dot!r}")


def _accumulate(acc, t, b_s, fma: bool):
    """acc += t * b_s  (t = tile_dot * a_s already rounded to fp32)."""
    if fma:
        return (acc.to(torch.float64) + t.to(torch.float64) * b_s.to(torch.float64)).to(torch.float32)
    return acc + t * b_s


# ---- matmuls ----------------------------------------------------------------------------------------
def w8a8_block_matmul(A, B, Bs, block_size, output_dtype=None, activation_scale=None, *, dot="exact", fma=False,
                      scale_div="ieee"):
    """C = A @ B.T with block-wise FP8 weights; dynamic per-row/per-K-tile activation quantization (or static
    with activation_scale).  A [..., K] raw activations, B [N, K] e4m3, Bs [ceil(N/bn), ceil(K/bk)]."""
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D [N, K], got {tuple(B.shape)}")
    n, k = B.shape
    if A.shape[-1] != k:
        raise ValueError(f"K mismatch: A {tuple(A.shape)} vs B {tuple(B.shape)}")
    bn, bk = _block(block_size, B)
    if tuple(Bs.shape) != (math.ceil(n / bn), math.ceil(k / bk)):
        raise ValueError(f"Bs shape {tuple(Bs.shape)} != ({math.ceil(n / bn)}, {math.ceil(k / bk)})")
    out_dtype = A.dtype if output_dtype is None else output_dtype
    m = A.numel() // k
    a = A.reshape(m, k)
    if activation_scale is None:
        qa, sa = act_quant_dynamic(a, bk, scale_div)                       # [M, Kp], [M, KB]
    else:
        qa, sa = act_quant_static(a, activation_scale, bk), None
    kb = qa.shape[-1] // bk
    if qa.shape[-1] != k:
        B = F.pad(B.view(torch.uint8), (0, qa.shape[-1] - k)).view(FP8_DTYPE)   # zero codes beyond K
    b_s = decode_scale(Bs).repeat_interleave(bn, 0)[:n].t().unsqueeze(1)  # [KB, 1, N]: scale of column n, tile k
    d = _tile_dots(qa.view(m, kb, bk).transpose(0, 1), B.view(n, kb, bk).transpose(0, 1), dot)   # [KB, M, N] fp32
    if sa is not None:
        d = d * sa.t().unsqueeze(-1)                                       # (dot * a_s) ...
    acc = torch.zeros(m, n, dtype=torch.float32, device=A.device)
    for t in range(kb):                                                    # K tiles in order, as the kernel loops
        acc = _accumulate(acc, d[t], b_s[t], fma)                          # ... * b_s, then +=
    if sa is None:
        acc = acc * activation_scale.reshape(()).to(torch.float32)         # static: a_s applied once at the end
    return acc.to(out_dtype).reshape(*A.shape[:-1], n)


def w8a8_tensor_matmul(A, B, Bs, output_dtype=None, *, dot="exact", fma=False, scale_div="ieee"):
    """Tensor-wide weight scale (w8a8_tensor_dynamic_fp8_matmul): fp8_act_quant(A, K) per row over the whole
    K, raw fp32 accumulation over K, then acc * a_s[:, None] * b_s."""
    n, k = B.shape
    out_dtype = A.dtype if output_dtype is None else output_dtype
    m = A.numel() // k
    qa, sa = act_quant_dynamic(A.reshape(m, k), k, scale_div)              # one scale per row
    acc = _tile_dot(qa, B, dot)
    acc = acc * sa[:, :1] * decode_scale(Bs).reshape(())
    return acc.to(out_dtype).reshape(*A.shape[:-1], n)


def matmul_2d(A, B, Bs, block_size, output_dtype=None, activation_scale=None, *, dot="exact", fma=False,
              scale_div="ieee"):
    """Same signature and dispatch as the Hub kernel's matmul_2d (the entry FP8Linear reaches)."""
    if is_mx(B, Bs):
        raise NotImplementedError("MX / FP4 weights (group-32 ue8m0 scales) are not covered by this reference")
    if is_tensor_wide(block_size, B):
        if activation_scale is not None:
            raise NotImplementedError("static activation scale with a tensor-wide weight scale (the kernel refuses it too)")
        return w8a8_tensor_matmul(A, B, Bs, output_dtype, dot=dot, fma=fma, scale_div=scale_div)
    return w8a8_block_matmul(A, B, Bs, block_size, output_dtype, activation_scale, dot=dot, fma=fma,
                             scale_div=scale_div)


def exact_w8a8(A, B, Bs, block_size, activation_scale=None, scale_div="ieee") -> torch.Tensor:
    """The same quantized operands with no rounding after quantization at all (float64 everywhere): the value
    every correct implementation approximates before its final cast.  [..., N] float64."""
    n, k = B.shape
    bn, bk = _block(block_size, B)
    m = A.numel() // k
    a = A.reshape(m, k)
    if activation_scale is None:
        qa, sa = act_quant_dynamic(a, bk, scale_div)
        a_deq = (qa.to(torch.float64).unflatten(-1, (-1, bk)) * sa.to(torch.float64).unsqueeze(-1)).flatten(-2)[:, :k]
    else:
        a_deq = act_quant_static(a, activation_scale, bk).to(torch.float64)[:, :k] * activation_scale.double()
    w = dequant_weight(B, Bs, (bn, bk), torch.float64)
    return (a_deq @ w.t()).reshape(*A.shape[:-1], n)


# ---- the kernel bundle transformers calls ------------------------------------------------------------
class ReferenceKernel:
    """Duck-type of the `kernels-community/finegrained-fp8` module as transformers uses it
    (FineGrainedFP8(matmul=..., batched_matmul=..., grouped_matmul=...)).  Only the dense 2-D path exists."""

    def __init__(self, dot="exact", fma=False, scale_div="ieee"):
        if dot not in DOT_MODES or scale_div not in SCALE_DIVS:
            raise ValueError(f"dot must be one of {DOT_MODES}, scale_div one of {SCALE_DIVS}")
        self.dot, self.fma, self.scale_div = dot, fma, scale_div

    def describe(self):
        return dict(kind="reference", module=__name__, dot=self.dot, fma=self.fma, scale_div=self.scale_div)

    def matmul(self, A, B, Bs, block_size, output_dtype=None, activation_scale=None):
        return matmul_2d(A, B, Bs, block_size, output_dtype, activation_scale, dot=self.dot, fma=self.fma,
                         scale_div=self.scale_div)

    def batched_matmul(self, *a, **k):
        raise NotImplementedError("FP8 MoE batched matmul is not covered by the reference")

    grouped_matmul = batched_matmul


def install_into_transformers(kernel) -> object:
    """Make transformers' FP8Linear (and only it) compute through `kernel` (a ReferenceKernel, or the Hub
    module itself): fills the module global `_FINEGRAINED_FP8` that `load_finegrained_fp8_kernel()` returns.
    Keyed on nothing but the FP8Linear class transformers itself builds from `quantization_config`; no model
    code or layer names involved.  Returns the previous bundle (None = not loaded yet)."""
    from transformers.integrations import finegrained_fp8 as tf_fp8
    prev = tf_fp8._FINEGRAINED_FP8
    tf_fp8._FINEGRAINED_FP8 = tf_fp8.FineGrainedFP8(matmul=kernel.matmul, batched_matmul=kernel.batched_matmul,
                                                      grouped_matmul=kernel.grouped_matmul)
    return prev


# ---- literal re-derivation (validation only) -----------------------------------------------------------
def literal_kernel_emulation(A, B, Bs, block_size, output_dtype=None):
    """Slow, line-by-line transcription of w8a8_block_dynamic_fp8_matmul_kernel's tile loop (grid of
    BLOCK_SIZE_M x BLOCK_SIZE_N output tiles; inline per-tile act quant; exact tile dot rounded to fp32;
    `acc += dot * a_s[:, None] * b_s[None, :]`; masked store).  Independent code path from
    w8a8_block_matmul, used to check it bit for bit where no CUDA device exists."""
    n, k = B.shape
    bn, bk = _block(block_size, B)
    m = A.numel() // k
    a2 = A.reshape(m, k)
    bm = min(max(1 << max(m - 1, 0).bit_length(), 16), 128)               # adaptive_block_size_m
    out = torch.empty(m, n, dtype=A.dtype if output_dtype is None else output_dtype)
    bsf = decode_scale(Bs)
    bf = B.to(torch.float32)                                              # exact; indexing float8 is avoided
    for pm in range(math.ceil(m / bm)):
        rows = torch.tensor([(pm * bm + i) % m for i in range(bm)])      # offs_am = (...) % M
        for pn in range(math.ceil(n / bn)):
            cols = torch.tensor([(pn * bn + j) % n for j in range(bn)])  # offs_bn = (...) % N
            acc = torch.zeros(bm, bn, dtype=torch.float32)
            for kt in range(math.ceil(k / bk)):
                kk = torch.arange(kt * bk, (kt + 1) * bk)
                valid = kk < k                                            # mask=offs_k < k_remaining
                kc = kk.clamp_max(k - 1)
                a_raw = torch.where(valid[None, :], a2[rows][:, kc].to(torch.float32), torch.zeros(()))
                a_s = a_raw.abs().max(dim=1).values / 448.0
                a_q = (a_raw / torch.maximum(a_s[:, None], torch.tensor(1e-12))).to(FP8_DTYPE)
                b = torch.where(valid[None, :], bf[cols][:, kc], torch.zeros(()))
                b_s = bsf[cols // bn, kt]
                d = (a_q.double() @ b.double().t()).float()                 # tl.dot, exact then fp32
                acc += d * a_s[:, None] * b_s[None, :]
            c = acc.to(out.dtype)                                         # _store_masked
            r1, c1 = min(m, (pm + 1) * bm), min(n, (pn + 1) * bn)
            out[pm * bm:r1, pn * bn:c1] = c[:r1 - pm * bm, :c1 - pn * bn]
    return out.reshape(*A.shape[:-1], n)


# ---- validation CLI --------------------------------------------------------------------------------
def _ordered(t: torch.Tensor) -> torch.Tensor:
    """Float bit patterns on a monotone integer line (so |a - b| counts representable values between)."""
    t = t.contiguous()
    view = {4: torch.int32, 2: torch.int16, 1: torch.int8}[t.element_size()]
    i = t.view(view).long()
    top = 1 << (t.element_size() * 8 - 1)
    return torch.where(i < 0, -(i + top), i)


def _ulp_stats(x: torch.Tensor, y: torch.Tensor) -> dict:
    """x vs y (same dtype): bitwise equality, ulp distance where both have the same sign, sign flips (values
    near zero after cancellation), relative L2 and the largest error relative to the row's magnitude."""
    d = (_ordered(x) - _ordered(y)).abs()
    xd, yd = x.double(), y.double()
    same_sign = (torch.signbit(xd) == torch.signbit(yd)) | (xd == 0) | (yd == 0)
    rowmax = yd.abs().reshape(-1, yd.shape[-1]).amax(-1, keepdim=True).clamp_min(1e-300) if yd.ndim else yd.abs()
    err = (xd - yd).abs().reshape(-1, yd.shape[-1]) / rowmax if yd.ndim else (xd - yd).abs()
    return dict(n=int(x.numel()), bitwise_equal=bool(torch.equal(_ordered(x), _ordered(y))),
                frac_equal=float((d == 0).double().mean()), n_diff=int((d != 0).sum()),
                max_ulp_same_sign=int(d[same_sign].max()) if bool(same_sign.any()) else 0,
                n_sign_flips=int((~same_sign).sum()),
                max_err_over_row_absmax=float(err.max()) if err.numel() else 0.0,
                rel_l2=float((xd - yd).norm() / yd.norm().clamp_min(1e-300)))


def _rel(x, y):
    x, y = x.double(), y.double()
    return float((x - y).norm() / y.norm().clamp_min(1e-300))


def _synthetic_activations(m, k, gen, dtype=torch.bfloat16):
    """Heavy-tailed rows: Gaussian body, a few outlier channels, one all-zero row, a row scaled by 1e-4, and
    magnitudes that differ across K tiles (so some groups land in the e4m3 subnormal range)."""
    x = torch.randn(m, k, generator=gen)
    outl = torch.randint(0, k, (max(1, k // 64),), generator=gen)
    x[:, outl] *= 30.0
    tiles = math.ceil(k / 128)
    x *= torch.logspace(-3, 1, tiles).repeat_interleave(128)[:k]          # magnitudes differ across K tiles
    if m > 2:
        x[1] = 0.0
        x[2] *= 1e-4
    return x.to(dtype)


def _synthetic_weight(n, k, gen, scale_dtype=torch.float32, block=(128, 128)):
    w = torch.randn(n, k, generator=gen) * 0.02
    bn, bk = block
    nb, kb = math.ceil(n / bn), math.ceil(k / bk)
    wp = F.pad(w, (0, kb * bk - k, 0, nb * bn - n)).reshape(nb, bn, kb, bk)
    s = wp.abs().amax(dim=(1, 3)) / FP8_MAX
    s = s.to(scale_dtype).float().clamp_min(1e-12)
    q = (wp / s[:, None, :, None]).reshape(nb * bn, kb * bk)[:n, :k].to(FP8_DTYPE)
    return q, s.to(scale_dtype)


def _load_hub_kernel(args):
    """The official Hub kernel exactly as transformers loads it (lazy_load_kernel -> kernels.get_kernel,
    repo kernels-community/finegrained-fp8, version 4), from the local HF cache."""
    import os
    os.environ.setdefault("TNCA_TRITON", "0")        # tnca's triton driver stays inactive: kernels run on CUDA
    for p in reversed(args.triton_pythonpath or []):
        sys.path.insert(0, p)
    from transformers.integrations.hub_kernels import lazy_load_kernel
    mod = lazy_load_kernel("finegrained-fp8")
    if mod is None:
        raise SystemExit("Hub kernel finegrained-fp8 could not be loaded")
    return mod


VARIANTS = [(d, fm, sd) for sd in SCALE_DIVS for d in DOT_MODES for fm in (False, True)]


def _tag(d, fm, sd):
    return f"{d}{'_fma' if fm else ''}_{sd}"


def cmd_validate(args):
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(0)
    result = dict(schema="tnca-fp8-blockwise-ref-validate/2", torch=torch.__version__, checks={},
                  variant_names="<dot>[_fma]_<scale_div>: dot exact|fp32, fma contraction, scale_div ieee|recip")
    shapes = [(1, 256, 384), (7, 384, 256), (16, 1024, 2048), (64, 2048, 1024), (130, 256, 6144), (5, 200, 328)]
    cases = []
    for (m, n, k) in shapes:
        for sd in (torch.float32, torch.bfloat16):
            wq, ws = _synthetic_weight(n, k, gen, sd)
            cases.append(dict(name=f"M{m}_N{n}_K{k}_scale-{str(sd)[6:]}", x=_synthetic_activations(m, k, gen), w=wq, s=ws))
    n_synth = len(cases)
    for path, keys in zip(args.checkpoint or [], args.layers or []):
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            for key in keys.split(","):
                wq, ws = f.get_tensor(key + ".weight"), f.get_tensor(key + ".weight_scale_inv")
                for m in (1, 37):
                    cases.append(dict(name=f"{Path(path).parent.parent.parent.name}:{key}:M{m}",
                                      x=_synthetic_activations(m, wq.shape[1], gen), w=wq, s=ws))
    block = [128, 128]

    # 1. vectorized reference == literal transcription of the kernel's tile loop, bit for bit (synthetic cases)
    lit = {}
    for c in cases[:n_synth]:
        t0 = time.time()
        lit[c["name"]] = dict(**_ulp_stats(w8a8_block_matmul(c["x"], c["w"], c["s"], block, torch.float32, dot="exact"),
                                           literal_kernel_emulation(c["x"], c["w"], c["s"], block, torch.float32)),
                              seconds=round(time.time() - t0, 1))
    result["checks"]["reference_vs_literal_kernel_transcription_fp32"] = lit

    # 2. variants and the dequantize-to-BF16 alternative, against the exact W8A8 value
    var = {}
    for c in cases:
        x, w, s = c["x"], c["w"], c["s"]
        exact = exact_w8a8(x, w, s, block)
        ref32 = {_tag(*v): w8a8_block_matmul(x, w, s, block, torch.float32, dot=v[0], fma=v[1], scale_div=v[2]) for v in VARIANTS}
        refbf = w8a8_block_matmul(x, w, s, block, dot="exact")
        deq = F.linear(x, dequant_weight(w, s, block, x.dtype))
        unq = x.double() @ dequant_weight(w, s, block, torch.float64).t()          # activations not quantized
        var[c["name"]] = dict(
            fp32_out_rel_l2_vs_exact={k2: _rel(v, exact) for k2, v in ref32.items()},
            fp32_dot_vs_exact_dot=_ulp_stats(ref32["fp32_ieee"], ref32["exact_ieee"]),
            fma_vs_nofma=_ulp_stats(ref32["exact_fma_ieee"], ref32["exact_ieee"]),
            recip_vs_ieee_scale=_ulp_stats(ref32["exact_recip"], ref32["exact_ieee"]),
            bf16_out_rel_l2_vs_exact=_rel(refbf, exact),
            bf16_rounding_of_exact=_rel(exact.to(torch.bfloat16), exact),
            dequant_bf16_rel_l2_vs_exact_w8a8=_rel(deq, exact),
            dequant_bf16_vs_w8a8_bf16=_ulp_stats(deq, refbf),
            w8a8_rel_l2_vs_unquantized_activation=_rel(refbf, unq),
            dequant_bf16_rel_l2_vs_unquantized_activation=_rel(deq, unq))
    result["checks"]["variants"] = var

    # 3. against the Hub kernel itself, on CUDA
    if args.cuda:
        hub = _load_hub_kernel(args)
        from transformers.integrations.finegrained_fp8 import fp8_linear
        result["hub_kernel"] = dict(module=getattr(hub, "__name__", str(hub)), file=getattr(hub, "__file__", None),
                                    device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        hk = {}
        for c in cases:
            x, w, s = c["x"], c["w"], c["s"]
            xc, wc, sc = x.cuda().contiguous(), w.cuda(), s.cuda()
            row = {}
            if x.shape[-1] % 128 == 0:                 # the standalone act-quant kernel (same math as the inline one)
                hq, hs = (t.cpu() for t in hub.fp8_act_quant(xc, 128))
                for sd in SCALE_DIVS:
                    rq, rs = act_quant_dynamic(x, 128, sd)
                    row[f"act_quant_codes_{sd}"] = _ulp_stats(hq, rq)
                    row[f"act_quant_scales_{sd}"] = _ulp_stats(hs, rs)
            h32 = hub.matmul_2d(xc, wc, sc, block, torch.float32).cpu()
            hbf = hub.matmul_2d(xc, wc, sc, block, x.dtype).cpu()
            for v in VARIANTS:
                row[f"fp32_out_vs_ref_{_tag(*v)}"] = _ulp_stats(
                    h32, w8a8_block_matmul(x, w, s, block, torch.float32, dot=v[0], fma=v[1], scale_div=v[2]))
            for v in VARIANTS:
                if not v[1]:
                    row[f"bf16_out_vs_ref_{_tag(*v)}"] = _ulp_stats(
                        hbf, w8a8_block_matmul(x, w, s, block, dot=v[0], fma=v[1], scale_div=v[2]))
            row["hub_fp32_rel_l2_vs_exact_ieee"] = _rel(h32, exact_w8a8(x, w, s, block))
            row["hub_fp32_rel_l2_vs_exact_recip"] = _rel(h32, exact_w8a8(x, w, s, block, scale_div="recip"))
            row["hub_bf16_rel_l2_vs_exact_ieee"] = _rel(hbf, exact_w8a8(x, w, s, block))
            # transformers' own entry (fp8_linear -> finegrained_fp8_linear -> kernel), output dtype = input dtype
            row["transformers_fp8_linear_vs_hub_matmul_2d"] = _ulp_stats(fp8_linear(xc, wc, sc, block_size=block).cpu(), hbf)
            hk[c["name"]] = row
        result["checks"]["hub_kernel_cuda"] = hk
        result["summary"] = _summary(hk)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps(result.get("summary", {}), indent=1))
    print("literal:", {k2: (v["bitwise_equal"], v["max_ulp_same_sign"]) for k2, v in lit.items()})


def _summary(hk):
    keys = sorted({k for r in hk.values() for k in r if isinstance(r[k], dict)})
    s = {}
    for k in keys:
        rows = [r[k] for r in hk.values() if k in r]
        s[k] = dict(cases=len(rows), all_bitwise=all(r["bitwise_equal"] for r in rows),
                    min_frac_equal=min(r["frac_equal"] for r in rows),
                    max_ulp_same_sign=max(r["max_ulp_same_sign"] for r in rows),
                    sign_flips=sum(r["n_sign_flips"] for r in rows),
                    max_err_over_row_absmax=max(r["max_err_over_row_absmax"] for r in rows),
                    max_rel_l2=max(r["rel_l2"] for r in rows))
    for k in sorted({k for r in hk.values() for k in r if isinstance(r[k], float)}):
        s[k] = dict(max=max(r[k] for r in hk.values()))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="check this reference (literal transcription; and the Hub kernel with --cuda)")
    p.add_argument("--out", required=True)
    p.add_argument("--cuda", action="store_true", help="also run the official Hub kernel on the CUDA device and compare")
    p.add_argument("--triton-pythonpath", nargs="*", help="dirs put first on sys.path before triton is imported "
                   "(torchenv's triton discovers a tnca backend and needs the tnca package importable)")
    p.add_argument("--checkpoint", nargs="*", help="safetensors files with real FP8 layers")
    p.add_argument("--layers", nargs="*", help="per checkpoint: comma-separated layer prefixes")
    p.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dict(validate=cmd_validate)[a.cmd](a)


if __name__ == "__main__":
    main()
