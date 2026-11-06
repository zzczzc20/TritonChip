"""Plain-Torch references for block-scaled FP8 kernels, in the order TNCA executes them.

Independent of the compiler and of Triton: every value is re-derived here with
torch CPU arithmetic on the kernel's inputs, one binary32 rounding where the
compiled kernel rounds once, in the order the hardware performs the steps.

  * `tl.dot` of two FP8 tiles: ISA sec 9C.1 accumulates one product per K index,
    in ascending K, from +0.0, in binary32. An FP8 x FP8 product is exact in
    binary32, so each step is one rounding of `d + q*w` -- a torch add.
  * The per-K-block rescale `acc += dot * a_s[:, None] * b_s[None, :]` is three
    roundings, as written: (dot * a_s), (* b_s), (acc +). TNCA does not contract
    a multiply and an add into an FMA.
  * Division. The ISA has none (sec 8.2); the compiler forms `a / y` as `a * r`,
    with `r` the SFU reciprocal (sec 8.8) refined by one Newton step
    (compiler/python/tnca_cc/mathlib.py: faithful, <= 1 ulp). `device_rcp` below
    evaluates exactly that: a bit-exact port of xls/tnca_sfu.x's reciprocal, read
    from the same coefficient table the RTL and the SystemC model use, and the two
    FMAs of the refinement with one rounding each. `ieee=True` swaps in IEEE
    division instead, which is what CPU torch computes; comparing the two
    attributes every difference to the faithful division and nothing else.

Nothing here names a model or a layer.
"""
from fractions import Fraction
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[2]

F32_1E_12 = struct.unpack('<f', struct.pack('<f', 1e-12))[0]   # Triton's 1e-12, as binary32


# ------------------------------------------------------------- SFU port ----
_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        text = (ROOT / 'xls' / 'tnca_sfu_tables.x').read_text()
        consts = {n: int(v) for n, v in re.findall(r'pub const (\w+): u32 = u32:(\d+);', text)}
        tables = {n: [int(v) for v in re.findall(r's32:(-?\d+)', body)]
                  for n, _size, body in re.findall(r'pub const (\w+) = s32\[(\d+)\]:\[(.*?)\];', text, re.S)}
        _TABLES = consts, tables
    return _TABLES


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _s64(v):
    v &= (1 << 64) - 1
    return v - (1 << 64) if v & (1 << 63) else v


_FRAC = 30


def _poly2(c0, c1, c2, u):
    t = _s32(c1 + _s32((c2 * u) >> _FRAC))
    return _s32(c0 + _s32((t * u) >> _FRAC))


def _pack(sign, unb, mant):
    round_add = _s64(mant + (1 << 6))
    carried = round_add >= (1 << 31)
    m = round_add >> 1 if carried else round_add
    e = _s32(unb + 1) if carried else unb
    frac = (m >> 7) & 0x7FFFFF
    bexp = _s32(e + 127)
    if bexp >= 255:
        return (sign << 31) | 0x7F800000
    if bexp <= 0:
        return sign << 31
    return (sign << 31) | ((bexp & 255) << 23) | frac


def sfu_rcp_bits(x):
    """`v.rcp.approx` on one binary32 bit pattern (xls/tnca_sfu.x, tnca_sfu.hpp)."""
    consts, tables = _tables()
    log2seg = consts['LOG2_SEGMENTS']
    sign, bexp, frac = x >> 31, (x >> 23) & 255, x & 0x7FFFFF
    unb = bexp - 127
    idx = frac >> (23 - log2seg)
    f30 = _s32(frac << (_FRAC - 23))
    u = _s32(f30 - _s32(idx << (_FRAC - log2seg)))
    v = _poly2(tables['RCP_C0'][idx], tables['RCP_C1'][idx], tables['RCP_C2'][idx], u)
    low = v < (1 << _FRAC)
    mant = _s64(v << 1) if low else v
    e = -unb - 1 if low else -unb
    if bexp == 255 and frac:
        return 0x7FC00000
    if bexp == 0:
        return (sign << 31) | 0x7F800000
    if bexp == 255:
        return sign << 31
    return _pack(sign, e, mant)


# ----------------------------------------------------- exact binary32 ----
def f32_bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def f32_from_bits(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def _is_nan_bits(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x7FFFFF) != 0


def _round_f32(q):
    """The binary32 nearest the rational `q`, ties to even (bits)."""
    if q == 0:
        return 0
    sign = 1 if q < 0 else 0
    q = abs(q)
    # floor(log2 q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    e = max(e, -126)                      # subnormals share the minimum exponent
    scaled = q * Fraction(2) ** (23 - e)  # the significand in units of the last place
    n, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and n & 1):
        n += 1
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return (sign << 31) | 0x7F800000
    if n < 1 << 23:                       # subnormal
        return (sign << 31) | n
    return (sign << 31) | ((e + 127) << 23) | (n - (1 << 23))


def _exact(b):
    sign = -1 if b >> 31 else 1
    bexp, frac = (b >> 23) & 255, b & 0x7FFFFF
    if bexp == 0:
        return sign * Fraction(frac, 1 << 149)
    return sign * Fraction((1 << 23) | frac, 1) * Fraction(2) ** (bexp - 150)


def fma_bits(a, b, c):
    """fma(a, b, c) on binary32 bit patterns, one rounding; a NaN result is
    the canonical quiet NaN (ISA sec 8.2, tnca_numeric.hpp fma32)."""
    import math
    if any((x & 0x7F800000) == 0x7F800000 for x in (a, b, c)):
        fa, fb, fc = (f32_from_bits(x) for x in (a, b, c))
        v = fa * fb + fc if not (math.isinf(fa) and fb == 0 or math.isinf(fb) and fa == 0) else math.nan
        return 0x7FC00000 if math.isnan(v) else f32_bits(v)
    p = _exact(a) * _exact(b)
    q = p + _exact(c)
    if q == 0:
        # IEEE 754 sec 6.3: an exact zero sum is +0 in round-to-nearest,
        # except (-0) + (-0).
        both_negative_zero = p == 0 and (a ^ b) >> 31 and c == 0x80000000
        return 0x80000000 if both_negative_zero else 0
    return _round_f32(q)


_RCP_CACHE = {}


def device_rcp_bits(x):
    """1/x as mathlib.rcp emits it: SFU reciprocal, then one Newton step
    (e = x*r0 - 1, t = r0*e - r0, r1 = -t), keeping r0 where r1 is NaN."""
    r = _RCP_CACHE.get(x)
    if r is None:
        r0 = sfu_rcp_bits(x)
        e = fma_bits(x, r0, f32_bits(-1.0))
        t = fma_bits(r0, e, r0 ^ 0x80000000)
        r1 = t ^ 0x80000000 if not _is_nan_bits(t) else 0x7FC00000
        r = r0 if _is_nan_bits(r1) else r1
        _RCP_CACHE[x] = r
    return r


# ------------------------------------------------------- torch helpers ----
def _torch():
    import torch
    return torch


def device_rcp(t):
    """Elementwise device reciprocal of a float32 tensor."""
    torch = _torch()
    flat = t.detach().cpu().contiguous().view(torch.int32).flatten().tolist()
    out = [device_rcp_bits(b & 0xFFFFFFFF) for b in flat]
    out = [v - (1 << 32) if v & 0x80000000 else v for v in out]
    return torch.tensor(out, dtype=torch.int32).view(torch.float32).reshape(t.shape)


def divide(a, y, ieee=False):
    """`a / y` (y broadcast) the way the compiled kernel divides, or IEEE."""
    torch = _torch()
    if ieee:
        return a / y
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    return a * device_rcp(y.float())


# Which `tl.dot` reduction the reference follows. 'mma' is ISA sec 9C.1 (one
# rounding per K index); 'mma.dp' is sec 9C.4's m.mma.dp.async for E4M3 x E4M3
# (groups of four consecutive K, one rounding per group). It is set by the
# harness from the profile's MATRIX_DP capability, never from a kernel's name.
_DOT_MODE = 'mma'


def set_dot_mode(mode):
    global _DOT_MODE
    if mode not in ('mma', 'mma.dp'):
        raise ValueError(mode)
    _DOT_MODE = mode


def dot_mode_for_profile(profile):
    """'mma.dp' when the profile has CAPABILITY0.MATRIX_DP (the compiler then
    lowers every matrix command to m.mma.dp.async), else 'mma'. The profile is
    the only input; no kernel, shape or model is consulted."""
    import sys
    sys.path.insert(0, str(ROOT / 'python'))
    from tnca import profiles
    return 'mma.dp' if profiles.target(profile)['capabilities'].get('matrix_dp') else 'mma'


def fp8_dot(q, w):
    """sum_k q[:, k] * w[:, k] for FP8-valued float32 q (M, K) and w (N, K),
    ascending K, from +0.0, in binary32: one rounding per K index (ISA sec 9C.1),
    or one per group of four K (sec 9C.4) when the dot mode is 'mma.dp'."""
    if _DOT_MODE == 'mma.dp':
        return fp8_dot_dp(q, w)
    torch = _torch()
    d = torch.zeros(q.shape[0], w.shape[0], dtype=torch.float32)
    for k in range(q.shape[1]):
        d = d + q[:, k:k + 1] * w[:, k][None, :]
    return d


def fp8_dot_dp(q, w, group=4):
    """E4M3 x E4M3 grouped dot product, exactly as ISA sec 9C.4 defines it.

    Every E4M3 x E4M3 group sum is a multiple of 2^-18 below 2^38 in magnitude
    (tools/model/fdp.py checks this over all 256 encodings), so it is exact in
    float64 and RO_48 is the identity. C + S is then one float64 add with its
    exact error term (TwoSum); the binary32 rounding of that pair is the
    correctly rounded RNE_f32(C + S): float64 rounding of a value that is not
    a binary32 midpoint cannot cross a midpoint, and the one case it can hit
    (the float64 sum lands on a midpoint) is resolved by the error's sign.
    Zero padding to a whole group adds exactly zero, as the transfer engine's
    zero fill does. Independent of the compiler, the model and the RTL."""
    import numpy as np
    torch = _torch()
    qa = q.detach().cpu().double().numpy()
    wa = w.detach().cpu().double().numpy()
    m, kk = qa.shape
    n = wa.shape[0]
    pad = (-kk) % group
    if pad:
        qa = np.pad(qa, ((0, 0), (0, pad)))
        wa = np.pad(wa, ((0, 0), (0, pad)))
    c = np.zeros((m, n), dtype=np.float32)
    for k0 in range(0, qa.shape[1], group):
        sgrp = np.zeros((m, n))
        for j in range(group):
            sgrp += qa[:, k0 + j][:, None] * wa[:, k0 + j][None, :]
        cd = c.astype(np.float64)
        t = cd + sgrp
        bb = t - cd
        err = (cd - (t - bb)) + (sgrp - bb)           # exact: C + S == t + err
        r = t.astype(np.float32)
        rd = r.astype(np.float64)
        diff = t - rd
        up = np.nextafter(r, np.float32(np.inf)).astype(np.float64)
        dn = np.nextafter(r, np.float32(-np.inf)).astype(np.float64)
        with np.errstate(invalid='ignore'):
            tie_up = np.isfinite(t) & (diff == (up - rd) / 2) & (err != 0)
            tie_dn = np.isfinite(t) & (diff == (dn - rd) / 2) & (err != 0)
        r = np.where(tie_up & (err > 0), np.nextafter(r, np.float32(np.inf)), r)
        r = np.where(tie_dn & (err < 0), np.nextafter(r, np.float32(-np.inf)), r)
        c = r.astype(np.float32)
    return torch.from_numpy(c)


_FDP_FORMAT = {'torch.float16': 'f16', 'torch.bfloat16': 'bf16',
               'torch.float8_e5m2': 'e5m2', 'torch.float8_e4m3fn': 'e4m3'}


def dp_chain(a, b):
    """C[M, N] = A[M, K] . B[K, N] by m.mma.dp.async (ISA sec 9C.4), from +0.0,
    ascending K, for FP16/BF16/FP8 operands given in their own dtype.

    The group is four consecutive K for E4M3 x E4M3 and two for every other
    pair; K is zero padded to a whole group (the transfer engine's zero fill).
    E4M3 x E4M3 takes the vectorised exact form (fp8_dot_dp); every other pair
    is evaluated group by group with tools/model/fdp.py (exact rationals,
    RO_48), which is slow and meant for test-sized operands. Independent of the
    compiler, the model and the RTL."""
    torch = _torch()
    fa, fb = _FDP_FORMAT[str(a.dtype)], _FDP_FORMAT[str(b.dtype)]
    if fa == 'e4m3' and fb == 'e4m3':
        return fp8_dot_dp(a.float(), b.t().float())
    import sys
    sys.path.insert(0, str(ROOT / 'tools' / 'model'))
    import fdp
    wa = 8 if fa in ('e4m3', 'e5m2') else 16
    wb = 8 if fb in ('e4m3', 'e5m2') else 16
    ai = a.contiguous().view(torch.uint8 if wa == 8 else torch.int16).to(torch.int64) & ((1 << wa) - 1)
    bi = b.contiguous().view(torch.uint8 if wb == 8 else torch.int16).to(torch.int64) & ((1 << wb) - 1)
    al, bl = ai.tolist(), bi.tolist()
    m, k = len(al), a.shape[1]
    n = b.shape[1]
    pad = (-k) % 2
    out = []
    for i in range(m):
        row = []
        arow = al[i] + [0] * pad
        for j in range(n):
            bcol = [bl[t][j] for t in range(k)] + [0] * pad
            acc = 0
            for k0 in range(0, k + pad, 2):
                g = fdp.group_sum(fa, fb, arow[k0:k0 + 2], bcol[k0:k0 + 2])
                if g[0] == 'fin':
                    g = ('fin', fdp.ro(g[1]), g[2])
                acc = fdp.add_round(g, acc)
            row.append(acc - (1 << 32) if acc & 0x80000000 else acc)
        out.append(row)
    return torch.tensor(out, dtype=torch.int32).view(torch.float32)


def act_quant_inline(x, ieee=False):
    """utils.fp8_act_quant_inline on one K block x (M, k) float32: per-row
    amax / 448, values divided by max(scale, 1e-12), cast to E4M3."""
    torch = _torch()
    a_s = divide(x.abs().amax(dim=1), 448.0, ieee)
    clamped = torch.maximum(a_s, torch.tensor(F32_1E_12))
    q = divide(x, clamped[:, None], ieee).to(torch.float8_e4m3fn)
    return q, a_s


# --------------------------------------------- kernel re-derivations ----
def block_dynamic(A, W, Ws, block_size, out_dtype, ieee=False):
    """w8a8_block_dynamic_fp8_matmul_kernel: C = A @ W.T with activations
    quantised inline per row and per K block, weight scales per block."""
    torch = _torch()
    bn, bk = block_size
    N, K = W.shape
    a = A.reshape(-1, K).float()
    w = W.float()
    ws = Ws.float()
    col_block = torch.arange(N) // bn
    acc = torch.zeros(a.shape[0], N, dtype=torch.float32)
    for kb in range(-(-K // bk)):
        lo, hi = kb * bk, min(K, kb * bk + bk)
        q, a_s = act_quant_inline(a[:, lo:hi], ieee)
        d = fp8_dot(q.float(), w[:, lo:hi])
        acc = acc + (d * a_s[:, None]) * ws[col_block, kb][None, :]
    return acc.to(out_dtype).reshape(*A.shape[:-1], N)


def block_static(A, W, Ws, As, block_size, out_dtype, ieee=False):
    """w8a8_block_static_fp8_matmul_kernel: activations divided by the static
    scalar As and cast inline; per-block weight scales inside the K loop, As
    once after it."""
    torch = _torch()
    bn, bk = block_size
    N, K = W.shape
    a = A.reshape(-1, K).float()
    w = W.float()
    ws = Ws.float()
    s = As.reshape(1).float()
    col_block = torch.arange(N) // bn
    acc = torch.zeros(a.shape[0], N, dtype=torch.float32)
    for kb in range(-(-K // bk)):
        lo, hi = kb * bk, min(K, kb * bk + bk)
        q = divide(a[:, lo:hi], s, ieee).to(torch.float8_e4m3fn).float()
        acc = acc + fp8_dot(q, w[:, lo:hi]) * ws[col_block, kb][None, :]
    return (acc * s).to(out_dtype).reshape(*A.shape[:-1], N)


def act_quant(x, block, ieee=False, clamp=True):
    """_fp8_act_quant_kernel: per `block` contiguous elements, s = amax / 448,
    y = x / max(s, 1e-12) as E4M3 (v1's kernel has no clamp: clamp=False)."""
    torch = _torch()
    xf = x.float().reshape(-1, block)
    s = divide(xf.abs().amax(dim=1), 448.0, ieee)
    den = torch.maximum(s, torch.tensor(F32_1E_12)) if clamp else s
    y = divide(xf, den[:, None], ieee).to(torch.float8_e4m3fn)
    return y.reshape(x.shape), s.reshape(*x.shape[:-1], x.shape[-1] // block)


def tensor_scaled(qA, W, As, Bs, out_dtype):
    """w8a8_tensor_*_fp8_matmul_kernel: one FP8 dot over all of K, accumulated
    in place in ascending K, then * a_s[:, None] * b_s."""
    torch = _torch()
    N, K = W.shape
    d = fp8_dot(qA.reshape(-1, K).float(), W.float())
    out = (d * As.reshape(-1).float()[:, None]) * Bs.reshape(1).float()
    return out.to(out_dtype).reshape(*qA.shape[:-1], N)


def tensor_dynamic(A, W, Bs, out_dtype, ieee=False):
    """v4 w8a8_tensor_dynamic_fp8_matmul: act_quant over whole rows, then the
    tensor-scaled GEMM."""
    K = W.shape[1]
    qA, As = act_quant(A, K, ieee)
    return tensor_scaled(qA, W, As, Bs, out_dtype)


def block_prequantised(qA, W, As, Ws, block_size, out_dtype):
    """v1 w8a8_block_fp8_matmul_kernel: pre-quantised A with per-row, per-K-block
    scales As, per-block weight scales Ws."""
    torch = _torch()
    bn, bk = block_size
    N, K = W.shape
    a = qA.reshape(-1, K).float()
    s = As.reshape(a.shape[0], -1).float()
    w = W.float()
    ws = Ws.float()
    col_block = torch.arange(N) // bn
    acc = torch.zeros(a.shape[0], N, dtype=torch.float32)
    for kb in range(-(-K // bk)):
        lo, hi = kb * bk, min(K, kb * bk + bk)
        d = fp8_dot(a[:, lo:hi], w[:, lo:hi])
        acc = acc + (d * s[:, kb][:, None]) * ws[col_block, kb][None, :]
    return acc.to(out_dtype).reshape(*qA.shape[:-1], N)
