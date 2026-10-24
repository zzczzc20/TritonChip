"""Blockwise-scaled `_scaled_mm` (BlockWise1x128 / BlockWise128x128) on tnca.

torch 2.14 lowers a blockwise `aten._scaled_mm` / `aten._scaled_mm_v2` in
Inductor only through `scaled_mm_device_tma_main_loop_scaling`, a persistent
CUDA-TMA template that `use_triton_tma_template` gates
(`config.triton.enable_persistent_tma_matmul` and `can_use_tma`, which asks
`has_triton_tma_device()`: a CUDA device of capability >= 9). It also
dequantises the FP8 tiles before the dot (`a * scale_a_block`, an FP32 dot)
and reads the scales as contiguous whatever their strides. On tnca the
lowering therefore had no choice (NoValidChoicesError), and there is no host
fallback: CPU eager refuses blockwise scales.

This lowering is the K-block form of the same op, as a Triton template:

    for each 128-deep K block kb (ascending):
        acc += tl.dot(a[:, kb], b[kb, :]) * sa[row, kb][:, None] * sb[kb, col][None, :]

The FP8 operands reach `tl.dot` as stored (the FP8 MMA, one binary32 rounding
per product-add from +0.0 in ascending K; an E4M3 x E5M2 pair in 16-deep
steps into the same partial, the compiler's exact FP16 MMA); then `* sa`, `* sb`, `+ acc`, each
rounded once. A bias, an FP8 output's `scale_result` and the conversion to the
output dtype are pointwise nodes Inductor fuses into the template's store, as
the epilogue of upstream's scaled-mm template does (`acc * s + bias` in
binary32, rounded once).

Scale semantics (the scale tile is 128 along K; along M or N it is 1 for
BlockWise1x128 and 128 for BlockWise128x128). The scales' strides are read
from the graph, so any layout is accepted; which scale dimension is the K
block follows the op's convention:

  * `_scaled_mm` (v1): the meta function's shapes -- scale_a (rows, kb),
    scale_b (kb, cols) -- the pairs (1x128, 128x128), (1x128, 1x128) and
    (128x128, 1x128), told apart by shape as upstream's `get_scaling_options`
    does (a tensorwise or rowwise shape keeps upstream's lowering).
  * `_scaled_mm_v2`: the recipe is given. The CUDA kernels' shapes
    (ATen ScaledBlas.cpp): 1x128 scale_a (M, kb), 1x128 scale_b (N, kb),
    128x128 (round_up(kb, 4), M/128 or N/128). The transposed shape is
    accepted too where it is not the same shape; K-block padding beyond
    ceil(K/128) is never read.

Nothing here names a model or a layer; every other `_scaled_mm` keeps its
lowering.
"""
import os

TILE = 128
_MARK = '_tnca_blockwise'

TEMPLATE_SOURCE = r'''
{{def_kernel("A", "B", "SA", "SB")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
{%- if SA_KB_FIRST %}
    sa_k = {{stride("SA", 0)}}
    sa_r = {{stride("SA", 1)}}
{%- else %}
    sa_r = {{stride("SA", 0)}}
    sa_k = {{stride("SA", 1)}}
{%- endif %}
{%- if SB_KB_FIRST %}
    sb_k = {{stride("SB", 0)}}
    sb_c = {{stride("SB", 1)}}
{%- else %}
    sb_c = {{stride("SB", 0)}}
    sb_k = {{stride("SB", 1)}}
{%- endif %}

    pid = tl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = rm % M
    offs_n = rn % N
    offs_k = tl.arange(0, KSUB)
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    sa_ptrs = SA + (offs_m // TILE_A) * sa_r
    sb_ptrs = SB + (offs_n // TILE_B) * sb_c

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        a_s = tl.load(sa_ptrs + kb * sa_k)
        b_s = tl.load(sb_ptrs + kb * sb_k)
{%- if KSUB == BLOCK_K %}
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
{%- else %}
        # Two FP8 encodings: the dot in KSUB-deep steps into one binary32
        # partial, the same product-adds in the same order.
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, BLOCK_K // KSUB):
            k_left = k_remaining - kk * KSUB
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
            part = tl.dot(a, b, part)
            a_ptrs += KSUB * stride_ak
            b_ptrs += KSUB * stride_bk
        acc += part * a_s[:, None] * b_s[None, :]
{%- endif %}

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
'''

_template = None


def template():
    global _template
    if _template is None:
        from torch._inductor.kernel.mm_common import mm_grid
        from torch._inductor.select_algorithm import TritonTemplate
        _template = TritonTemplate(name='tnca_scaled_mm_blockwise', grid=mm_grid,
                                   source=TEMPLATE_SOURCE)
    return _template


def _cdiv(a, b):
    return -(-a // b)


def _static(x):
    from torch._inductor.virtualized import V
    return int(V.graph.sizevars.guard_int(x)) if not isinstance(x, int) else x


def _scale_axes(scale_size, rows, row_tile, k, prefer_kb_first, pad4_kb_first):
    """Which dimension of a 2-D scale is the K block: returns kb_first, or
    None when the shape is not this recipe's. `prefer_kb_first` picks the
    op's own convention when both readings fit; `pad4_kb_first` admits the
    CUDA 128x128 padding (round_up(kb, 4)) on a K-block-first scale."""
    if len(scale_size) != 2:
        return None
    s0, s1 = (_static(s) for s in scale_size)
    rb, kb = _cdiv(rows, row_tile), _cdiv(k, TILE)
    kb_ok = (lambda d, padded: d == kb or (padded and d == _cdiv(kb, 4) * 4))
    first = s1 == rb and kb_ok(s0, pad4_kb_first)          # (kb, rows)
    second = s0 == rb and kb_ok(s1, False)                 # (rows, kb)
    if first and second:
        return prefer_kb_first
    if first:
        return True
    if second:
        return False
    return None


def _realize(x):
    from torch._inductor.select_algorithm import realize_inputs
    x = realize_inputs(x)
    try:
        x.freeze_layout()               # the template reads the strides
    except (AttributeError, NotImplementedError):
        pass
    return x


def lower(mat_a, mat_b, scale_a, scale_b, tile_a, tile_b, kb_first_a, kb_first_b,
          bias, out_dtype, scale_result=None):
    """The blockwise GEMM: template with a binary32 output, then the
    epilogue (bias, an FP8 output's 1/scale_result, the conversion)."""
    import torch
    import torch._inductor.lowering as L
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.select_algorithm import autotune_select_algorithm
    out_dtype = out_dtype or mat_a.get_dtype()
    m, n, k, layout, mat_a, mat_b = mm_args(mat_a, mat_b, out_dtype=torch.float32)
    sa, sb = _realize(scale_a), _realize(scale_b)
    # BLOCK_M: the GEMM tile's M (TNCA_GEMM_TILE, as the mm template). BLOCK_N
    # and BLOCK_K: 128, the Hub block kernel's tile, which the compiler's
    # extended lowering takes with a 128-deep FP8 K block. A narrower N with
    # a 128-deep K does not compile (16: out of vector registers; 32 and 64:
    # mismatched register-block widths), and BLOCK_K is the scale's K block.
    bm, bn = int(os.environ.get('TNCA_GEMM_TILE', '16,16,16').split(',')[0]), TILE
    # E4M3 x E5M2: ISA sec 9C names one input format per MMA, and the
    # compiler builds the exact FP16 substitute for 16-deep K steps only.
    ksub = TILE if mat_a.get_dtype() == mat_b.get_dtype() else 16
    choices = []
    err = template().maybe_append_choice(
        choices, input_nodes=(mat_a, mat_b, sa, sb), layout=layout,
        num_stages=1, num_warps=1,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=TILE, TILE_A=tile_a, TILE_B=tile_b, KSUB=ksub,
        SA_KB_FIRST=bool(kb_first_a), SB_KB_FIRST=bool(kb_first_b))
    if not choices:
        raise RuntimeError(f'tnca blockwise scaled_mm template: {err}')
    out, _ = autotune_select_algorithm('tnca_scaled_mm_blockwise', choices,
                                       [mat_a, mat_b, sa, sb], layout)
    if bias is not None:
        out = L.add(out, L.to_dtype(bias, torch.float32))
    if scale_result is not None and out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out = L.mul(out, L.reciprocal(scale_result))
    return L.to_dtype(out, out_dtype) if out_dtype != torch.float32 else out


def _fp32(x):
    import torch
    return x.get_dtype() == torch.float32


def v1_plan(mat_a, mat_b, scale_a, scale_b):
    """(tile_a, tile_b, kb_first_a, kb_first_b) for a blockwise v1 call,
    else None. Tensorwise/rowwise shapes are left to upstream."""
    if not (_fp32(scale_a) and _fp32(scale_b)):
        return None
    sa, sb = scale_a.get_size(), scale_b.get_size()
    if len(sa) != 2 or len(sb) != 2:
        return None
    M, K = (_static(s) for s in mat_a.get_size())
    N = _static(mat_b.get_size()[1])
    sa_s, sb_s = [_static(s) for s in sa], [_static(s) for s in sb]
    if (sa_s == [M, 1] and sb_s == [1, N]) or (sa_s == [1, 1] and sb_s == [1, 1]):
        return None                                          # rowwise / tensorwise
    kb = _cdiv(K, TILE)
    # The meta function's pairs, in its order.
    if sa_s == [M, kb] and sb_s == [kb, _cdiv(N, TILE)]:
        return (1, TILE, False, True)
    if sa_s == [M, kb] and sb_s == [kb, N]:
        return (1, 1, False, True)
    if sa_s == [_cdiv(M, TILE), kb] and sb_s == [kb, N]:
        return (TILE, 1, False, True)
    return None


def v2_plan(mat_a, mat_b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b):
    from torch.nn.functional import ScalingType
    if (len(scale_a) != 1 or len(scale_b) != 1 or any(swizzle_a) or any(swizzle_b)
            or not (_fp32(scale_a[0]) and _fp32(scale_b[0]))):
        return None
    tiles = {ScalingType.BlockWise1x128.value: 1, ScalingType.BlockWise128x128.value: TILE}
    ra, rb = recipe_a[0], recipe_b[0]
    if ra not in tiles or rb not in tiles:
        return None
    if tiles[ra] == TILE and tiles[rb] == TILE:
        raise NotImplementedError('_scaled_mm_v2 BlockWise128x128 x BlockWise128x128: '
                                  'torch defines no such pairing (ATen ScaledBlas.cpp)')
    M, K = (_static(s) for s in mat_a.get_size())
    N = _static(mat_b.get_size()[1])
    ta, tb = tiles[ra], tiles[rb]
    # CUDA's v2 convention: 1x128 scales (rows, kb); 128x128 (round_up(kb, 4), blocks).
    fa = _scale_axes(scale_a[0].get_size(), M, ta, K, ta == TILE, True)
    fb = _scale_axes(scale_b[0].get_size(), N, tb, K, tb == TILE, True)
    if fa is None or fb is None:
        raise ValueError(f'_scaled_mm_v2 blockwise scale shapes {list(scale_a[0].get_size())}, '
                         f'{list(scale_b[0].get_size())} do not fit M={M} N={N} K={K}')
    return (ta, tb, fa, fb)


def install(device_of, name):
    """Wrap the v1 and v2 lowerings: a blockwise call on `name` takes the
    template above, everything else the lowering that was installed."""
    import torch
    import torch._inductor.lowering as L
    aten = torch.ops.aten

    def on_device(x):
        d = device_of(x)
        return d is not None and d.type == name

    up1 = L.lowerings.get(aten._scaled_mm.default)
    if up1 is not None and not getattr(up1, _MARK, False):
        def scaled_mm(mat_a, mat_b, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                      use_fast_accum=False, layout=None):
            plan = v1_plan(mat_a, mat_b, scale_a, scale_b) if on_device(mat_a) and layout is None else None
            if plan is None:
                return up1(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype,
                           use_fast_accum, layout)
            return lower(mat_a, mat_b, scale_a, scale_b, *plan, bias, out_dtype, scale_result)
        setattr(scaled_mm, _MARK, True)
        scaled_mm._tnca = True
        L.lowerings[aten._scaled_mm.default] = scaled_mm

    up2 = L.lowerings.get(aten._scaled_mm_v2.default)
    if up2 is not None and not getattr(up2, _MARK, False):
        def scaled_mm_v2(mat_a, mat_b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
                         bias=None, out_dtype=None, contraction_dim=None, use_fast_accum=False,
                         layout=None):
            plan = None
            if on_device(mat_a) and layout is None and not contraction_dim:
                plan = v2_plan(mat_a, mat_b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b,
                               swizzle_b)
            if plan is None:
                return up2(mat_a, mat_b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b,
                           swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum, layout)
            return lower(mat_a, mat_b, scale_a[0], scale_b[0], *plan, bias, out_dtype)
        setattr(scaled_mm_v2, _MARK, True)
        scaled_mm_v2._tnca = True
        L.lowerings[aten._scaled_mm_v2.default] = scaled_mm_v2
