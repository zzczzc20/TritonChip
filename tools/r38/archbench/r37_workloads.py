"""Semantic demand of the R3.7 numerical cases (tools/r37/upstream_cases.py).

Shapes are copied from the cases, not read from compiled programs. Where a
term's minimum is uncertain the term is OMITTED rather than guessed:
dropping a term keeps a lower bound valid (it only loosens it), while an
over-counted term makes it wrong. Padded convolution taps are therefore
counted as valid taps only.

Operators without a v0 bound (sorting, selection, histogram, attention,
multi-stage pipelines whose semantic instance spans several launches)
are listed as unavailable with the reason.
"""
import itertools
import math

from . import bounds as B


def _valid_taps(spatial, kernel, pad):
    out = tuple(n + 2 * pad - k + 1 for n, k in zip(spatial, kernel))
    total = 0
    for loc in itertools.product(*(range(n) for n in out)):
        for tap in itertools.product(*(range(k) for k in kernel)):
            if all(0 <= i + j - pad < n for i, j, n in zip(loc, tap, spatial)):
                total += 1
    return total, math.prod(out)


def build():
    w = {}

    def put(x):
        w[x.id] = x

    n = 13
    for op, fp, sfu in (('mul2', 1, 0), ('relu', 1, 0), ('leaky_relu', 1, 0), ('sigmoid', 2, 1)):
        put(B.elementwise(op, op, n, inputs=1, outputs=1, fp_per_elem=fp, sfu_per_elem=sfu))
    put(B.elementwise('quantize_global', 'quantize_global', n, inputs=1, outputs=0, fp_per_elem=0))
    w['quantize_global'].write_bytes = n * 2
    put(B.elementwise('vector_add', 'vector_add', n, inputs=2, outputs=1, fp_per_elem=1))
    put(B.elementwise('swiglu', 'swiglu', n, inputs=2, outputs=1, fp_per_elem=3, sfu_per_elem=1))
    put(B.elementwise('fused_activation', 'fused_activation', n, inputs=3, outputs=1,
                      fp_per_elem=3, sfu_per_elem=1))
    d = B.elementwise('dropout', 'dropout', n, inputs=1, outputs=1, fp_per_elem=1)
    d.read_bytes += n * 4                          # the i32 keep mask
    put(d)

    for n in (7, 37):
        put(B.row_reduction(f'softmax/n{n}', 'softmax', 1, n, writes_per_row=0, writes_per_elem=1,
                            fp_per_elem=3, sfu_per_elem=1))
        put(B.row_reduction(f'layernorm/n{n}', 'layernorm', 1, n, reads_per_elem=3, writes_per_row=0,
                            writes_per_elem=1, fp_per_elem=5))
        put(B.row_reduction(f'rmsnorm/n{n}', 'rmsnorm', 1, n, reads_per_elem=2, writes_per_row=0,
                            writes_per_elem=1, fp_per_elem=3))
        put(B.row_reduction(f'l2_norm/n{n}', 'l2_norm', 1, n, writes_per_row=0, writes_per_elem=1,
                            fp_per_elem=2))
        put(B.row_reduction(f'kl_divergence/n{n}', 'kl_divergence', 1, n, reads_per_elem=2,
                            fp_per_elem=2))
        a = B.row_reduction(f'argmax/n{n}', 'argmax', 1, n, writes_per_row=0, fp_per_elem=1)
        a.write_bytes = 8                           # one i64 index
        put(a)
    put(B.row_reduction('mean_reduction', 'mean_reduction', 1, 16, fp_per_elem=1))
    ce = B.row_reduction('cross_entropy', 'cross_entropy', 1, 16, fp_per_elem=2, sfu_per_elem=1)
    ce.read_bytes += 4                              # the i32 target
    put(ce)

    put(B.copy('matrix_copy', 'matrix_copy', 16))
    put(B.copy('matrix_transpose', 'matrix_transpose', 256))
    put(B.copy('reverse_array', 'reverse_array', 13))
    put(B.copy('interleave', 'interleave', 14))

    for dt in ('fp16', 'fp32', 'e4m3', 'e5m2'):
        put(B.gemm(f'matmul_fp32_fp16_fp8/{dt}', 'matmul_fp32_fp16_fp8', 16, 16, 32,
                   in_dtype=dt, out_dtype=dt))
    put(B.gemm('batched_matmul', 'batched_matmul', 16, 16, 16, batch=2,
               in_dtype='fp16', out_dtype='fp16'))
    # INT8: the MACs may run on integer or FP lanes, and v0 does not
    # establish their combined capacity -- so the compute term is omitted
    # and the bound is the bytes alone. Valid, and loose.
    i8 = B.Workload('matmul_int8', 'matmul_int8', 'gemm',
                    read_bytes=16 * 64 + 16 * 16, write_bytes=256 * 4,
                    note='bytes-only bound: INT8 MAC capacity not established in v0')
    put(i8)

    for rank, spatial, kernel, pad in [(1, (7,), (3,), 1), (2, (3, 4), (3, 3), 1),
                                       (3, (3, 3, 3), (2, 2, 2), 0)]:
        taps, out_pts = _valid_taps(spatial, kernel, pad)
        c = B.conv(f'{rank}d_conv', f'{rank}d_conv', out_pts, 1, math.prod(spatial),
                   2 * math.prod(kernel), out_channels=2)
        c.fp_ops = c.macs = 2 * taps                 # valid taps only, both output channels
        put(c)
    taps, _ = _valid_taps((3, 13), (3, 3), 1)
    gb = B.Workload('gaussian_blur', 'gaussian_blur', 'conv', fp_ops=taps, macs=taps,
                    read_bytes=(39 + 9) * 4, write_bytes=39 * 4)
    put(gb)
    put(B.Workload('jacobi_stencil_2d', 'jacobi_stencil_2d', 'stencil',
                   read_bytes=39 * 4, write_bytes=39 * 4,
                   note='bytes-only: interior arithmetic is below the memory term'))
    put(B.Workload('2d_max_pooling', '2d_max_pooling', 'stencil',
                   read_bytes=39 * 4, write_bytes=39 * 4, note='bytes-only'))

    for op, why in (
        ('bitonic_sort', 'sorting: algorithm-independent comparison bound not established in v0'),
        ('radix_sort', 'sorting: multi-stage instance'),
        ('top_k_selection', 'selection: no v0 bound'),
        ('moe_topk_gating', 'selection: no v0 bound'),
        ('histogramming', 'atomic conflicts depend on the distribution; no v0 bound'),
        ('flash_attention', 'attention IO bound (section 4.4) not implemented in v0'),
        ('block_sparse_attention', 'attention: no v0 bound'),
        ('flash_decode', 'attention: no v0 bound'),
        ('linear_self_attention', 'multi-stage instance'),
        ('batch_normalization', 'multi-stage instance'),
        ('streamk_matmul', 'multi-program instance; grid is serial on this runtime'),
        ('rope', 'no v0 bound'), ('destindex', 'no v0 bound'),
        ('dequantize_rowwise', 'no v0 bound'), ('weight_dequant', 'no v0 bound'),
    ):
        put(B.unavailable(op, op, 'unbounded', why))
    return w
