"""S1-07 edge cases for the transposed transfer, through the ORIGINAL TileBench
matmul kernel (`tl.dot(a, b.T)` with host tensor descriptors).

Shapes are chosen for the engine's corner cases rather than for performance:
M, N, K that are not multiples of 16, so the transposed B tile has source
rows past the descriptor's N extent and columns past its K extent, both of
which the engine must write as zeros without a memory access; and BF16 as
well as FP16. Values are multiples of 0.5 in [-1, 1], so every product and
partial sum is exact in the 16-bit output and the comparison is exact.
"""
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev_cases as D                                      # noqa: E402

def _bf16_encode(v):
    import struct
    bits = struct.unpack('<I', struct.pack('<f', v))[0]
    if bits & 0xFFFF:
        raise ValueError(f'{v} is not exact in bf16')
    return bits >> 16


def _bf16_decode(b):
    import struct
    return struct.unpack('<f', struct.pack('<I', (b & 0xFFFF) << 16))[0]


# The R3.7 executor has no bf16; these values are exact, so truncation is exact.
D.E.DTYPES.setdefault('bf16', (2, _bf16_encode, _bf16_decode))

SHAPES = [(16, 16, 24), (16, 24, 16), (24, 16, 16), (16, 40, 40), (40, 24, 48), (16, 16, 8)]


def build(source):
    root = D._root(source)
    fn = D.U.load(root / 'matmul_fp32_fp16_fp8' / 'impl_triton.py')['matmul_kernel']
    out = {}
    for dtype in ('fp16', 'bf16'):
        for M, N, K in SHAPES:
            a = D.values(f'xa/{M}/{N}/{K}', M * K, -1, 1, .5)
            bt = D.values(f'xb/{M}/{N}/{K}', N * K, -1, 1, .5)
            expected = [sum(a[i * K + k] * bt[j * K + k] for k in range(K))
                        for i in range(M) for j in range(N)]
            desc = {'a_desc': {'buffer': 'a', 'shape': [M, K], 'strides': [K, 1], 'block': [16, 16]},
                    'b_desc': {'buffer': 'bt', 'shape': [N, K], 'strides': [K, 1], 'block': [16, 16]},
                    'c_desc': {'buffer': 'c', 'shape': [M, N], 'strides': [N, 1], 'block': [16, 16]}}
            sig, ce, _ = D.U.configuration('matmul_fp32_fp16_fp8', fn)
            for name in ('a', 'bt', 'c'):
                if name in sig:
                    sig[name] = '*' + dtype
            sig.update({k: f'tensordesc<{dtype}[16,16]>' for k in desc})
            case = D.E.Case('matmul_fp32_fp16_fp8', kernel_spec=(fn, sig, ce),
                            buffers={'a': (dtype, a), 'bt': (dtype, bt), 'c': (dtype, [0] * (M * N))},
                            scalars={'M': M, 'N': N, 'K': K}, out='c', expect=expected,
                            program_id=(math.ceil(M / 16) * math.ceil(N / 16) - 1, 0, 0),
                            descriptors=desc)
            out[f'xpose-gemm-{dtype}/{M}x{N}x{K}'] = dict(case=case, operator='matmul_fp32_fp16_fp8',
                                                         shape={'M': M, 'N': N, 'K': K})
    return out
