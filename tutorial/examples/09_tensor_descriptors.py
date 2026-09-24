#!/usr/bin/env python3
"""Host tensor descriptors, runtime K loop and tl.dot via AOT + Session."""
import struct
import tempfile
from pathlib import Path
import triton
import triton.language as tl
from tnca import aot, device
from tnca.runtime import Session, TensorDescriptor

PROFILE = 'core-vrf-sync-v1'


@triton.jit
def descriptor_gemm(a, bt, c, K, OUTPUT_DTYPE: tl.constexpr):
    m = tl.program_id(0)*16
    n = tl.program_id(1)*16
    acc = tl.full((16, 16), 0, tl.float32)
    for k in range(0, tl.cdiv(K, 16)):
        aa = a.load([m, k*16])
        bb = bt.load([n, k*16])
        acc = tl.dot(aa, bb.T, acc, input_precision='ieee')
    c.store([m, n], acc.to(OUTPUT_DTYPE))


def main():
    device.prepare(PROFILE, verbose=False)
    M, N, K = 17, 16, 32
    a = [((i*7)%11-5)*.25 for i in range(M*K)]
    bt = [((i*3)%9-4)*.25 for i in range(N*K)]
    expected = [sum(a[m*K+k]*bt[n*K+k] for k in range(K))
                for m in range(M) for n in range(N)]
    with tempfile.TemporaryDirectory(prefix='tnca-descriptors-') as directory:
        for level in (0, 1, 2):
            artifact = aot.compile_aot(
                descriptor_gemm,
                signature={'a': 'tensordesc<fp16[16,16]>', 'bt': 'tensordesc<fp16[16,16]>',
                           'c': 'tensordesc<fp32[16,16]>', 'K': 'i32', 'OUTPUT_DTYPE': 'constexpr'},
                constexprs={'OUTPUT_DTYPE': tl.float32}, profile=PROFILE,
                options={'opt_level': level}, output=str(Path(directory)/f'descriptor-O{level}.tnca'))
            with Session(profile=PROFILE, deadline_cycles=500_000) as session:
                module = session.load(artifact.path)
                da, db, dc = (session.allocate(size) for size in (M*K*2, N*K*2, M*N*4+8))
                session.upload(da, struct.pack('<'+'e'*len(a), *a))
                session.upload(db, struct.pack('<'+'e'*len(bt), *bt))
                session.upload(dc, b'\xcd'*(M*N*4)+b'\xa5'*8)
                args = {'a': TensorDescriptor(da, (M, K), (K, 1), (16, 16), 'fp16'),
                        'bt': TensorDescriptor(db, (N, K), (K, 1), (16, 16), 'fp16'),
                        'c': TensorDescriptor(dc, (M, N), (N, 1), (16, 16), 'fp32'), 'K': K}
                event = session.submit(module, args, grid=(2, 1, 1))
                session.wait(event)
                raw = session.download(dc)
                got = list(struct.unpack('<'+'f'*(M*N), raw[:-8]))
                assert event.state == 'SUCCEEDED', event.error
                assert got == expected, f'O{level}: incorrect descriptor product'
                assert raw[-8:] == b'\xa5'*8, 'output guard overwritten'
                print(f'O{level} descriptor GEMM {M}x{N}x{K}: PASS; {event.device_cycles} cycles', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
