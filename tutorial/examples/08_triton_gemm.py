#!/usr/bin/env python3
"""Masked tl.dot GEMM with runtime dimensions, through the public AOT API."""
import argparse
import struct
import tempfile
from pathlib import Path

import triton
import triton.language as tl
from tnca import aot, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'


@triton.jit
def gemm(a, b, c, M, N, K, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK)):
        kk = start * BLOCK + ks
        aa = tl.load(a + rows[:, None] * K + kk[None, :],
                     (rows[:, None] < M) & (kk[None, :] < K), other=0)
        bb = tl.load(b + kk[:, None] * N + cols[None, :],
                     (kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(aa, bb, acc, input_precision='ieee')
    tl.store(c + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def pack(values, dtype):
    fmt = 'e' if dtype == 'fp16' else 'f'
    return struct.pack('<'+fmt*len(values), *values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
    args = parser.parse_args()
    device.prepare(PROFILE, verbose=False)
    with tempfile.TemporaryDirectory(prefix='tnca-gemm-') as directory:
        for level in args.levels:
            for dtype in ('fp16', 'fp32'):
                artifact = aot.compile_aot(
                    gemm, signature={'a': '*'+dtype, 'b': '*'+dtype, 'c': '*fp32',
                                     'M': 'i32', 'N': 'i32', 'K': 'i32', 'BLOCK': 'constexpr'},
                    constexprs={'BLOCK': 16}, profile=PROFILE, options={'opt_level': level},
                    output=str(Path(directory)/f'gemm-{dtype}-O{level}.tnca'))
                with Session(profile=PROFILE, deadline_cycles=500_000) as session:
                    module = session.load(artifact.path)
                    # Reuse the SAME artifact for new dimensions and addresses.
                    # The second launch has row/column tails and two K iterations.
                    for M, N, K in ((16, 16, 16), (17, 19, 32)):
                        a = [((i*7)%13-6)*.25 for i in range(M*K)]
                        b = [((i*3)%11-5)*.25 for i in range(K*N)]
                        expected = [sum(a[m*K+k]*b[k*N+n] for k in range(K))
                                    for m in range(M) for n in range(N)]
                        ab, bb = pack(a, dtype), pack(b, dtype)
                        da, db = session.allocate(len(ab)), session.allocate(len(bb))
                        dc = session.allocate(M*N*4+8)
                        session.upload(da, ab)
                        session.upload(db, bb)
                        session.upload(dc, b'\xcd'*(M*N*4)+b'\xa5'*8)
                        event = session.submit(module, dict(a=da, b=db, c=dc, M=M, N=N, K=K),
                                               grid=(triton.cdiv(M, 16), triton.cdiv(N, 16), 1))
                        session.wait(event)
                        raw = session.download(dc)
                        got = list(struct.unpack('<'+'f'*(M*N), raw[:-8]))
                        assert event.state == 'SUCCEEDED', event.error
                        assert got == expected, f'O{level} {dtype} {M}x{N}x{K}: incorrect product'
                        assert raw[-8:] == b'\xa5'*8, 'output guard overwritten'
                        assert session.download(da) == ab and session.download(db) == bb
                        print(f'O{level} {dtype} {M}x{N}x{K}: PASS; {event.device_cycles} cycles', flush=True)
                        for buffer in (da, db, dc):
                            session.free(buffer)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
