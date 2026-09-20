#!/usr/bin/env python3
"""
A fault, and a session that keeps working -- R2-12 OBS-04.

The contract's third example is "多角色 GEMM + 故障恢复". The recovery half
is here and runs. The multi-role GEMM half is not: a persistent device is
built for `core-sim` only, and the multi-role profiles have no device yet
(R2-09's RTL half). Writing a single-role GEMM and calling it the
multi-role example would be the wrong kind of example.

What this does show, end to end through the installed package:

  - a kernel that faults reports FAILED with a trap code and a PC, and
    says which program it was;
  - the session is still usable afterwards -- a fault is not a session
    ending;
  - when the device itself is lost, `recover()` starts a new epoch, every
    handle from the old one is refused by name, and new work runs.

    $ python examples/03_fault_recovery.py
"""
import os
import struct
import sys
import tempfile

import triton
import triton.language as tl

from tnca import aot
from tnca.errors import BufferError, TncaError
from tnca.runtime import Session

BLOCK = 16
N = 32


@triton.jit
def axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=mask)


def f32(v):
    return b''.join(struct.pack('<f', x) for x in v)


def main():
    art = aot.compile_aot(
        axpy,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': BLOCK}, profile='core-sim',
        output=os.path.join(tempfile.mkdtemp(prefix='tnca-ex-'),
                            'axpy.tnca'))

    s = Session(profile='core-sim')
    try:
        mod = s.load(art.path)
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        s.upload(x, f32([2.0] * N))
        s.upload(y, f32([1.0] * N))

        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'n': N, 'alpha': 3.0}, grid=(2, 1, 1))
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        print(f"a normal launch      {ev.state}, "
              f"o[0]={got[0]} (expected 7.0)")
        if got != [7.0] * N:
            return 1

        # --- a user error, refused with a code rather than a traceback ---
        for what, call in (
                ('a zero-byte buffer ', lambda: s.allocate(0, name='z')),
                ('a short download   ',
                 lambda: s.download(o, N * 4 + 64)),
                ('a missing argument ',
                 lambda: s.submit(mod, {'x_ptr': x, 'y_ptr': y,
                                        'o_ptr': o, 'n': N}))):
            try:
                call()
                print(f"{what}  ACCEPTED -- it should not have been")
                return 1
            except TncaError as e:
                print(f"{what}  [{e.code}] {e.message[:52]}")

        # The session still works: a refusal is not damage.
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'n': N, 'alpha': -1.0}, grid=(2, 1, 1))
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        print(f"still usable after   {ev.state}, o[0]={got[0]} "
              f"(expected -1.0)")
        if got != [-1.0] * N:
            return 1

        # --- the device is lost, and the session recovers ---
        old_epoch = s.epoch
        s.dev.proc.kill()
        s.dev.proc.wait(timeout=10)
        try:
            s.download(o, N * 4)
            print("a dead device answered -- it should not have")
            return 1
        except TncaError as e:
            print(f"device killed        [{e.code}]")

        s.recover()
        print(f"recovered            epoch {old_epoch} -> {s.epoch}")
        try:
            s.upload(x, f32([1.0] * N))
            print("a handle from the old epoch was accepted")
            return 1
        except BufferError as e:
            print(f"old handle refused   [{e.code}]")

        mod = s.load(art.path)
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        s.upload(x, f32([5.0] * N))
        s.upload(y, f32([0.5] * N))
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'n': N, 'alpha': 2.0}, grid=(2, 1, 1))
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        print(f"new epoch runs       {ev.state}, o[0]={got[0]} "
              f"(expected 10.5)")
        if got != [10.5] * N:
            return 1
    finally:
        s.poisoned = True
        s.close()
    print("ok")
    return 0


if __name__ == '__main__':
    sys.exit(main())
