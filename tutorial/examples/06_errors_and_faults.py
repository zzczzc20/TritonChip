#!/usr/bin/env python3
"""What goes wrong, and what you are told about it.

    python tutorial/examples/06_errors_and_faults.py

A compiler that accepts nonsense and a runtime that returns plausible
rubbish are both worse than ones that stop. Every case here is a
mistake somebody makes, and what you see is what you get told.
"""
import struct
import sys
import textwrap

import triton
import triton.language as tl

from tnca import aot, device
from tnca.errors import TncaError
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
SIG = {'x_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'}


@triton.jit
def double_it(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x + x, mask=mask)


def show(what, fn):
    """Run something that should fail, and print how it failed."""
    try:
        fn()
        print(f"  {what:<44s} NO ERROR -- which is the problem")
        return False
    except TncaError as e:
        print(f"  {what:<44s} [{getattr(e, 'code', '?')}]")
        for line in textwrap.wrap(str(e).splitlines()[0], 70):
            print(f"  {'':<44s} {line}")
        return True
    except Exception as e:                                 # noqa: BLE001
        print(f"  {what:<44s} {type(e).__name__}: "
              f"{str(e).splitlines()[0][:52]}")
        return True


def main():
    device.prepare(PROFILE, verbose=False)
    ok = True

    print("at compile time:")
    ok &= show("an option this driver does not know",
               lambda: aot.compile_aot(
                   double_it, signature=SIG, constexprs={'BLOCK': 16},
                   profile=PROFILE, options={'optimise_level': 2},
                   output='/tmp/tnca-tutorial-06a.tnca'))
    ok &= show("a profile that does not exist",
               lambda: aot.compile_aot(
                   double_it, signature=SIG, constexprs={'BLOCK': 16},
                   profile='core-imaginary',
                   output='/tmp/tnca-tutorial-06b.tnca'))
    ok &= show("a block wider than one element per lane",
               lambda: aot.compile_aot(
                   double_it, signature=SIG, constexprs={'BLOCK': 256},
                   profile=PROFILE,
                   output='/tmp/tnca-tutorial-06c.tnca'))

    art = aot.compile_aot(double_it, signature=SIG,
                          constexprs={'BLOCK': 16}, profile=PROFILE,
                          options={'opt_level': 2},
                          output='/tmp/tnca-tutorial-06.tnca')
    ref = aot.compile_aot(double_it, signature=SIG,
                          constexprs={'BLOCK': 16}, profile='core-sim',
                          options={'opt_level': 2},
                          output='/tmp/tnca-tutorial-06-ref.tnca')

    print()
    print("at load and launch time:")
    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        before = s.dev.seq
        ok &= show("an artifact built for another profile",
                   lambda: s.load(ref.path))
        print(f"  {'':<44s} and {s.dev.seq - before} device request(s) "
              f"were made")
        ok &= (s.dev.seq - before) == 0

        mod = s.load(art.path)
        x = s.allocate(64, name='x')
        o = s.allocate(64, name='o')
        ok &= show("an argument the kernel does not take",
                   lambda: s.submit(mod, {'x_ptr': x, 'o_ptr': o, 'n': 16,
                                          'nonsense': 1}))
        ok &= show("a missing argument",
                   lambda: s.submit(mod, {'x_ptr': x, 'n': 16}))

    print()
    print("a kernel that really faults on the device:")

    # The simplest fault a legitimate compile can produce: an assertion
    # on a value that is not what it claims. Note that `n` being huge in
    # the kernel above does NOT fault -- it only widens a mask, and all
    # sixteen lanes were already inside the buffer. A fault has to come
    # from the program doing something the machine refuses.
    def trapping(b, target):
        v = b.const(1)
        b.assert_eq(v, 0xDEAD, 7)          # 1 is not 0xDEAD
        b.exit()

    trap = aot.compile_aot(trapping, kernel_name='trapper', signature={},
                           source_kind='tnca-ir', profile=PROFILE,
                           output='/tmp/tnca-tutorial-06-trap.tnca')
    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        mod = s.load(trap.path)
        ev = s.submit(mod, {})
        try:
            s.wait(ev)
        except TncaError as e:
            print(f"  wait() raised [{getattr(e, 'code', '?')}]: "
                  f"{str(e).splitlines()[0][:62]}")
        print(f"  the event ended {ev.state}, ok={ev.ok}")
        if ev.error is not None:
            print(f"  and says why: [{getattr(ev.error, 'code', '?')}] "
                  f"{str(ev.error).splitlines()[0][:62]}")
        ok &= not ev.ok and ev.state == 'FAILED'

        # A drainable fault leaves the session usable. One that does not
        # says so rather than letting you keep going on a broken device.
        print(f"  the session afterwards: "
              f"{'poisoned' if s.poisoned else 'still usable'}")

    print()
    print("the point: every one of these is a named code you can branch "
          "on, not a")
    print("traceback you have to read. A wrong-profile artifact is "
          "refused after zero")
    print("device requests, so nothing half-ran.")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
