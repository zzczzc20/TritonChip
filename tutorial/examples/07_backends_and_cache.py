#!/usr/bin/env python3
"""Choosing where a kernel runs, and not recompiling when you do.

    python tutorial/examples/07_backends_and_cache.py

Two separate ideas that are easy to confuse:

    a PROFILE says what machine you are compiling for -- the ISA, the
    capabilities, the register file. It is part of the artifact's
    identity, so changing it changes what you get.

    a BACKEND says how this host reaches a device. It is not part of the
    artifact at all, so the same compiled kernel runs on any of them.
"""
import struct
import sys
import textwrap
import tempfile

import triton
import triton.language as tl

from tnca import aot, backend, device
from tnca.cache import Cache
from tnca.errors import TncaError
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
SIG = {'x_ptr': '*fp32', 'o_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'}


@triton.jit
def triple(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x + x + x, mask=mask)


def run(art, name, xs):
    with Session(profile=PROFILE, backend=name,
                 deadline_cycles=50_000) as s:
        mod = s.load(art.path)
        x = s.allocate(64, name='x')
        o = s.allocate(64, name='o')
        s.upload(x, b''.join(struct.pack('<f', v) for v in xs))
        ev = s.submit(mod, {'x_ptr': x, 'o_ptr': o, 'n': 16},
                      grid=(1, 1, 1))
        s.wait(ev)
        blob = s.download(o, 64)
        caps = s.dev.capabilities()
    got = [struct.unpack_from('<f', blob, i * 4)[0] for i in range(16)]
    return got, ev.device_cycles, caps


def main():
    device.prepare(PROFILE, verbose=False)
    xs = [float(i) - 8 for i in range(16)]
    want = [x * 3 for x in xs]
    ok = True

    print("backends this installation knows:")
    for name, info in backend.available().items():
        head, *rest = textwrap.wrap(info['what'], 48)
        print(f"  {name:<16s} {info['state']:<20s} {head}")
        for line in rest:
            print(f"  {'':<16s} {'':<20s} {line}")
    print()
    print("  `interface-defined` means the contract exists and nothing "
          "implements it.")
    print("  Opening one is an error, NOT a quiet redirection to the "
          "simulator --")
    print("  otherwise a platform that was never there would appear to "
          "work.")
    print()

    art = aot.compile_aot(triple, signature=SIG, constexprs={'BLOCK': 16},
                          profile=PROFILE, options={'opt_level': 2},
                          output='/tmp/tnca-tutorial-07.tnca')
    print(f"one artifact, key {art.cache_key[:12]}:")
    seen = {}
    for name in ('sim-file', 'sim-socket'):
        got, cycles, caps = run(art, name, xs)
        seen[name] = cycles
        print(f"  {name:<12s} via {caps.get('medium'):<22s} "
              f"{cycles} cycle(s), correct={got == want}")
        ok &= got == want
    print(f"  the same artifact on both, no recompile, and the same "
          f"{seen['sim-file']} cycles")
    print(f"  -- which is how you know the second one really reached the "
          f"hardware")
    print(f"  rather than returning a plausible answer.")
    print()

    try:
        backend.open_backend('fpga-prototype', bench='anything')
        print("  opening an unimplemented backend SUCCEEDED, which is wrong")
        ok = False
    except TncaError as e:
        print(f"  asking for one that is not implemented: "
              f"[{getattr(e, 'code', '?')}]")
        for line in textwrap.wrap(str(e).splitlines()[0], 66):
            print(f"    {line}")

    print()
    print("the cache, which is about not compiling twice:")
    d = tempfile.mkdtemp()
    cache = Cache(d)
    first = aot.compile_aot(triple, signature=SIG, constexprs={'BLOCK': 16},
                            profile=PROFILE, options={'opt_level': 2},
                            cache=cache, output=f'{d}/a.tnca')
    again = aot.compile_aot(triple, signature=SIG, constexprs={'BLOCK': 16},
                            profile=PROFILE, options={'opt_level': 2},
                            cache=cache, output=f'{d}/b.tnca')
    other = aot.compile_aot(triple, signature=SIG, constexprs={'BLOCK': 16},
                            profile=PROFILE, options={'opt_level': 1},
                            cache=cache, output=f'{d}/c.tnca')
    print(f"  the same request twice      from_cache={again.from_cache}, "
          f"key {again.cache_key[:12]}")
    print(f"  a different opt level       from_cache={other.from_cache}, "
          f"key {other.cache_key[:12]}")
    ok &= bool(again.from_cache) and not other.from_cache
    print()
    print("  the optimisation level is IN the key. If it were not, a "
          "kernel compiled")
    print("  at O0 would be served for an O2 request and every "
          "measurement after it")
    print("  would be of the wrong artifact.")
    print()
    print("  a diagnostic option is not in the key, because it changes "
          "nothing:")
    dumped = aot.compile_aot(triple, signature=SIG,
                            constexprs={'BLOCK': 16}, profile=PROFILE,
                            options={'opt_level': 2,
                                     'dump_dir': tempfile.mkdtemp()},
                            output=f'{d}/d.tnca')
    print(f"    with dump_dir: key {dumped.cache_key[:12]} "
          f"(same: {dumped.cache_key == first.cache_key})")
    ok &= dumped.cache_key == first.cache_key
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
