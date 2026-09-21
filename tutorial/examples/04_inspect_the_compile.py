#!/usr/bin/env python3
"""Seeing what the compiler did: the IR at each stage, and the assembly.

    python tutorial/examples/04_inspect_the_compile.py

Triton dumps its stages when you ask it to, and so does this compiler.
The stages are the input IR, the IR after the upstream passes, the TNCA
IR, the assembly and the machine words -- and the optimised stages are
absent when nothing optimised, which is how you tell.
"""
import os
import sys
import tempfile

import triton
import triton.language as tl

from tnca import aot

PROFILE = 'core-vrf-sync-v1'


@triton.jit
def scale_and_offset(x_ptr, o_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * alpha + 1.0, mask=mask)


def build(opts, where):
    return aot.compile_aot(
        scale_and_offset,
        signature={'x_ptr': '*fp32', 'o_ptr': '*fp32', 'alpha': 'fp32',
                   'n': 'i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, profile=PROFILE,
        options=dict(opts, dump_dir=where),
        output=os.path.join(where, 'k.tnca'))


def main():
    base = tempfile.mkdtemp(prefix='tnca-tutorial-04-')

    plain = build({}, os.path.join(base, 'b0'))
    opt = build({'opt_level': 2}, os.path.join(base, 'o2'))

    print("stages the compiler kept:")
    print(f"  default     {sorted(plain.asm)}")
    print(f"  at O2       {sorted(opt.asm)}")
    print()
    print("  `.opt.ttir` and `.opt.tnir` appear only when something "
          "optimised. A file")
    print("  written anyway would report an optimisation that did not "
          "happen.")
    print()

    for name in sorted(os.listdir(os.path.join(base, 'o2'))):
        d = os.path.join(base, 'o2', name)
        if os.path.isdir(d):
            print(f"  files under {name}/: {sorted(os.listdir(d))}")

    print()
    print("the assembly at O2, which is what actually runs:")
    for line in opt.asm['asm'].splitlines():
        print(f"    {line}")

    print()
    print("what the upstream passes did to the input IR:")
    prov = (opt.manifest.get('optional_metadata') or {}).get('provenance', {})
    for k in sorted(prov):
        if k.startswith('opt_'):
            print(f"    {k:<28s} {prov[k]}")

    print()
    print("and the difference the passes made, in one line:")
    print(f"    {len(plain.asm['hex'].splitlines())} machine word(s) "
          f"at the default, "
          f"{len(opt.asm['hex'].splitlines())} at O2")
    print()
    print("and it still computes the right thing, at both levels:")
    from tnca import device
    from tnca.runtime import Session
    import struct as _s
    device.prepare(PROFILE, verbose=False)
    xs = [float(i) - 8 for i in range(16)]
    want = [x * 2.5 + 1.0 for x in xs]
    good = True
    for label, art in (('default', plain), ('O2', opt)):
        with Session(profile=PROFILE, deadline_cycles=50_000) as s:
            mod = s.load(art.path)
            x = s.allocate(64, name='x')
            o = s.allocate(64, name='o')
            s.upload(x, b''.join(_s.pack('<f', v) for v in xs))
            ev = s.submit(mod, {'x_ptr': x, 'o_ptr': o, 'alpha': 2.5,
                                'n': 16}, grid=(1, 1, 1))
            s.wait(ev)
            blob = s.download(o, 64)
            got = [_s.unpack_from('<f', blob, i * 4)[0] for i in range(16)]
        good &= got == want
        print(f"    {label:<8s} {ev.state}, {ev.device_cycles} cycle(s), "
              f"correct={got == want}")
    print()
    print("    note the `1.0` in the kernel: a float literal is a 32-bit")
    print("    constant, and v.movi only takes sixteen bits, so the "
          "compiler")
    print("    materialises it in a scalar register and broadcasts it. "
          "You can")
    print("    see that in the assembly above as `.li` followed by "
          "`v.splat`.")
    print()
    print("you can get the same dump without writing a script:")
    print(f"    TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=/tmp/dump "
          f"python your_program.py")
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
