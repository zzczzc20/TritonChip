#!/usr/bin/env python3
"""Lower one TTIR file through the Triton plugin's path, outside Inductor.

    TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=/tmp/dump python -m archbench.run ... --compile-only
    python tools/r38/ttir_lower.py /tmp/dump/<kernel>/<kernel>.ttir [--profile core-sim] [--opt 2] [--asm]

(the plugin writes the stages of every kernel, including one that fails to
lower, when TNCA_KERNEL_DUMP=1 -- doc/dumping_ir.md)

The same call the plugin makes (tnca.triton_backend.compiler.make_tnca ->
tnca.aot._lower_ttir), on a module parsed from text: the fast loop for a
compiler change, with Python tracebacks intact.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ttir')
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--opt', type=int, default=2)
    ap.add_argument('--asm', action='store_true', help='print the assembly')
    ap.add_argument('--schedule', type=int, default=None,
                    help='the compile option schedule (R39-01); prints what the scheduler did')
    args = ap.parse_args()
    import triton                                       # noqa: F401  (backends first)
    from triton._C.libtriton import ir
    from tnca import aot, dump as _dump
    from tnca.adapters import r1 as adapter
    from tnca.triton_backend.compiler import ttir_params
    ctx = ir.context()
    ir.load_dialects(ctx)
    module = ir.parse_mlir_module(args.ttir, ctx)
    module.context = ctx
    text = str(module)
    kernel_name, params = ttir_params(text)
    prov = adapter.Provenance()
    target, _pinfo = adapter.target_for(args.profile, prov)
    stages = _dump.Dump(kernel_name)
    options = {'opt_level': args.opt}
    if args.schedule is not None:
        options['schedule'] = args.schedule
    out = aot._lower_ttir(module, ctx, kernel_name, params, target, prov,
                          options, stages, args.opt)
    asm_text = out[2]
    print(f'{kernel_name}: {len(asm_text.splitlines())} asm lines, params {[n for n, _ in params]}')
    for k, v in prov.as_dict().items():
        if k.startswith('schedule'):
            print(f'  {k}: {v}')
    if args.asm:
        print(asm_text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
