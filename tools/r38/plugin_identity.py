#!/usr/bin/env python3
"""P1+ T1: one compile path. The Triton backend and compile_aot, same code.

For every R3.7 case, compile the ORIGINAL JIT function twice:

  * through Triton's own `triton.compile`, which runs the tnca backend plugin
    (tnca.triton_backend.compiler), with no argument specialisation -- the
    same TTIR compile_aot sees;
  * through tnca.aot.compile_aot.

and require the `.text` code segments to be byte-identical. The manifests
legitimately differ in provenance (which entry point, a TTIR hash rather than
a source hash) and in declared aliases (the plugin declares Triton's pointer
aliasing for LSU-only programs), so they are not compared.

When Triton's JIT does specialise an argument (an integer equal to 1 becomes a
constant, divisibility attributes) the TTIR is a different program and the
code may differ; tools/r38/triton_native_run.py covers that path numerically.

    ~/src/torchenv/bin/python tools/r38/plugin_identity.py [--level 2]
"""
import argparse
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402
import triton                                               # noqa: E402
from triton.backends.compiler import GPUTarget              # noqa: E402
from triton.compiler.compiler import ASTSource              # noqa: E402
from tnca import aot, artifact                              # noqa: E402
from tnca.triton_backend.compiler import _schedule, _index_opt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--only', nargs='+')
    args = ap.parse_args()
    import os
    os.environ['TNCA_OPT_LEVEL'] = str(args.level)
    cases = H.cases()
    diffs, n = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, case in cases.items():
            if args.only and case.kernel not in args.only and name not in args.only:
                continue
            fn, sig, ce = case.kernel_spec
            a_path = Path(tmp) / 'aot.tnca'
            # The plugin's own choices for the R39 switches (TNCA_SCHEDULE,
            # TNCA_INDEX_OPT), passed explicitly: the AOT entry point
            # defaults them off.
            aot.compile_aot(fn, signature=sig, constexprs=ce, profile='core-sim',
                            options={'opt_level': args.level, 'schedule': _schedule(),
                                     'index_opt': _index_opt()},
                            output=str(a_path))
            src = ASTSource(fn, signature=sig, constexprs=ce)
            k = triton.compile(src, target=GPUTarget('tnca', 0, 1))
            p_path = Path(tmp) / 'plugin.tnca'
            p_path.write_bytes(k.asm['tnca'])
            _, a_seg = artifact.load(str(a_path))
            _, p_seg = artifact.load(str(p_path))
            n += 1
            if a_seg.get('text') != p_seg.get('text'):
                diffs.append(name)
                print('DIFF', name, flush=True)
    print(f'plugin identity O{args.level}: {n - len(diffs)}/{n} code segments byte-identical; '
          f'{"PASS" if not diffs else "FAIL"}')
    return 1 if diffs else 0


if __name__ == '__main__':
    sys.exit(main())
