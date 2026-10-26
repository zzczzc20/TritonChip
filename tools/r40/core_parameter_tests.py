#!/usr/bin/env python3
"""Check elaborated core parameters through the mesh/tile wrappers.

Elaboration/resource plumbing only; not native multicore execution evidence.
"""
from pathlib import Path
import re
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
SWITCHES = dict(USE_BANKED_VRF=1, VRF_BYPASS=1, FE_NOBUBBLE=1,
    VRF_PREFETCH=1, RCTX_SWITCH=2, MTX_ABUF=1, LSU_INFLIGHT=2,
    FPU_OVERLAP=1, RSLICE=1, RPAIR=1, HAS_SYS_ATOMICS=0, HAS_TENSOR_XPOSE=1,
    HAS_SFU=1, NUM_RCTX=4, RCTX_W=2, PHYS_VREGS=256, PHYS_W=8)


def cores(path):
    found, current = [], None
    with path.open() as source:
        for line in source:
            if '.scope ' in line:
                current = {} if '.scope module, "u_core" "tnca_core"' in line else None
                if current is not None:
                    found.append(current)
            elif current is not None:
                match = re.search(r'\.param/l "([^"]+)".*[+]C4<([01]+)>', line)
                if match:
                    current[match[1]] = int(match[2], 2)
    return found


def main():
    libraries = ['common', 'core', 'gen', 'nuca', 'noc', 'tile', 'fpga']
    with tempfile.TemporaryDirectory(prefix='r40-core-params-') as tmp:
        for n, nx, ny, selected in [(4, 2, 2, False), (1, 1, 1, True),
                                   (2, 2, 1, True), (3, 3, 1, True),
                                   (4, 2, 2, True), (8, 4, 2, True)]:
            expected = SWITCHES if selected else dict((k, 0) for k in SWITCHES
                if k not in ('NUM_RCTX', 'RCTX_W', 'PHYS_VREGS', 'PHYS_W', 'LSU_INFLIGHT'))
            if not selected:
                expected.update(NUM_RCTX=2, RCTX_W=1, PHYS_VREGS=128, PHYS_W=7, LSU_INFLIGHT=1)
            out = Path(tmp) / f'{n}_{int(selected)}.vvp'
            cmd = ['iverilog', '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc', '-s', 'tnca_top',
                   f'-Ptnca_top.NX={nx}', f'-Ptnca_top.NY={ny}', '-o', str(out)]
            for directory in libraries:
                cmd.extend(['-y', 'rtl/' + directory])
            if selected:
                cmd.extend(f'-Ptnca_top.{k}={v}' for k, v in expected.items())
            cmd.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / 'rtl/common').glob('*.v')))
            cmd.append('rtl/tnca_top.v')
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            actual = cores(out)
            assert len(actual) == n, (n, len(actual))
            for core in actual:
                assert {k: core.get(k) for k in expected} == expected, (core, expected)
            print(f'PASS {n} elaborated cores: {"v5" if selected else "legacy"} parameters reach every core', flush=True)
    print('r40-core-parameter-tests PASS (6/6)', flush=True)


if __name__ == '__main__':
    main()
