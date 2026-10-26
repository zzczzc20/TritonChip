#!/usr/bin/env python3
"""JIT option identity and the option passed to lowering must agree."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
from triton.backends.compiler import GPUTarget
from tnca.triton_backend.compiler import TNCABackend, TNCAOptions


class Captured(Exception):
    pass


class Module:
    context = None

    def __str__(self):
        return 'module { tt.func public @options_probe(%n: i32) { tt.return } }'


def main():
    backend = TNCABackend(GPUTarget('tnca', 'core-sim', 1))
    switches = dict(schedule='TNCA_SCHEDULE', index_opt='TNCA_INDEX_OPT',
                    rctx_spmd='TNCA_RCTX_SPMD', sparse_tensormaps='TNCA_SPARSE_TENSORMAPS',
                    rctx_roles='TNCA_RCTX_ROLES', bank_alloc='TNCA_BANK_ALLOC',
                    window_reuse='TNCA_WINDOW_REUSE', packed_window_unpack='TNCA_PACKED_WINDOW_UNPACK')
    count = 0
    for name, env in switches.items():
        cases = ((0, 0), (0, 1), (1, 1), (1, 0))
        if name == 'rctx_roles':
            cases += ((2, 0), (2, 1), (2, 2), (0, 2))
        for environment, selected in cases:
            captured = {}
            def lower(*args):
                captured.update(args[6])
                raise Captured()
            with patch.dict(os.environ, dict({env: str(environment)}, TNCA_KERNEL_DUMP='0')):
                options = TNCAOptions(profile='core-sim', **{name: selected})
                with patch('tnca.aot._lower_ttir', side_effect=lower):
                    try:
                        backend.make_tnca(Module(), {}, options)
                    except Captured:
                        pass
                expected = None if environment == selected == 0 else selected
                assert captured.get(name) == expected, (name, captured)
                assert TNCAOptions(**{name: 0}).hash() != TNCAOptions(**{name: 1}).hash()
            count += 1
            print(f'PASS {name}: environment={environment}, selected={selected}, lowering={captured.get(name)}')
    print(f'r40-compile-switch-tests PASS ({count}/{count})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
