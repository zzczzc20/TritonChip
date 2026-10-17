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
    for environment, selected in ((0, 0), (0, 1), (1, 1), (1, 0)):
        captured = {}

        def lower(*args):
            captured.update(args[6])
            raise Captured()

        with patch.dict(os.environ, TNCA_WINDOW_REUSE=str(environment), TNCA_KERNEL_DUMP='0'):
            options = TNCAOptions(profile='core-sim', window_reuse=selected)
            with patch('tnca.aot._lower_ttir', side_effect=lower):
                try:
                    backend.make_tnca(Module(), {}, options)
                except Captured:
                    pass
            expected = None if environment == selected == 0 else selected
            assert captured.get('window_reuse') == expected, captured
            assert TNCAOptions(window_reuse=0).hash() != TNCAOptions(window_reuse=1).hash()
        print(f'PASS environment={environment}, selected={selected}, '
              f'lowering={captured.get("window_reuse", "default off")}')
    print('r40-window-options-tests PASS (4/4)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
