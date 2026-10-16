#!/usr/bin/env python3
"""The ring-buffer count must survive AOT/JIT validation and cache identity."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm')]
from triton.backends.compiler import GPUTarget
from tnca.aot import check_options
from tnca.triton_backend.compiler import TNCAOptions, TNCABackend
from compile_switch_tests import Captured, Module


def main():
    backend = TNCABackend(GPUTarget('tnca', 'core-sim', 1))
    count = 0
    for environment in (1, 2, 3):
        for selected in (1, 2, 3):
            captured = {}
            def lower(*args):
                captured.update(args[6])
                raise Captured()
            with patch.dict(os.environ, TNCA_RCTX_STAGES=str(environment), TNCA_KERNEL_DUMP='0'):
                options = backend.parse_options({'rctx_stages': selected})
                assert check_options({'rctx_stages': selected})['rctx_stages'] == selected
                expected = None if environment == selected == 2 else selected
                with patch('tnca.aot._lower_ttir', side_effect=lower):
                    try:
                        backend.make_tnca(Module(), {}, options)
                    except Captured:
                        pass
                assert captured.get('rctx_stages') == expected, captured
                assert TNCAOptions(rctx_stages=2).hash() != TNCAOptions(rctx_stages=3).hash()
            count += 1
    with patch.dict(os.environ, TNCA_RCTX_STAGES='2'):
        assert 'rctx_stages' not in check_options({}), 'default changes frozen provenance'
        for bad in (0, 4, -1, 2.5, True, '3'):
            for validate in (lambda: check_options({'rctx_stages': bad}),
                             lambda: backend.parse_options({'rctx_stages': bad})):
                try:
                    validate()
                except ValueError:
                    count += 1
                else:
                    raise AssertionError(f'invalid stages accepted: {bad!r}')
    for value in (1, 2, 3):
        with patch.dict(os.environ, {'TNCA_PIPELINE': str(value)}, clear=True):
            resolved = check_options({})
            assert resolved.get('rctx_stages', 2) == value
            assert TNCAOptions().rctx_stages == value
            assert len({TNCAOptions(rctx_stages=n).hash() for n in (1, 2, 3)}) == 3
        count += 1
    with patch.dict(os.environ, {'TNCA_PIPELINE': '1', 'TNCA_RCTX_STAGES': '3'}, clear=True):
        for validate in (lambda: check_options({}), lambda: TNCAOptions()):
            try:
                validate()
            except ValueError as error:
                assert 'conflicts' in str(error)
                count += 1
            else:
                raise AssertionError('conflicting stage controls accepted')
    # A resolved explicit JIT/AOT option overrides both environment defaults.
    for selected in (1, 2, 3):
        captured = {}
        def lower(*args):
            captured.update(args[6])
            raise Captured()
        with patch.dict(os.environ, {'TNCA_PIPELINE': '1', 'TNCA_RCTX_STAGES': '3'}, clear=True):
            options = backend.parse_options({'rctx_stages': selected})
            assert check_options({'rctx_stages': selected})['rctx_stages'] == selected
            with patch('tnca.aot._lower_ttir', side_effect=lower):
                try:
                    backend.make_tnca(Module(), {}, options)
                except Captured:
                    pass
            assert captured.get('rctx_stages') == selected, captured
        count += 1
    print(f'r40-pipeline-stage-options PASS ({count}/{count})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
