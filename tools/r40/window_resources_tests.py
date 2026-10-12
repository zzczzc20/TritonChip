#!/usr/bin/env python3
"""Audit window register ranges in preserved machine code; no timing claim.

This reads original artifacts without modifying their manifests. It exposes
implicit DMA/MMA registers omitted by older explicit-operand counters.
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
from tnca import artifact
from tnca.aot import measure_resources
from tnca.adapters.r1 import target_for
from tnca_dis import disassemble
from tnca_isa import decode_fields, MAJ


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    checked = 0
    for path in sorted(args.directory.glob('*.tnca')):
        manifest, segments = artifact.load(str(path))
        prov = manifest['optional_metadata']['provenance']
        layout = prov.get('window_lifetime')
        if layout is None:
            continue
        code = segments['text']
        words = [int.from_bytes(code[i:i + 8], 'little') for i in range(0, len(code), 8)]
        assembly = '\n'.join(disassemble(word) for word in words)
        assert '.dword' not in assembly, 'unknown instruction in window artifact'
        definitions = [decode_fields(word) for word in words
                       if decode_fields(word)['major'] == MAJ['TDESC']
                       and decode_fields(word)['func'] == 0x10]
        registers = {r for f in definitions for r in range(f['b'], f['b'] + f['c'])}
        assert len(registers) == layout['reserved_registers'], (path, layout)
        target, info = target_for(manifest['target']['profile'])
        assert manifest['target']['config_hash'] == info['config_hash']
        assert all(0 <= r < target.vregs_per_rctx for r in registers)
        measured = measure_resources(assembly, target, code=code, include_windows=True)
        assert len(registers) <= measured['vector_regs_used'] <= target.vregs_per_rctx
        print(f'PASS {path.name}: physical windows={len(registers)}, '
              f'all vector registers={measured["vector_regs_used"]}, '
              f'text_sha256={hashlib.sha256(code).hexdigest()}', flush=True)
        checked += 1
    assert checked, 'no window-reuse artifact found'
    print(f'r40-window-resources-tests PASS ({checked}/{checked}); '
          'original artifacts unchanged, counters audited from machine code', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
