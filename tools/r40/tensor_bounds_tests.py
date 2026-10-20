#!/usr/bin/env python3
"""Prove the RTL descriptor row predicate against its original 64-bit contract.

All 160 input bits are unconstrained: signed coordinates, unsigned iterators,
and unsigned extents. No frontend range assumptions or clock cycles are used.
"""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    if not shutil.which('yosys'):
        parser.error('yosys is required for the equivalence proof')
    source = ROOT / 'rtl/core/tnca_tensor.v'
    functions = re.findall(r'\bfunction in_dim;.*?\bendfunction\b',
                           source.read_text(), re.S)
    if len(functions) != 1:
        raise RuntimeError('expected exactly one RTL in_dim function')
    miter = '''module bounds_miter(input [63:0] c, input [31:0] i,
                                  input [63:0] e, output equal);
''' + functions[0] + '''
    wire [63:0] original_sum = c + {32'd0, i};
    wire original_valid = !original_sum[63] && (original_sum < e);
    assign equal = (in_dim(c, i, e) == original_valid);
endmodule
'''
    with tempfile.TemporaryDirectory(prefix='r40-tensor-bounds-') as temporary:
        work = Path(temporary)
        (work / 'miter.v').write_text(miter)
        (work / 'proof.ys').write_text(
            'read_verilog miter.v\nprep -top bounds_miter -flatten\n'
            'sat -verify -prove equal 1 -show-inputs\n')
        result = subprocess.run(['yosys', '-s', 'proof.ys'], cwd=work,
                                capture_output=True, text=True, timeout=120)
    log = ('RTL sha256: ' + hashlib.sha256(source.read_bytes()).hexdigest() +
           '\n' + result.stdout + result.stderr)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(log)
    if result.returncode or 'SUCCESS!' not in result.stdout:
        print(log[-6000:])
        return 1
    print('r40-tensor-bounds PASS: all 160 input bits, exact 64-bit wrap/sign/bounds')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
