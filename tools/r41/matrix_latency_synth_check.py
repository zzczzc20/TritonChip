#!/usr/bin/env python3
"""Check Vivado's synthesized MAC_LAT for all 32 matrix pipeline cut masks.

This uses the declaration from rtl/core/tnca_matrix.v and checks the actual
synthesized output constants; simulation alone does not catch the Vivado
one-bit expression-width behavior that broke the run-2 image.
"""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vivado', default='vivado')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source = (ROOT/'rtl/core/tnca_matrix.v').read_text()
    matches = re.findall(r'^\s*(localparam(?:\s+integer)?\s+MAC_LAT\s*=\s*[^;]+;)', source, re.M)
    if len(matches) != 1:
        raise ValueError('expected exactly one matrix MAC_LAT declaration')
    args.out.mkdir(parents=True, exist_ok=False)
    module = ('module latency_case #(parameter [4:0] MAC_CUTS=0) (output [2:0] value);\n'
              + matches[0] + '\nassign value=MAC_LAT;\nendmodule\n'
              + 'module latency_probe(output [95:0] result);\n'
              + ''.join(f'latency_case #(.MAC_CUTS(5\'d{i})) c{i}(.value(result[{i*3}+:3]));\n'
                        for i in range(32)) + 'endmodule\n')
    (args.out/'probe.v').write_text(module)
    (args.out/'probe.tcl').write_text('read_verilog probe.v\n'
        'synth_design -top latency_probe -part xcu280-fsvh2892-2L-e\n'
        'write_verilog -force netlist.v\nexit\n')
    with (args.out/'vivado.stdout').open('w') as log:
        subprocess.run([args.vivado, '-mode', 'batch', '-source', 'probe.tcl'], cwd=args.out,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    netlist = (args.out/'netlist.v').read_text()
    values = {}
    for index, value in re.findall(r'OBUF\s+\\result_OBUF\[(\d+)\]_inst\s*\n\s*\(\.I\(\\<const([01])>', netlist):
        values[int(index)] = int(value)
    if len(values) != 96:
        raise ValueError(f'expected 96 synthesized output bits, found {len(values)}')
    for mask in range(32):
        actual = sum(values[mask*3+i] << i for i in range(3))
        if actual != mask.bit_count():
            raise ValueError(f'MAC_CUTS={mask}: synthesized latency {actual}, expected {mask.bit_count()}')
    (args.out/'PASS').write_text('Vivado: all 32 synthesized MAC cut-mask latencies match popcount\n')
    print('PASS: all 32 synthesized MAC cut-mask latencies')


if __name__ == '__main__':
    main()
