#!/usr/bin/env python3
"""Check that line storage maps to FPGA block RAM, rather than data flops.

This is a synthesis resource check. It establishes neither board timing nor
the cache protocol; the randomized bridge suite verifies the latter.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'rtl/fpga/tnca_axi_line_ram.v'


def main():
    with tempfile.TemporaryDirectory(prefix='r40-line-ram-') as work:
        mapped = Path(work) / 'mapped.json'
        commands = ('read_verilog rtl/fpga/tnca_axi_line_ram.v; '
                    'synth_xilinx -family xcu -top tnca_axi_line_ram -noiopad -noclkbuf; '
                    f'write_json {mapped}')
        result = subprocess.run(['yosys', '-Q', '-T', '-p', commands], cwd=ROOT,
                                capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-1500:])
        cells = json.loads(mapped.read_text())['modules']['tnca_axi_line_ram']['cells']
        counts = {}
        for cell in cells.values():
            counts[cell['type']] = counts.get(cell['type'], 0) + 1
        ram = sum(n for kind, n in counts.items() if kind.startswith('RAMB'))
        flops = sum(n for kind, n in counts.items() if kind.startswith('FD'))
        assert ram > 0 and flops <= 64, counts
        print(json.dumps(dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                              cells=counts, block_ram_cells=ram, flip_flops=flops,
                              geometry_words=128, word_bits=64, measured_performance=False),
                         sort_keys=True))
        print('r40-ram-cache-inference-tests PASS: block RAM inferred; no timing claim')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
