#!/usr/bin/env python3
"""Synthesize new control/line modules and refuse invalid geometries."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'tnca_elastic_link': ['rtl/common/tnca_elastic_link.v', 'rtl/common/tnca_fifo.v'],
    'tnca_line_axi': ['rtl/fpga/tnca_line_axi.v'],
    'tnca_line_link': ['rtl/common/tnca_line_link.v', 'rtl/common/tnca_fifo.v'],
    'tnca_multicore_mmio': ['rtl/fpga/tnca_multicore_mmio.v'],
    'tnca_mem_ports': ['rtl/nuca/tnca_mem_ports.v', 'rtl/nuca/tnca_mem_arbiter.v'],
}


def main():
    for top, sources in CASES.items():
        script = f"read_verilog -sv {' '.join(sources)}; hierarchy -check -top {top}; proc; check -assert; memory -nomap; stat"
        result = subprocess.run(['yosys', '-Q', '-T', '-p', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise AssertionError(f'{top} synthesis failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}')
        print(f'SYNTHESIS PASS {top}', flush=True)
    bad = [('tnca_elastic_link', 'STAGES', -1), ('tnca_line_axi', 'DATA_W', 96), ('tnca_line_axi', 'OUT_N', 0),
           ('tnca_line_axi', 'MEM_BASE', 1), ('tnca_line_link', 'STAGES', -1),
           ('tnca_multicore_mmio', 'PORTS', 4), ('tnca_multicore_mmio', 'MEM_BASE', 1),
           ('tnca_mem_ports', 'PORTS', 4), ('tnca_mem_ports', 'N_W', 1)]
    with tempfile.TemporaryDirectory(prefix='r40-invalid-geometry-') as temp:
        for top, parameter, value in bad:
            result = subprocess.run(['iverilog', '-g2012', '-s', top,
                                     f'-P{top}.{parameter}={value}', '-o', str(Path(temp) / 'bad.vvp'),
                                     *CASES[top]], cwd=ROOT, capture_output=True, text=True, timeout=60)
            assert result.returncode and 'invalid_' in result.stderr, (top, parameter, result.stderr)
    print(f'MULTICORE CONTROL SYNTHESIS PASS {len(CASES)} modules and {len(bad)} rejected geometries', flush=True)


if __name__ == '__main__':
    main()
