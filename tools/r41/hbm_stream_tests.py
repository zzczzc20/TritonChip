#!/usr/bin/env python3
"""Functional stream-engine AXI tests; no board or image performance claims."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    cases = [(stall, error, base) for stall in (0, 1) for error in range(8)
             for base in ((0x100000000, 0x1f0000000) if error == 0 else (0x100000000,))]
    with tempfile.TemporaryDirectory(prefix='tnca-hbm-stream-') as temporary:
        for stall, error, base in cases:
            binary = str(Path(temporary) / 'stream.vvp')
            subprocess.run([
                'iverilog', '-g2012', '-s', 'tb_stream_axi',
                f'-Ptb_stream_axi.STALL={stall}', f'-Ptb_stream_axi.ERROR={error}',
                f"-Ptb_stream_axi.PC_BASE=64'h{base:x}", '-o', binary,
                'rtl/fpga/tnca_stream_axi.v', 'rtl/fpga/tnca_stream_fifo.v',
                'rtl/common/tnca_fifo.v', 'tb/fpga/tb_stream_axi.v',
            ], cwd=ROOT, check=True, timeout=60)
            subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=60)
    print(f'HBM STREAM FUNCTIONAL PASS {len(cases)} configurations', flush=True)


if __name__ == '__main__':
    main()
