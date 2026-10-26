#!/usr/bin/env python3
"""Replicated stream control with real lane RTL and independent AXI slaves."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-stream-bank-') as temporary:
        for lanes in (2, 8, 32):
            binary = str(Path(temporary)/'bank.vvp')
            subprocess.run(['iverilog', '-g2012', '-s', 'tb_stream_bank',
                f'-Ptb_stream_bank.LANES={lanes}', '-o', binary,
                'rtl/fpga/tnca_stream_bank.v', 'rtl/fpga/tnca_stream_axi.v',
                'rtl/fpga/tnca_stream_fifo.v', 'rtl/common/tnca_fifo.v',
                'tb/fpga/tb_stream_bank.v'], cwd=ROOT, check=True, timeout=60)
            subprocess.run(['vvp', '-i', binary], cwd=ROOT, check=True, timeout=600)
    print('HBM STREAM BANK FUNCTIONAL PASS 2/8/32 real lanes', flush=True)


if __name__ == '__main__':
    main()
