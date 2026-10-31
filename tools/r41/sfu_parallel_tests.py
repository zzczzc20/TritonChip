#!/usr/bin/env python3
"""Verify parallel SFU bit equivalence, launch timing and reset behavior."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-sfu-parallel-') as temp:
        for lanes in (2, 3, 4, 8):
            binary = str(Path(temp)/'sfu.vvp')
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-s', 'tb_sfu_parallel',
                f'-Ptb_sfu_parallel.PAR={lanes}', '-o', binary, 'tb/core/tb_sfu_parallel.v',
                'rtl/core/tnca_sfu.v', 'rtl/gen/tnca_sfu_exp2_xls.v',
                'rtl/gen/tnca_sfu_rcp_xls.v', 'rtl/gen/tnca_sfu_rsqrt_xls.v',
                'rtl/gen/tnca_sfu_log2_xls.v'], cwd=ROOT, check=True, timeout=120)
            subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=120)
    print('SFU PARALLEL ALL CONFIGURATIONS PASS', flush=True)


if __name__ == '__main__':
    main()
