#!/usr/bin/env python3
"""Compare real multi-unit LSU data and memory effects to the legacy pair."""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-lsu-bank-') as temp:
        for units in (2, 4, 8):
            for seed in (1, 7, 39):
                for loads in (0, 1):
                    binary = str(Path(temp)/'bank.vvp')
                    args = ['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/core',
                            '-s', 'tb_lsu_bank', '-o', binary, f'-Ptb_lsu_bank.UNITS={units}',
                            f'-Ptb_lsu_bank.SEED={seed}', '-Ptb_lsu_bank.ACCESSES=600',
                            f'-Ptb_lsu_bank.FORCE_LOADS={loads}',
                            f'-Ptb_lsu_bank.MAXLAT={200 if loads else 32}', 'tb/core/tb_lsu_bank.v']
                    subprocess.run(args, cwd=ROOT, check=True, timeout=60)
                    subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=120)
        for units in (2, 4, 8):
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/core',
                '-s', 'tb_lsu_bank_reset', f'-Ptb_lsu_bank_reset.UNITS={units}',
                '-o', binary, 'tb/core/tb_lsu_bank_reset.v'], cwd=ROOT, check=True, timeout=60)
            subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=30)
        subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/core',
            '-s', 'tb_lsu_wide', '-o', binary, 'tb/core/tb_lsu_wide.v'], cwd=ROOT, check=True, timeout=60)
        subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=60)
        subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/core',
            '-s', 'tnca_lsu_bank', '-Ptnca_lsu_bank.UNITS=8', '-Ptnca_lsu_bank.NUM_BEATS=8',
            '-o', binary, 'rtl/core/tnca_lsu_bank.v'], cwd=ROOT, check=True, timeout=60)
        bad = subprocess.run(['vvp', binary], cwd=ROOT, text=True, capture_output=True, timeout=30)
        assert bad.returncode != 0 and 'exceed tag capacity' in bad.stdout, bad
        print('LSU BANK EXCESS TAG CAPACITY REFUSAL PASS', flush=True)
    print('LSU BANK ALL 18 CONFIGURATIONS PASS', flush=True)


if __name__ == '__main__':
    main()
