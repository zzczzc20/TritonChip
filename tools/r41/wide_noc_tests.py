#!/usr/bin/env python3
"""Real mesh/NI packet integrity, backpressure and sustained issue checks."""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]


def geometry_guards(temp):
    for name in ('tnca_core_ni', 'tnca_slice_ni'):
        binary = str(Path(temp)/'invalid.vvp')
        subprocess.run(['iverilog', '-g2012', '-Irtl/noc', '-s', name,
            f'-P{name}.DATA_W=512', f'-P{name}.BYTE_W=8', '-o', binary,
            f'rtl/noc/{name}.v'], cwd=ROOT, check=True, timeout=60)
        bad = subprocess.run(['vvp', binary], cwd=ROOT, text=True, capture_output=True, timeout=30)
        assert bad.returncode and 'invalid NI' in bad.stdout, bad
        print(name, 'INCONSISTENT GEOMETRY REFUSED', flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-wide-noc-') as temp:
        for width in (64, 512):
            for stages in (0, 2):
                for stall in (0, 1):
                    binary = str(Path(temp)/'mesh.vvp')
                    subprocess.run(['iverilog', '-g2012', '-Irtl/noc', '-y', 'rtl/noc',
                        '-y', 'rtl/common', '-s', 'tb_wide_ni', f'-Ptb_wide_ni.DATA_W={width}',
                        f'-Ptb_wide_ni.LINK_STAGES={stages}', f'-Ptb_wide_ni.STALL={stall}',
                        '-o', binary, 'tb/noc/tb_wide_ni.v'], cwd=ROOT, check=True, timeout=60)
                    subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=180)
        geometry_guards(temp)
    print('WIDE NOC ALL 8 CONFIGURATIONS PASS', flush=True)


if __name__ == '__main__':
    main()
