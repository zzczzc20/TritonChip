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
    configurations = 0
    failures = []
    with tempfile.TemporaryDirectory(prefix='tnca-wide-noc-') as temp:
        for width, tag_w in ((64, 5), (512, 5), (512, 7)):
            for stages in (0, 2):
                for stall in (0, 1):
                    binary = str(Path(temp)/'mesh.vvp')
                    subprocess.run(['iverilog', '-g2012', '-Irtl/noc', '-y', 'rtl/noc',
                        '-y', 'rtl/common', '-s', 'tb_wide_ni', f'-Ptb_wide_ni.DATA_W={width}',
                        f'-Ptb_wide_ni.TAG_W={tag_w}', f'-Ptb_wide_ni.LINK_STAGES={stages}',
                        f'-Ptb_wide_ni.STALL={stall}', '-o', binary, 'tb/noc/tb_wide_ni.v'],
                        cwd=ROOT, check=True, timeout=60)
                    try:
                        subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=180)
                        configurations += 1
                    except subprocess.CalledProcessError:
                        failures.append((width, tag_w, stages, stall))
                        print(f'WIDE NOC FAIL data={width} tag={tag_w} links={stages} stall={stall}', flush=True)
        geometry_guards(temp)
    if failures:
        raise RuntimeError(f'WIDE NOC {len(failures)} CONFIGURATIONS FAILED: {failures}')
    print(f'WIDE NOC ALL {configurations} CONFIGURATIONS PASS', flush=True)


if __name__ == '__main__':
    main()
