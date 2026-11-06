#!/usr/bin/env python3
"""G4 home-slice options: byte-shadow regression of the wide slice (RTL simulation).

    python3 tools/r41/g4_home_tests.py

LINE_WRITE_NOFILL (full-line stores allocate without reading the line):
  * tb_nuca_wide's randomised byte shadow (reads checked against every earlier
    write, partial and full; dirty evictions; reordered, stalling DRAM) with
    the option on, under every combination of the cache data pipeline, hit
    chaining with fast fill, and background reads, three seeds each; the
    bench fails if no full-line store took the no-fill path;
  * the same configurations with the option off, unchanged from before;
  * streaming full-line stores (SERVICE_MODE 3): with the option the slice
    issues no memory read at all, without it one per line.
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/r41'))
from wide_home_tests import run_case  # noqa: E402


def main():
    n = 0
    with tempfile.TemporaryDirectory(prefix='tnca-g4-home-') as temp:
        binary = str(Path(temp)/'home.vvp')
        for nofill in (0, 1):
            for pipe in (0, 1):
                for chain in (0, 1):
                    for bg in (0, 1):
                        for seed in (1, 7, 39):
                            out = run_case(binary, 'tb_nuca_wide',
                                           {'CACHE_DATA_PIPELINE': pipe, 'HIT_CHAIN': chain, 'FAST_FILL_READ': chain,
                                            'BACKGROUND_READ_REQUESTS': bg, 'LINE_WRITE_NOFILL': nofill},
                                           'tb/nuca/tb_nuca_wide.v', ['+nreq=3000', f'+seed={seed}'])
                            m = re.search(r'MEMORY reads=(\d+) writes=(\d+) nofill_allocs=(\d+)', out)
                            print(f'CASE nofill={nofill} pipe={pipe} chain={chain} bg={bg} seed={seed} '
                                  f'reads={m.group(1)} writes={m.group(2)} nofill_allocs={m.group(3)}', flush=True)
                            if not nofill:
                                assert m.group(3) == '0', out
                            n += 1
        reads = {}
        for nofill in (0, 1):
            out = run_case(binary, 'tb_nuca_wide', {'SERVICE_MODE': 3, 'LINE_WRITE_NOFILL': nofill, 'HIT_CHAIN': 1,
                                                    'FAST_FILL_READ': 1},
                           'tb/nuca/tb_nuca_wide.v', ['+nreq=2048', '+min_wb=0'])
            reads[nofill] = int(re.search(r'MEMORY reads=(\d+)', out).group(1))
            span = re.search(r'span_cycles=(\d+)', out).group(1)
            print(f'STREAM full-line stores nofill={nofill}: memory reads={reads[nofill]} span_cycles={span}', flush=True)
            n += 1
        assert reads[1] == 0 and reads[0] > 0, reads
    print(f'G4 HOME ALL {n} CASES PASS', flush=True)


if __name__ == '__main__':
    main()
