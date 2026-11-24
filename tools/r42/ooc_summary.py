#!/usr/bin/env python3
"""Resource and timing table of the OOC screens (domain: out-of-context place and route, 10 ns, SLR0 pblock) from the gzipped
routed utilization reports under evidence/r42/lineelig/ooc/ and the WNS lines of wns.txt."""
import gzip, re, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else 'evidence/r42/lineelig/ooc')
runs = [('tnca_tensor, unchanged (base commit file)', 'out_base'), ('tnca_tensor, current file, LINE_HALF 0', 'out_new0'),
        ('tnca_tensor, current file, LINE_HALF 1', 'out_new1'), ('tnca_tensor, unchanged, 32 word tags', 'out_base_t5'),
        ('tnca_tensor, LINE_HALF 1, 32 word tags', 'out_new1_t5'),
        ('tnca_matrix, unchanged, ABUF 1', 'm/out_mbase'), ('tnca_matrix, current file, ABUF 1', 'm/out_mnew1'),
        ('tnca_matrix, current file, ABUF 2', 'm/out_mnew2')]
wns = {}
for line in (root / 'wns.txt').read_text().splitlines():
    m = re.match(r'(\S+) .*wns=(\S+) whs=(\S+)', line)
    if m:
        wns[m.group(1)] = (m.group(2), m.group(3))
def get(txt, label):
    m = re.search(r'^\| ' + re.escape(label) + r'\s*\|\s*(\d+)', txt, re.M)
    return int(m.group(1)) if m else None
print(f"{'unit / variant':46s} {'LUT':>7s} {'LUTRAM':>7s} {'FF':>7s} {'BRAM':>5s} {'URAM':>5s} {'DSP':>4s} {'WNS ns':>7s} {'WHS ns':>7s}")
for name, d in runs:
    txt = gzip.open(root / d / 'routed_util.rpt.gz', 'rt').read()
    key = d.split('/')[-1].replace('out_', '')
    w = wns.get(key, ('-', '-'))
    print(f"{name:46s} {get(txt, 'CLB LUTs'):7d} {get(txt, 'LUT as Memory'):7d} {get(txt, 'CLB Registers'):7d} {get(txt, 'Block RAM Tile'):5d} {get(txt, 'URAM'):5d} {get(txt, 'DSPs'):4d} {w[0]:>7s} {w[1]:>7s}")
