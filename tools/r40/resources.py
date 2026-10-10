#!/usr/bin/env python3
"""An image's resources and timing, from Vivado's routed reports, as JSON.

    python3 tools/r40/resources.py --util full_util_routed.rpt \
        --timing <..._timing_summary_routed.rpt> [--clock ap_clk] \
        [--image <xclbin or sha>] [--clock-mhz 100.0] --out resources.json

doc/r40_tasks_and_acceptance.md sec 3.2 item 4 and sec 3.4: every
comparison reports what the mechanism cost -- LUT, FF, BRAM, URAM and DSP
for the whole chip and for each SLR, the SLR crossings, and post-route setup
and hold slack of the kernel clock and of the design. tools/r40/ablate.py
--resources puts two of these beside a speedup. Reads the text reports the
v++ link leaves in its impl_1 run directory; nothing here runs Vivado.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TOTAL_ROWS = {'lut': 'CLB LUTs', 'ff': 'CLB Registers', 'bram': 'Block RAM Tile',
              'uram': 'URAM', 'dsp': 'DSPs'}


def _num(x):
    x = x.strip().lstrip('<')
    try:
        return int(x)
    except ValueError:
        return float(x)


def parse_util(text):
    """Totals (the first table row of each kind) and the per-SLR table."""
    total = {}
    for key, label in TOTAL_ROWS.items():
        m = re.search(r'^\|\s*' + re.escape(label) + r'\s*\|\s*([0-9.]+)\s*\|', text, re.M)
        if m:
            total[key] = _num(m.group(1))
    slr = {}
    sec = re.search(r'SLR CLB Logic and Dedicated Block Utilization\n-+\n(.*?)\n\* ', text, re.S)
    if sec:
        lines = [l for l in sec.group(1).splitlines() if l.startswith('|')]
        head = [c.strip() for c in lines[0].strip('|').split('|')]
        names = [h for h in head[1:] if h.startswith('SLR') and '%' not in h]
        for l in lines[1:]:
            cells = [c.strip() for c in l.strip('|').split('|')]
            label = cells[0]
            for key, want in TOTAL_ROWS.items():
                if label == want:
                    for i, n in enumerate(names):
                        slr.setdefault(n, {})[key] = _num(cells[1 + i])
    crossings = {}
    for m in re.finditer(r'^\|\s*(SLR\d) <-> (SLR\d)\s*\|\s*(\d+)\s*\|\s*\|\s*(\d+)', text, re.M):
        crossings[f'{m.group(1)}<->{m.group(2)}'] = dict(used=int(m.group(3)), available=int(m.group(4)))
    return total, slr, crossings


def parse_timing(text, clock):
    """Design WNS/WHS and the named clock's own, from the summary's head."""
    head = '\n'.join(text.splitlines()[:600])
    out = {}
    m = re.search(r'Design Timing Summary.*?\n\s*-+.*?\n\s*([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+\s+([-0-9.]+)',
                  head, re.S)
    if m:
        out['design_wns_ns'], out['design_whs_ns'] = float(m.group(1)), float(m.group(3))
    intra = head.split('Intra Clock Table', 1)
    if len(intra) == 2:
        m = re.search(r'^\s*' + re.escape(clock) + r'\s+([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+\s+([-0-9.]+)',
                      intra[1], re.M)
        if m:
            out['wns_ns'], out['whs_ns'] = float(m.group(1)), float(m.group(3))
    m = re.search(r'^\s*' + re.escape(clock) + r'\s+\{[0-9. ]+\}\s+([0-9.]+)\s+([0-9.]+)', head, re.M)
    if m:
        out['period_ns'], out['target_mhz'] = float(m.group(1)), float(m.group(2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--util', type=Path, required=True)
    ap.add_argument('--timing', type=Path, required=True)
    ap.add_argument('--clock', default='ap_clk')
    ap.add_argument('--image', help='the xclbin (hashed) or its sha256')
    ap.add_argument('--clock-mhz', type=float, help='the measured device clock')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    total, slr, crossings = parse_util(args.util.read_text())
    timing = parse_timing(args.timing.read_text(errors='replace'), args.clock)
    if args.clock_mhz:
        timing['clock_mhz'] = args.clock_mhz
    image = args.image
    if image and Path(image).is_file():
        image = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    missing = [k for k in TOTAL_ROWS if k not in total] + \
              [k for k in ('wns_ns', 'whs_ns') if k not in timing]
    doc = dict(schema='tnca-r40-resources/1', image=image, clock=args.clock, total=total, slr=slr,
               slr_crossings=crossings, timing=timing,
               sources={k: dict(name=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                        for k, f in (('util', args.util), ('timing', args.timing))})
    text = json.dumps(doc, indent=1) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    if missing:
        print(f'resources: not found in the reports: {missing}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
