#!/usr/bin/env python3
"""ooc_summary.py DIR... -- one line per OOC result directory (route_* if present, else synth_*): LUT, FF, DSP, CARRY8,
WNS, worst endpoint. Numbers are read from the Vivado reports in the directory, nothing else."""
import os
import re
import sys


def grab(path, pat, group=1, cast=float):
    if not os.path.exists(path):
        return None
    for line in open(path, errors='replace'):
        m = re.search(pat, line)
        if m:
            return cast(m.group(group)) if cast else m.group(group)
    return None


def main():
    print('%-22s %-6s %7s %7s %5s %6s %8s %8s  %s' % ('dir', 'stage', 'LUT', 'FF', 'DSP', 'CARRY8', 'WNS', 'WHS', 'worst endpoint'))
    for d in sys.argv[1:]:
        for stage in ('synth', 'route'):
            u = os.path.join(d, stage + '_util.rpt')
            if not os.path.exists(u):
                continue
            lut = grab(u, r'\| CLB LUTs\*?\s+\|\s+(\d+)', cast=int)
            ff = grab(u, r'\| CLB Registers\s+\|\s+(\d+)', cast=int)
            dsp = grab(u, r'\| DSPs\s+\|\s+(\d+)', cast=int)
            c8 = grab(u, r'\| CARRY8\s+\|\s+(\d+)', cast=int)
            s = os.path.join(d, stage + '_summary.rpt')
            wns = whs = None
            if os.path.exists(s):
                lines = open(s).read().splitlines()
                for i, l in enumerate(lines):
                    if 'WNS(ns)' in l:
                        v = lines[i + 2].split()
                        wns, whs = float(v[0]), float(v[4])
                        break
            ep = ''
            e = os.path.join(d, stage + '_endpoints.csv')
            if os.path.exists(e):
                rows = [l.strip().split(',') for l in open(e).read().splitlines()[1:] if l.strip()]
                if rows:
                    w = min(rows, key=lambda r: float(r[0]))
                    ep = '%s -> %s' % (w[1], w[2])
            print('%-22s %-6s %7s %7s %5s %6s %8s %8s  %s' % (os.path.basename(d.rstrip('/')), stage, lut, ff, dsp, c8, wns, whs, ep[:110]))


main()
