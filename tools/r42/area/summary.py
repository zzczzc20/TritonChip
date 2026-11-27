#!/usr/bin/env python3
"""summary.py DIR... -- one line per OOC result directory (route_* if present, else synth_*): LUT (logic/mem), FF, DSP, CARRY8,
BRAM, URAM, WNS, WHS and the worst endpoint. Numbers are read from the Vivado reports in the directory, nothing else."""
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
    print('%-26s %-5s %6s %6s %6s %6s %5s %6s %4s %4s %4s %8s %8s  %s' % ('dir', 'stage', 'LUT', 'logic', 'mem', 'FF', 'DSP', 'CARRY8', 'BRAM', 'URAM', 'CLB', 'WNS', 'WHS', 'worst endpoint'))
    for d in sys.argv[1:]:
        for stage in ('synth', 'route'):
            u = os.path.join(d, stage + '_util.rpt')
            if not os.path.exists(u):
                continue
            lut = grab(u, r'\|\s*CLB LUTs\*?\s+\|\s+(\d+)', cast=int)
            logic = grab(u, r'\|\s*LUT as Logic\s+\|\s+(\d+)', cast=int)
            mem = grab(u, r'\|\s*LUT as Memory\s+\|\s+(\d+)', cast=int)
            ff = grab(u, r'\|\s*CLB Registers\s+\|\s+(\d+)', cast=int)
            dsp = grab(u, r'\|\s*DSPs\s+\|\s+(\d+)', cast=int)
            c8 = grab(u, r'\|\s*CARRY8\s+\|\s+(\d+)', cast=int)
            bram = grab(u, r'\|\s*Block RAM Tile\s+\|\s+([\d.]+)')
            uram = grab(u, r'\|\s*URAM\s+\|\s+(\d+)', cast=int)
            clb = grab(u, r'\|\s*CLB\s+\|\s+(\d+)', cast=int)
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
            print('%-26s %-5s %6s %6s %6s %6s %5s %6s %4s %4s %4s %8s %8s  %s' % (os.path.basename(d.rstrip('/')), stage, lut, logic, mem, ff, dsp, c8, bram, uram, clb, wns, whs, ep[:90]))


main()
