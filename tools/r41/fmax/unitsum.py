#!/usr/bin/env python3
"""One line per unit screen: LUT, logic/memory, FF, CARRY8, DSP, CLB, WNS.

  unitsum.py OUT_DIR [OUT_DIR ...]
Reads routed_util.rpt when present (else synth_util.rpt) and routed_timing.rpt.
"""
import re
import sys
from pathlib import Path


def cell(text, name):
    m = re.search(r'^\|\s*' + re.escape(name) + r'\s*\|\s*(\d+)', text, re.M)
    return int(m.group(1)) if m else None


for d in map(Path, sys.argv[1:]):
    routed = (d / 'routed_util.rpt').exists()
    u = (d / ('routed_util.rpt' if routed else 'synth_util.rpt')).read_text()
    wns = '-'
    t = d / 'routed_timing.rpt'
    if t.exists():
        m = re.search(r'WNS\(ns\).*?\n.*?\n\s*(-?\d+\.\d+)', t.read_text(), re.S)
        wns = m.group(1) if m else '?'
    row = dict(lut=cell(u, 'CLB LUTs') or cell(u, 'CLB LUTs*'), logic=cell(u, 'LUT as Logic'),
               mem=cell(u, 'LUT as Memory'), ff=cell(u, 'CLB Registers') or cell(u, 'Register as Flip Flop'),
               carry=cell(u, 'CARRY8'), dsp=cell(u, 'DSPs'), clb=cell(u, 'CLB') if routed else None)
    print(f"{d.name:40s} {'routed' if routed else 'synth '} " +
          ' '.join(f'{k}={v}' for k, v in row.items()) + f' wns={wns}')
