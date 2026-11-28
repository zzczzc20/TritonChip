#!/usr/bin/env python3
"""R42 issue: summary table of the OOC runs in a directory (x86 ~/r42_issue_out, or a local copy of the reports).
  python3 ooc_collect.py DIR > ooc_summary.txt
Per run: synthesis LUT / LUT as logic / FF / DSP / RAMB / URAM from synth_util.rpt, and for place-and-route screens the routed WNS, TNS, failing and total endpoints
(timing_routed.rpt), routed LUT/FF (utilization_routed.rpt), the number of endpoints below 1.5 ns (endpoints.csv)."""
import os, re, sys

d = sys.argv[1]


def grab(p, pat, grp=1):
    try:
        for l in open(p):
            m = re.search(pat, l)
            if m:
                return m.group(grp)
    except OSError:
        pass
    return None


rows = []
for n in sorted(os.listdir(d)):
    p = os.path.join(d, n)
    if not os.path.isdir(p):
        continue
    su = os.path.join(p, 'synth_util.rpt')
    ur = os.path.join(p, 'utilization_routed.rpt')
    tr = os.path.join(p, 'timing_routed.rpt')
    r = dict(name=n)
    if os.path.exists(su):
        r['syn_lut'] = grab(su, r'\| CLB LUTs\*?\s*\|\s*(\d+)')
        r['syn_logic'] = grab(su, r'\| LUT as Logic\s*\|\s*(\d+)')
        r['syn_ff'] = grab(su, r'\| CLB Registers\s*\|\s*(\d+)')
        r['dsp'] = grab(su, r'\| DSPs\s*\|\s*(\d+)')
        r['uram'] = grab(su, r'\| URAM\s*\|\s*(\d+)')
    if os.path.exists(tr):
        txt = open(tr).read()
        m = re.search(r'Design Timing Summary.*?\n\s*-+\s*\n\s*\n\s*WNS\(ns\)[^\n]*\n[^\n]*\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)', txt, re.S)
        if m:
            r['wns'], r['tns'], r['fail'], r['endpoints'] = m.groups()
    if os.path.exists(ur):
        r['rt_lut'] = grab(ur, r'\| CLB LUTs\s*\|\s*(\d+)')
        r['rt_ff'] = grab(ur, r'\| CLB Registers\s*\|\s*(\d+)')
    ep = os.path.join(p, 'endpoints.csv')
    if os.path.exists(ep):
        r['ep_lt1.5'] = sum(1 for _ in open(ep)) - 1
    rows.append(r)
cols = ['name', 'syn_lut', 'syn_logic', 'syn_ff', 'dsp', 'uram', 'rt_lut', 'rt_ff', 'wns', 'tns', 'fail', 'endpoints', 'ep_lt1.5']
print('\t'.join(cols))
for r in rows:
    print('\t'.join(str(r.get(c, '')) for c in cols))
