#!/usr/bin/env python3
"""top_paths_by_unit.py ENDPOINTS_CSV [N] -- the N (default 3) worst kernel-clock setup paths of every unit of a routed census
(same unit grouping as clock_census.py), with slack, logic levels, datapath delay, net fraction, start and end register families."""
import csv, re, sys, collections
sys.argv = [sys.argv[0]] + sys.argv[1:]
path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
tile_re = re.compile(r'g_tile\[(\d)\]\.u_tile/(.*)')
def unit(pin):
    m = tile_re.search(pin)
    if not m:
        if 'hmss' in pin or 'blp' in pin or 'ulp_ucs' in pin or 'ii_level0' in pin: return 'platform'
        k = pin.split('tnca_u50_1/inst/')[-1].split('/')
        return 'kernel:' + (k[0] if len(k) > 1 else '(regs)')
    t, rest = m.group(1), m.group(2).split('/')
    if rest[0] == 'u_core': return 't%s u_core/%s' % (t, rest[1] if len(rest) > 2 else '(own regs)')
    return 't%s %s' % (t, rest[0] if len(rest) > 2 else '(own regs)')
def short(pin):
    p = pin.replace('level0_i/ulp/tnca_u50_1/inst/', '')
    p = re.sub(r'g_tile\[(\d)\]\.u_tile/', r't\1:', p)
    return p[:95]
by = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    if not ('kernel' in r['startclk'] and 'kernel' in r['endclk']): continue
    by[unit(r['end'])].append((float(r['slack']), r))
for u, lst in sorted(by.items(), key=lambda kv: min(x[0] for x in kv[1])):
    if len(lst) < 20: continue
    lst.sort(key=lambda x: x[0])
    print('== %s (%d endpoints below the file limit)' % (u, len(lst)))
    for sl, r in lst[:N]:
        print('  %.3f ns  %2s levels  %5.2f ns (net %.0f%%)  %s -> %s' % (sl, r['levels'], float(r['delay']), 100 * float(r['net']) / max(float(r['delay']), 1e-9), short(r['start']), short(r['end'])))
