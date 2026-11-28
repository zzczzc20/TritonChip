#!/usr/bin/env python3
"""tile_census.py ENDPOINTS_CSV PERIOD_NS [N]   (R42-shrink; domain: OOC tile screen, routed)

Setup endpoints of a one-tile place and route (tools/r42/area/tile_screen.tcl: endpoints.csv = every endpoint with slack below 1.5 ns: slack,start,end,levels,delay,logic,net)
grouped by the unit that owns the endpoint (u_core/<child>, or the first component for the home slice, NI, host mux). Per unit: endpoints with slack < 0 at PERIOD,
endpoints that would fail at 150 MHz (path > 6.667 ns) and at 200 MHz (path > 5 ns) if the routed path stayed as it is (path = PERIOD - slack; the file is cut at slack 1.5, so
the 200 MHz column is complete only for PERIOD <= 6.5), worst slack, and the N worst paths with levels, delay and net fraction.
"""
import collections
import csv
import re
import sys

path, period = sys.argv[1], float(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def unit(pin):
    p = pin.split('/')
    if p[0] == 'u_core':
        return 'u_core/' + (p[1] if len(p) > 3 else '(core own)')
    return p[0] if len(p) > 2 else '(tile own)'


def fam(pin):
    leaf = pin.split('/')[-2] if '/' in pin else pin
    return re.sub(r'\[\d+\]', '[]', leaf)


by = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    by[unit(r['end'])].append((float(r['slack']), r))
print('| unit | endpoints < 1.5 ns | failing at %.2f ns | path > 6.667 ns | path > 5 ns | worst slack | worst path (levels, ns, net %%) | start -> end |' % period)
print('|---|--:|--:|--:|--:|--:|---|---|')
tot = [0, 0, 0, 0]
for u, lst in sorted(by.items(), key=lambda kv: min(x[0] for x in kv[1])):
    lst.sort(key=lambda x: x[0])
    n = len(lst)
    f0 = sum(1 for s, _ in lst if s < 0)
    f150 = sum(1 for s, _ in lst if period - s > 6.667)
    f200 = sum(1 for s, _ in lst if period - s > 5.0)
    for i, v in enumerate((n, f0, f150, f200)):
        tot[i] += v
    s, r = lst[0]
    print('| %s | %d | %d | %d | %d | %.3f | %s lv, %.2f, %.0f %% | %s -> %s |' % (u, n, f0, f150, f200, s, r['levels'], float(r['delay']), 100 * float(r['net']) / max(float(r['delay']), 1e-9), fam(r['start']), fam(r['end'])))
print('| **tile** | %d | %d | %d | %d | | | |' % tuple(tot))
