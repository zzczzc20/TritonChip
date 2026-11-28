#!/usr/bin/env python3
"""tile_report.py ENDPOINTS.csv PERIOD [BASELINE.csv] -- per-unit census of a routed OOC tile screen (tools/r42/area/tile_screen.tcl endpoints.csv: every endpoint
with slack below 1.5 ns) and, for the units of R42-clk2 (matrix, vector FPU, transfer engine, barrier), the failing start -> end families. With BASELINE the
failing counts of the baseline are shown beside. Markdown on stdout."""
import bisect, collections, csv, re, sys

def unit(pin):
    p = pin.split('/')
    if p[0] == 'u_core':
        return 'u_core/' + (p[1] if len(p) > 3 else '(core own)')
    return p[0] if len(p) > 2 else '(tile own)'

def load(f):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        by[unit(r['end'])].append((float(r['slack']), r))
    return by

def norm(pin):
    pin = re.sub(r'g_mac\[\d+\]\.g_lane\.u_mac/', '', pin)
    pin = re.sub(r'_replica(_\d+)?', '', pin)
    return re.sub(r'\[\d+\]', '[]', pin)

def row(by, u, period):
    l = sorted(s for s, _ in by.get(u, []))
    return (len(l), bisect.bisect_left(l, 0), bisect.bisect_left(l, -(6.2 - period)), bisect.bisect_left(l, -(6.667 - period)), l[0] if l else 9.0)

f, period = sys.argv[1], float(sys.argv[2])
by = load(f); base = load(sys.argv[3]) if len(sys.argv) > 3 else None
units = sorted(by, key=lambda u: min(s for s, _ in by[u]))
print('| unit | endpoints < 1.5 | failing at %.2f | path > 6.2 | path > 6.67 | worst slack |%s' % (period, ' baseline failing / > 6.2 / > 6.67 |' if base else ''))
print('|---|--:|--:|--:|--:|--:|%s' % ('---|' if base else ''))
tot = [0] * 4
for u in units:
    r = row(by, u, period)
    for i in range(4): tot[i] += r[i]
    b = ''
    if base:
        x = row(base, u, period); b = ' %d / %d / %d |' % (x[1], x[2], x[3])
    print('| %s | %d | %d | %d | %d | %.3f |%s' % (u, r[0], r[1], r[2], r[3], r[4], b))
print('| **tile** | %d | %d | %d | %d | | |' % tuple(tot))
for u, title in (('u_core/u_mtx', 'matrix engine'), ('u_core/u_vfpu', 'vector FPU'), ('u_core/u_txf', 'transfer engine'), ('u_core/u_bar', 'barrier')):
    c = collections.defaultdict(lambda: [0, 0, 9.0, 0])
    for s, r in by.get(u, []):
        k = (norm(r['start'].rsplit('/', 1)[0]).replace(u + '/', '').replace('u_core/', ''), norm(r['end'].rsplit('/', 1)[0]).replace(u + '/', ''))
        d = c[k]; d[1] += 1; d[2] = min(d[2], s); d[3] = max(d[3], int(r['levels']))
        if s < 0: d[0] += 1
    print('\n### %s: failing families\n\n| failing | of | worst slack | levels | start -> end |\n|--:|--:|--:|--:|---|' % title)
    for k, d in sorted(c.items(), key=lambda kv: -kv[1][0])[:12]:
        if d[0] == 0: break
        print('| %d | %d | %.3f | %d | %s -> %s |' % (d[0], d[1], d[2], d[3], k[0], k[1]))
