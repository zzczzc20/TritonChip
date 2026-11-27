#!/usr/bin/env python3
"""clock_census.py ENDPOINTS_CSV [--period 10] -- group the endpoints of a routed timing census by owning unit.

Input: endpoints.csv of dcp_timing.tcl (slack, start pin, end pin, start clock, end clock, levels, delay, logic, net, skew,
requirement) or tile_screen.tcl (slack, start, end, levels, delay, logic, net).
For every unit (tile index, then the first hierarchy components below it) it lists the number of setup endpoints and, with
the constraint period P, how many would fail at 150 MHz (6.667 ns) and 200 MHz (5 ns) if the data path stayed as routed
(slack < P - 6.667, slack < P - 5), the worst slack, the median logic levels and datapath delay of the failing ones, and
the dominant start/end register families. Nothing is estimated beyond that subtraction: it is the routed path delay.
"""
import collections, csv, re, statistics, sys

args = sys.argv[1:]
if '--minn' in args:
    i = args.index('--minn'); del args[i:i + 2]
period = 10.0
if '--period' in args:
    i = args.index('--period'); period = float(args[i + 1]); del args[i:i + 2]
path = args[0]
rows = list(csv.DictReader(open(path)))
tile_re = re.compile(r'g_tile\[(\d)\]\.u_tile/(.*)')

def unit(pin):
    m = tile_re.search(pin)
    if not m:
        # outside a tile: first two components
        parts = pin.split('/')
        if 'hmss' in pin or 'blp' in pin or 'ulp_ucs' in pin or 'ii_level0' in pin: return 'platform'
        # kernel-level logic outside the tiles: outlets, mesh, dispatch, control
        k = pin.split('tnca_u50_1/inst/')[-1].split('/')
        return 'kernel:' + (k[0] if len(k) > 1 else '(regs)')
    t, rest = m.group(1), m.group(2).split('/')
    # unit = u_core/<child> for the core, else the first component
    if rest[0] == 'u_core':
        return 't%s u_core/%s' % (t, rest[1] if len(rest) > 2 else '(own regs)')
    return 't%s %s' % (t, rest[0] if len(rest) > 2 else '(own regs)')

def fam(pin):
    leaf = pin.split('/')[-2] if '/' in pin else pin
    return re.sub(r'\[\d+\]', '[]', leaf)

def f(x):
    try: return float(x)
    except: return None

lim150, lim200 = period - 6.6667, period - 5.0
units = collections.defaultdict(list)
for r in rows:
    sl = f(r['slack'])
    if sl is None: continue
    clk_ok = True
    if 'startclk' in r:
        clk_ok = ('kernel' in r['startclk'] and 'kernel' in r['endclk']) or r['startclk'] == r['endclk'] == 'clk'
    if not clk_ok: continue
    units[unit(r['end'])].append((sl, r))

print('period %.3f ns; endpoints with slack below %.2f ns are in the file; fail@150 = slack < %.2f, fail@200 = slack < %.2f' % (period, max(f(r['slack']) or 0 for r in rows), lim150, lim200))
print('%-28s %7s %8s %8s %8s %8s %6s %7s  %s' % ('unit (endpoint side)', 'n', 'worst', 'fail150', 'fail200', 'medLvl', 'medDP', 'netFrac', 'top end families'))
tot150 = tot200 = 0
MINN = 1
if '--minn' in sys.argv: MINN = int(sys.argv[sys.argv.index('--minn') + 1])
for u, lst in sorted(units.items(), key=lambda kv: min(x[0] for x in kv[1])):
    if len(lst) < MINN: continue
    worst = min(x[0] for x in lst)
    f150 = [x for x in lst if x[0] < lim150]; f200 = [x for x in lst if x[0] < lim200]
    tot150 += len(f150); tot200 += len(f200)
    lv = [f(x[1]['levels']) for x in f150 if f(x[1]['levels']) is not None]
    dp = [f(x[1]['delay']) for x in f150 if f(x[1]['delay']) is not None]
    nf = [f(x[1]['net']) / f(x[1]['delay']) for x in f150 if f(x[1].get('net')) is not None and f(x[1]['delay'])]
    fm = collections.Counter(fam(x[1]['end']) for x in f150).most_common(3)
    print('%-28s %7d %8.3f %8d %8d %8s %6s %7s  %s' % (u[:28], len(lst), worst, len(f150), len(f200),
          ('%.0f' % statistics.median(lv)) if lv else '-', ('%.1f' % statistics.median(dp)) if dp else '-',
          ('%.2f' % statistics.median(nf)) if nf else '-', ', '.join('%s(%d)' % kv for kv in fm)))
print('total endpoints failing at 150 MHz: %d, at 200 MHz: %d' % (tot150, tot200))
