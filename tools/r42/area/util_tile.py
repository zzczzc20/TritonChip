import sys, re, collections
# util_tile.py REPORT TILE : aggregate a hierarchical synthesis utilization report of tile TILE by (parent, module type)
rep, tile = sys.argv[1], int(sys.argv[2])
rows = []
on = False
for l in open(rep, errors='replace'):
    p = l.split('|')
    if len(p) < 12: continue
    name = p[1].rstrip(); nm = name.strip()
    if nm.startswith('g_tile[%d].u_tile' % tile): on = True; start_depth = len(name) - len(name.lstrip())
    elif on and re.match(r'g_tile\[\d+\]\.u_tile', nm) and not nm.startswith('g_tile[%d]' % tile): break
    if not on: continue
    depth = (len(name) - len(name.lstrip()) - start_depth) // 2
    vals = [v.strip() for v in p[3:13]]
    mod = re.sub(r'__(parameterized)?\d+$', '', p[2].strip())
    rows.append((depth, nm, mod, vals))
# depth-1 and depth-2 modules; for the named parents list children aggregated by module type
def agg(parent_filter, dmin, dmax):
    cur = None; out = collections.OrderedDict()
    for d, nm, mod, v in rows:
        pass
tot = {}
print('%-3s %-40s %-26s %8s %8s %7s %5s %7s %6s %6s %5s %5s' % ('d', 'instance', 'module', 'LUT', 'logic', 'LUTRAM', 'SRL', 'FF', 'RAMB36', 'RAMB18', 'URAM', 'DSP'))
parent = None
agg2 = collections.OrderedDict()
for d, nm, mod, v in rows:
    if nm.startswith('('): continue
    if d <= 2:
        print('%-3d %-40s %-26s %8s %8s %7s %5s %7s %6s %6s %5s %5s' % ((d, nm[:40], mod[:26]) + tuple(v[:9])))
    if d == 2: parent = nm
    if d == 1: parent = nm
    if d >= 3:
        key = (parent, mod)
        a = agg2.setdefault(key, [0, 0] + [0] * 9)
        if d == 3:   # count direct children only once per instance
            a[0] += 1
            for i in range(9): a[2 + i] += int(v[i])
print()
print('children of depth 3, by parent and module type (count, total LUT, logic, LUTRAM, SRL, FF, RAMB36, RAMB18, URAM, DSP)')
for (par, mod), a in agg2.items():
    if a[2] >= 300: print('%-34s %-30s n=%-3d %8d %8d %7d %5d %7d %6d %6d %5d %5d' % (par[:34], mod[:30], a[0], *a[2:11]))
