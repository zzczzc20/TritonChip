#!/usr/bin/env python3
"""image_census.py ENDPOINTS_CSV [KERNEL_CLOCK]  -- image-level path census of a routed image (domain: routed run-3 image, 10 ns, read-only analysis of tools/r42/area/dcp_timing.tcl output).

Input: endpoints.csv (slack,start,end,startclk,endclk,levels,delay,logic,net,skew,req) of the routed image. Kernel-clock endpoints (end clock = the kernel clock, default
clk_kernel_00_unbuffered_net) with slack below +2.0 ns at 10 ns (a path longer than 8 ns: fails an 8.0 ns link if nothing changes; below +1.5 ns fails 8.5 ns) are classified:
  A  tile-internal      start and end in the same tile (g_tile[k].u_tile), split by unit
  B  SLR crossing/mesh  mesh, home links, command links, outlet links, tile paths that start outside their tile
  C  shell/HBM/ctrl     HBM AXI outlet adapters, HBM subsystem and platform blocks, device top (control/MMIO), dispatch, deadline, registers, TMU, code store
  D  other
Prints a table per group: endpoints with slack < 2.0 / 1.5 / 1.0 / 0.5, worst slack, worst path (levels, delay, net share) and the top end-register families."""
import collections
import csv
import re
import sys

path = sys.argv[1]
KC = sys.argv[2] if len(sys.argv) > 2 else 'clk_kernel_00_unbuffered_net'
T = (2.0, 1.5, 1.0, 0.5)


def norm(s):
    return re.sub(r'\[\d+\]', '[]', s)


def tile_of(p):
    m = re.search(r'g_tile\[(\d+)\]\.u_tile', p)
    return m.group(1) if m else None


def unit_of(p):
    i = p.find('u_tile/')
    if i < 0:
        return '?'
    q = p[i + 7:].split('/')
    if q[0] == 'u_core':
        return 'u_core/' + (q[1] if len(q) > 3 else '(core own)')
    return q[0] if len(q) > 2 else '(tile own)'


def fam(p):
    q = p.split('/')
    return norm(q[-2] if len(q) > 1 else p)


def classify(s, e):
    te, ts = tile_of(e), tile_of(s)
    if te is not None:
        if ts == te:
            return 'A', 'tile-internal: ' + unit_of(e)
        return 'B', 'tile boundary: starts outside the tile (' + ('another tile' if ts else 'shell/mesh/link') + ')'
    ne = norm(e)
    if 'u_mesh' in e:
        return 'B', 'mesh (u_mesh)'
    if 'g_home_link' in e or 'g_cache_handoff' in e:
        return 'B', 'home links (SLR crossing)'
    if 'g_command_link' in e:
        return 'B', 'command links'
    if re.search(r'g_outlet\[\d+\]\.link', e):
        return 'B', 'HBM outlet links (SLR crossing)'
    if re.search(r'g_outlet\[\d+\]\.adapter', e):
        return 'C', 'HBM AXI outlet adapters'
    if '/hmss_0/' in e or '/blp/' in e or 'ulp_ucs' in e or 'ii_level0' in e:
        return 'C', 'platform / HBM subsystem'
    if 'g_local_tmu' in e:
        return 'C', 'TMU (shell, outside the tiles)'
    if 'g_dispatch' in e or 'g_code' in e or 'u_deadline' in e or 'invalid_id' in e:
        return 'C', 'dispatch / code / deadline (shell)'
    if e.endswith('/registers') or '/registers/' in e:
        return 'C', 'kernel registers'
    if re.search(r'/tnca_u50_1/inst/device/[A-Za-z_]+_reg', e) or re.search(r'/device/(control|mmio)', e):
        return 'C', 'device top: control / MMIO'
    if re.search(r'/tnca_u50_1/inst/device/', e) and 'system' not in e.split('/device/')[1].split('/')[0]:
        return 'C', 'device top: control / MMIO'
    return 'D', 'other: ' + '/'.join(ne.split('/')[-4:-1])


grp = collections.defaultdict(lambda: {'n': [0] * len(T), 'worst': 9.9, 'wp': None, 'fam': collections.Counter(), 'rst': [0] * len(T), 'rworst': 9.9})
tot = [0] * len(T)
for r in csv.DictReader(open(path)):
    if r['endclk'] != KC:
        continue
    s = float(r['slack'])
    if s >= T[0]:
        continue
    cat, sub = classify(r['start'], r['end'])
    g = grp[(cat, sub)]
    isr = 'reset_sync' in r['start']
    for i, t in enumerate(T):
        if s < t:
            g['n'][i] += 1
            tot[i] += 1
            if isr:
                g['rst'][i] += 1
    if isr and s < g['rworst']:
        g['rworst'] = s
    if s < g['worst']:
        g['worst'] = s
        g['wp'] = (r['levels'], r['delay'], r['net'])
    g['fam'][fam(r['end'])] += 1
print('| group | endpoints < 2.0 / 1.5 / 1.0 / 0.5 ns slack | of which start at the global reset synchroniser (reset_sync_reg) | worst slack | worst path (levels, ns, net ns) | top end registers |')
print('|---|---|---|--:|---|---|')
for (cat, sub), g in sorted(grp.items(), key=lambda kv: (kv[0][0], -kv[1]['n'][0])):
    print('| %s %s | %s | %s | %.3f | %s | %s |' % (cat, sub, ' / '.join(str(x) for x in g['n']), (' / '.join(str(x) for x in g['rst']) + (' (worst %.3f)' % g['rworst'] if g['rst'][0] else '')), g['worst'], ' lv, '.join(str(x) for x in g['wp'][:2]) + ', net ' + str(g['wp'][2]), ', '.join('%s(%d)' % kv for kv in g['fam'].most_common(3))))
rtot = [sum(g['rst'][i] for g in grp.values()) for i in range(len(T))]
print('| **all kernel-clock endpoints** | %s | %s | | | |' % (' / '.join(str(x) for x in tot), ' / '.join(str(x) for x in rtot)))
by = collections.defaultdict(lambda: [0] * len(T))
for (cat, sub), g in grp.items():
    for i in range(len(T)):
        by[cat][i] += g['n'][i]
for cat in 'ABCD':
    if cat in by:
        print('| group %s total | %s | | | | |' % (cat, ' / '.join(str(x) for x in by[cat])))


def detail(path, groups, topn=5):
    """second pass: the top start-family -> end-family pairs of the named groups (substring of the group name)."""
    pairs = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 9.9, None]))
    for r in csv.DictReader(open(path)):
        if r['endclk'] != KC:
            continue
        s = float(r['slack'])
        if s >= T[0]:
            continue
        cat, sub = classify(r['start'], r['end'])
        name = cat + ' ' + sub
        for g in groups:
            if g in name:
                k = (norm('/'.join(r['start'].split('/')[-5:-1])), fam(r['end']))
                p = pairs[name][k]
                p[0] += 1
                if s < p[1]:
                    p[1] = s
                    p[2] = (r['levels'], r['delay'], r['net'])
    for name, d in pairs.items():
        print('\n== %s' % name)
        for k, p in sorted(d.items(), key=lambda kv: -kv[1][0])[:topn]:
            print('  %6d  worst %.3f  (%s lv, %s ns, net %s)  %s -> %s' % (p[0], p[1], p[2][0], p[2][1], p[2][2], k[0], k[1]))


if len(sys.argv) > 3 and sys.argv[3] == 'detail':
    detail(path, sys.argv[4:])
