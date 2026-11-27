#!/usr/bin/env python3
"""census_table.py ENDPOINTS_CSV -- per unit type (the three tiles summed) the kernel-clock endpoints of a routed census with slack
below 0.5, 1.0 and 2.0 ns (all of them are in the file: it holds every endpoint below 2.44 ns), the worst slack and its path
(levels, datapath ns, net share). Slack s at the 10 ns constraint means a routed path of about 10 - s ns."""
import csv, re, sys, collections
tile_re = re.compile(r'g_tile\[(\d)\]\.u_tile/(.*)')
def unit(pin):
    m = tile_re.search(pin)
    if not m:
        if 'hmss' in pin or 'blp' in pin or 'ulp_ucs' in pin or 'ii_level0' in pin: return 'platform'
        k = pin.split('tnca_u50_1/inst/')[-1].split('/')
        return 'kernel:' + (k[0] if len(k) > 1 else '(regs)')
    rest = m.group(2).split('/')
    UNITS = ('g_lsu_hybrid.u_lsu', 'g_vrf_banked.u_vrf', 'u_mtx', 'u_vfpu', 'u_valu', 'u_txf', 'u_bar', 'u_srf', 'u_mul', 'u_prf', 'u_salu', 'u_dec', 'u_mmux', 'u_pfdec')
    if rest[0] == 'u_core':
        return 'core/' + rest[1] if len(rest) > 2 and rest[1] in UNITS else 'core control/pipeline regs'
    return rest[0] if rest[0].startswith(('g_home_pipe', 'u_cni', 'g_host_mux')) else 'tile (other)'
stat = collections.defaultdict(lambda: [0, 0, 0, 9.0, None])
for r in csv.DictReader(open(sys.argv[1])):
    if not ('kernel' in r['startclk'] and 'kernel' in r['endclk']): continue
    s = float(r['slack']); u = unit(r['end']); st = stat[u]
    if s < 0.5: st[0] += 1
    if s < 1.0: st[1] += 1
    if s < 2.0: st[2] += 1
    if s < st[3]: st[3] = s; st[4] = (r['levels'], float(r['delay']), 100 * float(r['net']) / max(float(r['delay']), 1e-9))
print('%-26s %7s %7s %8s %8s  %s' % ('unit (3 tiles summed)', '<0.5ns', '<1.0ns', '<2.0ns', 'worst', 'worst path: levels, datapath ns, net share'))
tot = [0, 0, 0]
for u, st in sorted(stat.items(), key=lambda kv: -kv[1][2]):
    if st[2] == 0 and st[3] > 2.0: continue
    print('%-26s %7d %7d %8d %8.3f  %s lvl, %.2f ns, net %.0f%%' % (u[:26], st[0], st[1], st[2], st[3], st[4][0], st[4][1], st[4][2]))
    for i in range(3): tot[i] += st[i]
print('%-26s %7d %7d %8d' % ('total', *tot))
