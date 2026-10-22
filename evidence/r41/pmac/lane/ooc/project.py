#!/usr/bin/env python3
"""project.py SUMMARY.json... -- lane OOC results -> per-core / three-core / per-SLR projection.

Lane area is post-route, the u_lane instance alone (pmac_ooc_top's w_*/y_q
registers excluded). Per core: 16 lanes (tnca_matrix), plus an ESTIMATE of the
matrix-level changes a DP opcode needs (B row latch 16 -> 32 bits per lane:
+256 FF; 32-bit A word select and sequencing: ~+200 LUT) -- not synthesised.
Three cores, one per SLR (regional_preflight.json placement); each SLR's delta
is added to the Stage 1 regional preflight estimate (A3 routed + M1-M4 + F07)
and CLB packing is scaled linearly with LUTs, as the preflight does (which
understates packing nonlinearity).
Timing: worst routed stage datapath per configuration, and an in-chip
indication = OOC routed x 1.54, the ratio of V0's matrix stage 0 projected
in-chip (F03: 9.48 ns) to its OOC routed delay here (6.162 ns).
"""
import json, sys

PRE = json.load(open(__file__.rsplit('/', 4)[0] + '/stage1/regional_preflight.json'))
MATRIX_EXTRA = {'luts': 200, 'ff': 256}
RATIO = 9.48 / 6.162

rows = []
for p in sys.argv[1:]:
    rows += json.load(open(p))
by = {r['tag']: r for r in rows if 'error' not in r}
v0 = by['V0_p10']['route']['area']

def worst(r):
    st = r['route']['stages']
    w = max(((float(v['dp']), k, v['lv']) for k, v in st.items() if v['n'] and v['dp'] != 'NA' and k != 'in_reg'))
    return w

out = {'v0_lane': v0, 'ratio_in_chip': RATIO, 'matrix_extra_estimate': MATRIX_EXTRA, 'configs': []}
print('| config | regs | latency | LUT | FF | DSP | CARRY8 | routed WNS @period | worst stage (ns, levels) | x1.54 |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
for tag in sorted(by):
    r = by[tag]; a = r['route']['area']
    cuts = tag.split('_c')[1].split('_')[0] if '_c' in tag else '1100'
    regs = cuts.count('1'); per = tag.rsplit('_p', 1)[1]
    w = worst(r)
    out['configs'].append(dict(tag=tag, regs=regs, latency=1 + regs, area=a, carry8=r['route']['carry8'],
                               wns=r['route']['wns'], wns_synth=r['synth']['wns'], worst_stage=w[1],
                               worst_dp=w[0], levels=w[2], in_chip=round(w[0] * RATIO, 2)))
    print(f"| {tag} | {regs} | {1 + regs} | {a['luts']} | {a['ff']} | {a['dsp']} | {r['route']['carry8']} | "
          f"{r['route']['wns']} @{per} | {w[1]} {w[0]:.2f} ({w[2]}) | {w[0] * RATIO:.2f} |")

print('\n| config | dLUT/lane | dFF/lane | dDSP/lane | dLUT/core | dLUT x3 | SLR0 LUT % | SLR0 CLB packing (scaled) | SLR1 packing | SLR2 packing |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for c in out['configs']:
    if not c['tag'].endswith('_p10') or c['tag'].startswith('V0'):
        continue
    a = c['area']
    d = dict(luts=a['luts'] - v0['luts'], ff=a['ff'] - v0['ff'], dsp=a['dsp'] - v0['dsp'])
    core = dict(luts=16 * d['luts'] + MATRIX_EXTRA['luts'], ff=16 * d['ff'] + MATRIX_EXTRA['ff'], dsp=16 * d['dsp'])
    slr = {}
    for s in ('SLR0', 'SLR1', 'SLR2'):
        e = PRE['slr'][s]['estimate']; base = PRE['slr'][s]['a3_routed']
        luts = e['luts'] + core['luts']
        slr[s] = dict(luts=luts, lut_percent=round(luts / (base['luts'] / base['lut_percent']), 2),
                      clb_packing_scaled=round(base['clb_packing_percent'] * luts / base['luts'], 2))
    c.update(delta_lane=d, delta_core=core, delta_3cores={k: 3 * v for k, v in core.items()}, slr=slr)
    print(f"| {c['tag']} | {d['luts']:+d} | {d['ff']:+d} | {d['dsp']:+d} | {core['luts']:+d} | {3 * core['luts']:+d} | "
          f"{slr['SLR0']['lut_percent']} | {slr['SLR0']['clb_packing_scaled']} | {slr['SLR1']['clb_packing_scaled']} | "
          f"{slr['SLR2']['clb_packing_scaled']} |")
json.dump(out, open(__file__.rsplit('/', 1)[0] + '/projection.json', 'w'), indent=1)
