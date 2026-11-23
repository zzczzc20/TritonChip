#!/usr/bin/env python3
"""make_tables.py EVIDENCE_DIR -- builds evidence/r42/kfuse_rtl/tables.md from the raw files in EVIDENCE_DIR:
  ooc/<run>/{synth,route}_{util,util_hier,summary}.rpt[.gz], {synth,route}_endpoints.csv[.gz]   (Vivado, x86 host)
  sim/*.log, sim/*.gen.log, sim/mutation_check.log                                               (iverilog, GB10)
Every number in the tables is read from those files; the only constants typed here are the recorded R41 anchors and
the SLR0 packing figures quoted from doc/r42_arch_headroom.md sec 1.3/1.4 (named below with their sources)."""
import csv
import glob
import gzip
import os
import re
import sys

EV = sys.argv[1]

# recorded anchors (source: origin/R42-arch:doc/r42_arch_headroom.md sec 1.4 and 1.3; evidence/r41/hbm_bw/RESOURCES.md)
R41_SHARED_REC = 2792          # LUT per lane, recorded: stages 689+639+448+526+490 (fma_dslx/README.md)
R41_SHARED_DSP = 8
V3_LUT, V3_DSP = 2929, 2      # V3 E4M3 exact 38-bit field, G=4 (pmac/README.md:131)
SLR0_PACK, SLR0_MAX, SLR0_LUT_PER_CLB, SLR0_HEADROOM_CLB = 95.56, 96.74, 6.13, 649
SLR0_CLB = SLR0_HEADROOM_CLB / ((SLR0_MAX - SLR0_PACK) / 100.0)     # CLBs in SLR0 implied by the two packing figures
SLR_HEAD_LUT = {'SLR0': 3975, 'SLR1': 22292, 'SLR2': 28133}


def rd(path):
    for p in (path, path + '.gz'):
        if os.path.exists(p):
            return (gzip.open(p, 'rt', errors='replace') if p.endswith('.gz') else open(p, errors='replace')).read()
    return None


def util(run, stage, base='ooc'):
    t = rd(os.path.join(EV, base, run, stage + '_util.rpt'))
    if t is None:
        return None
    def g(pat):
        m = re.search(pat, t)
        return int(m.group(1)) if m else None
    return dict(lut=g(r'\| CLB LUTs\*?\s+\|\s+(\d+)'), ff=g(r'\| CLB Registers\s+\|\s+(\d+)'),
                dsp=g(r'\| DSPs\s+\|\s+(\d+)'), c8=g(r'\| CARRY8\s+\|\s+(\d+)'))


def wns(run, stage, base='ooc'):
    t = rd(os.path.join(EV, base, run, stage + '_summary.rpt'))
    if t is None:
        return None
    lines = t.splitlines()
    for i, l in enumerate(lines):
        if 'WNS(ns)' in l:
            v = lines[i + 2].split()
            return float(v[0])
    return None


def hier(run):
    """back end / per product / max tree / glue from a -flatten_hierarchy none synthesis"""
    t = rd(os.path.join(EV, 'ooc', run, 'synth_util_hier.rpt'))
    if t is None:
        return None
    rows = {}
    for l in t.splitlines():
        if l.startswith('|') and not l.startswith('|--') and 'Instance' not in l:
            c = [x.strip() for x in l.strip('|').split('|')]
            if len(c) >= 10:
                rows[c[0]] = (int(c[2]), int(c[6]), int(c[-1]))
    def lut(name):
        return rows[name][0] if name in rows else 0
    prods = [v[0] for k, v in rows.items() if re.fullmatch(r'g_p\[\d+\]\.u_prod', k)]
    back = lut('u_b0') + lut('u_b1') + lut('u_b2')
    tot = rows.get('u_lane', (0, 0, 0))
    return dict(back=back, nprod=len(prods), prod=sum(prods), maxt=lut('u_max'), glue=lut('(u_lane)'), total=tot[0],
                prod_avg=(sum(prods) / len(prods)) if prods else 0)


def stages(run, stage='route'):
    import importlib.util
    t = rd(os.path.join(EV, 'ooc', run, stage + '_endpoints.csv'))
    if t is None:
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    S = [
        ('c0 decode regs', r'g_p\[\d+\]\.u_prod/(ma_q|mb_q|x_q|sg_q)'),
        ('DSP1 P', r'u_prod/p1_q'),
        ('T regs', r'u_prod/g_t\.t_r'),
        ('Tmax tree', r'u_max/'),
        ('DSP2 operands/P', r'u_prod/g_s[01]\.'),
        ('adder tree', r'u_tree/'),
        ('pkt reg', r'g_pk\.q'),
        ('B0 reg', r'g_b0\.q'),
        ('B1 reg', r'g_b1\.q'),
        ('y_q', r'^(y_q|valid_q)'),
    ]
    best = {}
    for r in csv.DictReader(t.splitlines()):
        name = None
        for n, p in S:
            if re.search(p, r['end']):
                name = n
                break
        if name is None:
            continue
        s = float(r['slack'])
        if name not in best or s < best[name]:
            best[name] = s
    return [(n, best.get(n)) for n, _ in S]


def f(x, fmt='%d'):
    return '-' if x is None else fmt % x


def lane_row(label, run, g, macs, c_note=''):
    u = util(run, 'route')
    if u is None:
        return None
    w = wns(run, 'route')
    return '| %s | %s | %s | %s | %s | %s | %s | %s |' % (
        label, f(u['lut']), f(u['ff']), f(u['dsp']), f(w, '%+.3f'), f(u['lut'] / macs, '%.0f'), f(u['dsp'] / macs, '%.2f'), c_note)


out = []
P = out.append

# ------------------------------------------------------------------ table 1: lanes
P('### T1. One lane, out-of-context place and route at 10 ns (xcu280, Vivado 2023.2, every port registered)\n')
P('| lane | LUT | FF | DSP48E2 | WNS (ns) | LUT per MAC/cycle | DSP per MAC/cycle | note |')
P('|---|---:|---:|---:|---:|---:|---:|---|')
rows = [
    ('K-fused E4M3 G=4, W=0', 'p2_g4', 4, 'C_LATE=1'),
    ('K-fused E4M3 G=8, W=0', 'p2_g8', 8, 'C_LATE=1'),
    ('K-fused E4M3 G=16, W=0', 'p2_g16', 16, 'C_LATE=1'),
    ('K-fused E4M3 G=16, W=0, c delayed inside', 'p2_g16_c0', 16, 'C_LATE=0'),
    ('R41 shared lane, mode tied to DP4 (E4M3, 4 products)', 'p_mac4', 4, 'measured here'),
    ('R41 shared lane, all modes (FP32, DP2, DP4)', 'p_mac', 4, 'measured here, DP4 counted'),
]
for lab, run, g, nt in rows:
    r = lane_row(lab, run, g, g, nt)
    if r:
        P(r)
P('| R41 shared lane, recorded (evidence/r41/hbm_bw/fma_dslx/README.md) | %d | - | %d | - | %.0f | %.2f | recorded, 4 products |' %
  (R41_SHARED_REC, R41_SHARED_DSP, R41_SHARED_REC / 4, R41_SHARED_DSP / 4))
P('| V3 E4M3 exact 38-bit field, recorded (pmac/README.md) | %d | - | %d | - | %.0f | %.2f | recorded, 4 products |' %
  (V3_LUT, V3_DSP, V3_LUT / 4, V3_DSP / 4))
P('')

# ------------------------------------------------------------------ table 2: arrays
P('### T2. A 16-lane array (independent a and b per lane), same flow\n')
P('| array | LUT | FF | DSP48E2 | WNS (ns) | MAC/cycle | LUT per MAC/cycle | vs 16 x 2,792 (recorded R41 array) |')
P('|---|---:|---:|---:|---:|---:|---:|---:|')
arr = {}
for g in (4, 8, 16):
    run = 'p2_arr%d' % g
    u = util(run, 'route')
    if u:
        arr[g] = u
        d = u['lut'] - 16 * R41_SHARED_REC
        P('| 16 x K-fused G=%d | %d | %d | %d | %+.3f | %d | %.0f | %+d LUT (%+.1f %%) |' % (
            g, u['lut'], u['ff'], u['dsp'], wns(run, 'route'), 16 * g, u['lut'] / (16 * g), d, 100.0 * d / (16 * R41_SHARED_REC)))
m4 = util('p_mac4', 'route')
mm = util('p_mac', 'route')
if m4:
    P('| 16 x R41 shared lane, DP4 tied (16 x lane, linear) | %d | %d | %d | - | 64 | %.0f | %+d LUT |' % (
        16 * m4['lut'], 16 * m4['ff'], 16 * m4['dsp'], 16 * m4['lut'] / 64, 16 * m4['lut'] - 16 * R41_SHARED_REC))
if mm:
    P('| 16 x R41 shared lane, all modes (16 x lane, linear) | %d | %d | %d | - | 64 | %.0f | %+d LUT |' % (
        16 * mm['lut'], 16 * mm['ff'], 16 * mm['dsp'], 16 * mm['lut'] / 64, 16 * mm['lut'] - 16 * R41_SHARED_REC))
P('| 16 x R41 shared lane, recorded | %d | - | %d | - | 64 | %.0f | 0 |' % (
    16 * R41_SHARED_REC, 16 * R41_SHARED_DSP, 16 * R41_SHARED_REC / 64))
P('')
lin = []
for g in (4, 8, 16):
    u = util('p2_g%d' % g, 'route')
    if u and g in arr:
        lin.append('G=%d: array %d LUT, 16 x lane %d LUT (%+.2f %%)' % (g, arr[g]['lut'], 16 * u['lut'], 100.0 * (arr[g]['lut'] - 16 * u['lut']) / (16 * u['lut'])))
if lin:
    P('Linearity check, array against 16 x its lane: ' + '; '.join(lin) + '.\n')

# ------------------------------------------------------------------ table 3: timing
P('### T3. Clock: G=16 lane, place and route at three periods (OOC, ideal clock, no SLR crossing)\n')
P('| period (ns) | WNS (ns) | slack-implied period (ns) | critical endpoint |')
P('|---:|---:|---:|---|')
for run, per in (('p2_g16', 10), ('p2_g16_t5', 5), ('p2_g16_t4', 4), ('p2_g8_t4', 4)):
    w = wns(run, 'route')
    if w is None:
        continue
    t = rd(os.path.join(EV, 'ooc', run, 'route_endpoints.csv'))
    ep = ''
    if t:
        rows_ = list(csv.DictReader(t.splitlines()))
        if rows_:
            r0 = min(rows_, key=lambda r: float(r['slack']))
            ep = '%s -> %s' % (r0['start'].split('/')[-2] if '/' in r0['start'] else r0['start'], r0['end'].rsplit('/', 1)[0])
    P('| %d (%s) | %+.3f | %.3f | `%s` |' % (per, run, w, per - w, ep[:90]))
P('')
st = {g: stages('p2_g16' if g == 10 else 'p2_g16_t5') for g in (10, 5)}
if st[10] and st[5]:
    P('Worst slack into each pipeline boundary of the G=16 lane (ns), routed:\n')
    P('| boundary | at 10 ns | at 5 ns |')
    P('|---|---:|---:|')
    for (n, a), (_, b) in zip(st[10], st[5]):
        P('| %s | %s | %s |' % (n, f(a, '%.3f'), f(b, '%.3f')))
    P('')

# ------------------------------------------------------------------ table 4: where the LUTs are
P('### T4. Where the LUTs are (post-synthesis, hierarchy kept, C_LATE=1)\n')
P('| G | lane LUT (synth) | R41 back end (B0+B1+B2) | per product (decode, DSP1/DSP2 support, T, shift) | Tmax tree | glue (abs S, pack, flags) |')
P('|---:|---:|---:|---:|---:|---:|')
H = {}
for g in (4, 8, 16):
    h = hier('h2_g%d' % g)
    if h:
        H[g] = h
        P('| %d | %d | %d (%.0f %%) | %d (%.1f / product) | %d | %d |' % (
            g, h['total'], h['back'], 100.0 * h['back'] / h['total'], h['prod'], h['prod_avg'], h['maxt'], h['glue']))
P('')

# ------------------------------------------------------------------ table 5: variants
P('### T5. Variants of the G=16 lane (place and route at 10 ns)\n')
P('| variant | LUT | FF | DSP48E2 | WNS (ns) |')
P('|---|---:|---:|---:|---:|')
for lab, run in (('baseline (SIGN_MODE 1, DSP adder tree, W=0, CUTS 0b111011)', 'p2_g16'),
                 ('SIGN_MODE 0 (add/subtract form)', 'p2_g16_sm0'),
                 ('TREE 1 (LUT adder tree)', 'p2_g16_tree1'),
                 ('W = 3', 'p2_g16_w3'),
                 ('CUTS = 0 (no optional registers)', 'p2_g16_cuts0'),
                 ('C_LATE = 0 (c delayed inside the lane)', 'p2_g16_c0')):
    u = util(run, 'route')
    if u:
        P('| %s | %d | %d | %d | %+.3f |' % (lab, u['lut'], u['ff'], u['dsp'], wns(run, 'route')))
P('')

# ------------------------------------------------------------------ table 6: per-core projection
P('### T6. Per-core projection (derived from T1/T2/T4 and the recorded SLR0 figures; not an image)\n')
P('Constants: current array = 16 x %d LUT (recorded); SLR0 packed %.2f %%, routed up to %.2f %%, %.2f LUT per CLB, %d CLB of headroom, hence %.0f CLB in SLR0; headroom in LUT: SLR0 %d, SLR1 %d, SLR2 %d (doc/r42_arch_headroom.md sec 1.4).\n' % (
    R41_SHARED_REC, SLR0_PACK, SLR0_MAX, SLR0_LUT_PER_CLB, SLR0_HEADROOM_CLB, SLR0_CLB, SLR_HEAD_LUT['SLR0'], SLR_HEAD_LUT['SLR1'], SLR_HEAD_LUT['SLR2']))
P('| option | G | LUT change per core | DSP change per core | SLR0 packing after | fits SLR1 / SLR2 / SLR0 headroom | LUT to remove in SLR0 to fit |')
P('|---|---:|---:|---:|---:|---|---:|')
cur_lut = 16 * R41_SHARED_REC
cur_dsp = 16 * R41_SHARED_DSP
def fits(d):
    return ' / '.join('yes' if d <= SLR_HEAD_LUT[s] else 'no' for s in ('SLR1', 'SLR2', 'SLR0'))
def pack(d):
    return SLR0_PACK + 100.0 * (d / SLR0_LUT_PER_CLB) / SLR0_CLB
for g in (8, 16):
    if g not in arr:
        continue
    a = arr[g]
    d1 = a['lut'] - cur_lut
    P('| (a) E4M3 K-fused array replaces the shared-lane array (FP32, FP16/BF16/E5M2 modes would have to live elsewhere) | %d | %+d | %+d | %.2f %% | %s | %d |' % (g, d1, a['dsp'] - cur_dsp, pack(d1), fits(d1), max(0, d1 - SLR_HEAD_LUT['SLR0'])))
    d2 = a['lut']
    P('| (b) K-fused array added beside the shared-lane array | %d | %+d | %+d | %.2f %% | %s | %d |' % (g, d2, a['dsp'], pack(d2), fits(d2), max(0, d2 - SLR_HEAD_LUT['SLR0'])))
    if g in H and g in arr:
        lane = util('p2_g%d' % g, 'route')
        front = lane['lut'] - H[g]['back'] * lane['lut'] / H[g]['total']    # route-scaled front-end share of the lane
        d3 = int(round(16 * front))
        P('| (c) K-fused front end only, feeding the shared lane\'s back end (front-end share of the routed lane x 16; the mode mux in front of B0 is not counted) | %d | %+d | %+d | %.2f %% | %s | %d |' % (g, d3, a['dsp'], pack(d3), fits(d3), max(0, d3 - SLR_HEAD_LUT['SLR0'])))
P('')

# ------------------------------------------------------------------ table 5b: first RTL
P('### T5b. First RTL (commits a13994fc and f891bab0, same logic: A gated by the shift test; same flow), for reference\n')
P('| run | LUT | FF | DSP48E2 | WNS (ns) |')
P('|---|---:|---:|---:|---:|')
for lab, run in (('lane G=4', 'p_g4'), ('lane G=8', 'p_g8'), ('lane G=16', 'p_g16'), ('lane G=16 at 5 ns', 'p5_g16'),
                 ('array 16 x G=4', 'p_arr4'), ('array 16 x G=8', 'p_arr8'), ('array 16 x G=16', 'p_arr16')):
    u = util(run, 'route', 'ooc_v1')
    if u:
        P('| %s | %d | %d | %d | %+.3f |' % (lab, u['lut'], u['ff'], u['dsp'], wns(run, 'route', 'ooc_v1')))
P('')

# ------------------------------------------------------------------ table 6b: equal MAC rate
P('### T6b. LUT and DSP for the same E4M3 MAC rate as today (64 MAC/cycle = 16 lanes x 4), lanes only, linear in lane count\n')
P('| design | lanes | LUT | DSP48E2 | note |')
P('|---|---:|---:|---:|---|')
if m4:
    P('| R41 shared lane, DP4 tied (measured) | 16 | %d | %d | the DP4 path alone, no FP32/DP2 mode |' % (16 * m4['lut'], 16 * m4['dsp']))
P('| R41 shared lane, recorded | 16 | %d | %d | all modes |' % (16 * R41_SHARED_REC, 16 * R41_SHARED_DSP))
for g in (8, 16):
    u = util('p2_g%d' % g, 'route')
    if u:
        n = 64 // g
        P('| K-fused G=%d | %d | %d | %d | the engine geometry (16 columns per row cycle) would change; not designed here |' % (g, n, n * u['lut'], n * u['dsp']))
P('')

# ------------------------------------------------------------------ table 7: simulation
P('### T7. RTL simulation against the model (iverilog 12 on GB10; expected values from kfuse_model.scalar_step)\n')
P('| run | G | W | variant | vectors | mismatches |')
P('|---|---:|---:|---|---:|---:|')
tot_v = 0
tot_m = 0
for lf in sorted(x for x in glob.glob(os.path.join(EV, 'sim', 'f_*.log')) if not x.endswith(('.compile.log', '.gen.log'))):
    t = open(lf, errors='replace').read()
    hd = re.search(r'tb_kfuse_lane G=(\d+) W=(\d+) CUTS=(\d+) IN_REG=(\d+) C_LATE=(\d+) SIGN_MODE=(\d+) TREE=(\d+) LAT=(\d+) LAT_FRONT=(\d+) N=(\d+) bubble=(\d+)%', t)
    rs = re.search(r'RESULT G=(\d+) W=(\d+) total=(\d+) mismatches=(\d+)', t)
    if not hd or not rs:
        P('| %s | | | NOT COMPLETE | | |' % os.path.basename(lf)[:-4])
        continue
    g, w, cuts, inreg, clate, sm, tree, lat, latf, n, bub = hd.groups()
    var = 'CUTS=%s IN_REG=%s C_LATE=%s SIGN_MODE=%s TREE=%s LAT=%s bubbles=%s%%' % (cuts, inreg, clate, sm, tree, lat, bub)
    P('| %s | %s | %s | %s | %s | %s |' % (os.path.basename(lf)[:-4], g, w, var, rs.group(3), rs.group(4)))
    tot_v += int(rs.group(3))
    tot_m += int(rs.group(4))
P('\nTotal: %d vectors, %d mismatches.\n' % (tot_v, tot_m))

mc = os.path.join(EV, 'sim', 'mutation_check.log')
if os.path.exists(mc):
    P('### T8. Mutation check of the bench (G=4, W=0, %s)\n' % 'directed prefix + 100,000 random vectors')
    P('```')
    P(open(mc).read().rstrip())
    P('```\n')

nls = sorted(glob.glob(os.path.join(EV, 'netlist', '*.sim.log')) + glob.glob(os.path.join(EV, 'netlist', '*.sim.clean.log')))
if nls:
    P('### T9. Post-synthesis functional netlist (Vivado write_verilog -mode funcsim, unisim models) against the same vectors\n')
    P('| netlist | G | W | vectors | mismatches |')
    P('|---|---:|---:|---:|---:|')
    for lf in nls:
        t = open(lf, errors='replace').read()
        hd = re.search(r'tb_kfuse_lane G=(\d+) W=(\d+)', t)
        rs = re.search(r'RESULT G=(\d+) W=(\d+) total=(\d+) mismatches=(\d+)', t)
        P('| %s | %s | %s | %s | %s |' % (os.path.basename(lf).split('.')[0], rs.group(1) if rs else '', rs.group(2) if rs else '',
                                         rs.group(3) if rs else 'NOT COMPLETE', rs.group(4) if rs else ''))
    P('')

open(os.path.join(EV, 'tables.md'), 'w').write('\n'.join(out) + '\n')
print('wrote', os.path.join(EV, 'tables.md'))
