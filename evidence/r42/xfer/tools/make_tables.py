#!/usr/bin/env python3
"""Every table of evidence/r42/xfer, computed from the raw files beside it (run from evidence/r42/xfer):

    python3 tools/make_tables.py > tables.txt

Domains: tlm = SystemC/TLM model, rtl-1c = single-core RTL simulation, rtl-3c = three-core board-top RTL stand-in,
rtl-eng = the transfer engine alone behind a fixed-latency memory, board = the existing ab-int2 records (no board run here)."""
import collections, glob, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)) + '/..'


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float('nan')


def rtl_cycles(path):
    r = json.loads(open(path).readline())
    t = r.get('time') or {}
    return (t.get('device') if r.get('status') == 'pass' else None), r.get('status')


print('1. tlm (config cal): cycles of the 24 GEMM-shaped points of tiny/small/medium, and against the board')
tl = json.load(open(HERE + '/tlm/collected_gemm24.json'))
board = json.load(open(HERE + '/tlm/board_ab_int2_points.json'))
base, auto = tl['base'], tl['auto12']
print('   base = TNCA_GEMM_AUTO=0 semantics (16x16x16 tile) on the R42 head; auto12 = the committed default (N blocking, >= 12 programs)')
print('%-58s %11s %11s %11s %8s %8s' % ('point', 'board', 'tlm base', 'tlm auto', 'auto/base', 'base/board'))
rs = []; rb = []
for p in sorted(base):
    b, a = base[p][0], auto[p][0]
    rs.append(a / b); rb.append(b / board[p])
    print('%-58s %11d %11d %11d %8.3f %8.3f' % (p, board[p], b, a, a / b, b / board[p]))
print('geometric mean auto/base %.3f (speed-up %.3f); model/board of the base %.3f; max_rel_err identical at %d of %d points'
      % (gm(rs), 1 / gm(rs), gm(rb), sum(base[p][2] == auto[p][2] for p in base), len(base)))

print('\n2. tlm (config cal): the large and hbm tiers through reduced-grid proxies (M or batch reduced, K, N, epilogue kept)')
px = json.load(open(HERE + '/tlm/collected_proxies_large.json'))
rs = []
print('%-58s %12s %12s %8s' % ('proxy point', 'base', 'auto', 'ratio'))
for p in sorted(px['px_base']):
    b, a = px['px_base'][p], px['px_auto'][p]
    if b[1] == 'pass' and a[1] == 'pass':
        rs.append(a[0] / b[0]); print('%-58s %12d %12d %8.3f   max_rel_err equal: %s' % (p, b[0], a[0], a[0] / b[0], b[2] == a[2]))
print('geometric mean auto/base %.3f (speed-up %.3f)' % (gm(rs), 1 / gm(rs)))
print('   cycles per 16x16x16 matrix command per core (three cores) of the op.* proxies; the board records of the full points are 487-605')
print('   (op.matmul/large 556, op.linear/large 605, op.linear/hbm 487, op.matmul/hbm 553: ab-int2 cycles x 3 / commands of board_engine_utilisation.txt)')
for p in sorted(px['px_base']):
    m = re.match(r'op\.(?:matmul|linear)/large/fp16-m(\d+)-k(\d+)-n(\d+)$', p)
    if m:
        mm, kk, nn = map(int, m.groups()); cmds = (mm // 16) * (kk // 16) * (nn // 16) / 3
        print('   %-52s base %6.0f   auto %6.0f' % (p, px['px_base'][p][0] / cmds, px['px_auto'][p][0] / cmds))

print('\n3. rtl-1c (profile core-vrf-sync-v5-x8-g2c4-dp, default knobs): cycles, base = TNCA_GEMM_AUTO=0')
print('%-48s %10s %10s %8s %10s %8s' % ('point', 'base', 'auto', 'ratio', 'K32 tile', 'ratio'))
for f in sorted(glob.glob(HERE + '/rtl_1c/rtl_base/*.jsonl')):
    n = os.path.basename(f)
    b = rtl_cycles(f)[0]
    a = rtl_cycles(HERE + '/rtl_1c/rtl_auto/' + n)[0] if os.path.exists(HERE + '/rtl_1c/rtl_auto/' + n) else None
    k = rtl_cycles(HERE + '/rtl_1c/k32/' + n)[0] if os.path.exists(HERE + '/rtl_1c/k32/' + n) else None
    print('%-48s %10s %10s %8s %10s %8s' % (n[:-6], b, a, '%.3f' % (a / b) if a else '-', k or '-', '%.3f' % (k / b) if k else '-'))

print('\n4. rtl-3c (three-core board-top stand-in at the R41 run-2 image geometry, line engine): cycles of the timed call')
print('   base_tiny/k32_tiny: 16x16x16 vs TNCA_GEMM_TILE=16,16,32. base0_small = TNCA_GEMM_AUTO=0 (the 16x16x16 baseline); auto_small_run1 = the default')
print('   (N blocking) and bn32_small_explicit = TNCA_GEMM_TILE=16,32,16: the same kernel, so their difference is the run-to-run noise of the stand-in')
for d in ('base_tiny', 'k32_tiny', 'base0_small', 'auto_small_run1', 'bn32_small_explicit'):
    for f in sorted(glob.glob(HERE + '/rtl_3c/%s/*.jsonl' % d)):
        print('   %-12s %-48s %s' % (d, os.path.basename(f)[:-6], rtl_cycles(f)[0]))

print('\n5. tlm K32 and hardware what-ifs (cycles / base cycles; FAIL = compile failure of the K32 tile)')
tags = [t for t in ('auto12', 'k32', 'ab_k32', 'le_base', 'le_auto', 'le_k32', 'leab_k32') if t in tl]
print('%-58s' % 'point' + ''.join(' %9s' % t[:9] for t in tags))
cols = collections.defaultdict(list)
for p in sorted(base):
    row = '%-58s' % p
    for t in tags:
        v = tl[t].get(p)
        if v is None or v[1] != 'pass':
            row += ' %9s' % 'FAIL'
        else:
            r = v[0] / base[p][0]; cols[t].append(r); row += ' %9.3f' % r
    print(row)
print('%-58s' % 'geometric mean (points that compiled)' + ''.join(' %9.3f' % gm(cols[t]) for t in tags))
print('   k32 = TNCA_GEMM_TILE=16,16,32 (this branch, gemm_kmerge); ab_k32 adds mtx_abuf_k32=1 (A buffer of 16 registers, WHAT-IF hardware);')
print('   le_* = config cal_le (line engine with the rtl eligibility rule and the transposed line buffer; NOT validated: it predicts')
print('   0.59-0.67 of the cal cycles on the tiny points where rtl-3c agrees with cal); leab_k32 = cal_le + A buffer of 16.')

print('\n6. rtl-eng: tnca_tensor.v alone, ideal memory of fixed latency, cycles per K step of KS elements of ONE operand')
d = collections.defaultdict(dict)
for l in open(HERE + '/rtl_microbench/xp_per_operand.txt'):
    m = re.search(r'KS=(\d+) XB=(\d+) XM=(\d+) XN=(\d+) fixlat=(\d+) word_engine_cycles=(\d+) line_engine_cycles=(\d+)', l)
    if m:
        ks, xb, xm, xn, lat, w, ln = map(int, m.groups()); d[(ks, xb, xm, lat)][xn] = (w, ln)
names = {(2, 1): 'A tile 16 x KS (plain)', (0, 0): 'W^T tile (transposed)', (1, 0): 'B tile KS x 16 (plain)'}
print('%-26s %-4s' % ('operand', 'KS') + ''.join('  lat=%-2d word/line' % l for l in (12, 24, 40, 60)))
for (xb, xm), nm in names.items():
    for ks in (16, 32):
        row = '%-26s %-4d' % (nm, ks)
        for lat in (12, 24, 40, 60):
            e = d.get((ks, xb, xm, lat), {})
            row += '   %5.0f / %-5.0f  ' % ((e[8][0] - e[4][0]) / 4, (e[8][1] - e[4][1]) / 4) if 4 in e and 8 in e else '   ---   '
        print(row)
print('   word = the word engine (single-core device, TENSOR_LINE 0); line = the line engine (board image, TENSOR_LINE 1).')

print('\n7. census of one K step (rtl-1c event logs, tools/census.py)')
for f in ('base_linear_tiny', 'k32_linear_tiny', 'matmul_small_base', 'matmul_small_auto'):
    print('  ', f)
    out = subprocess.run([sys.executable, HERE + '/tools/census.py', HERE + '/census/%s.probe' % f], capture_output=True, text=True).stdout
    print(''.join('     ' + l + '\n' for l in out.splitlines()))
    s = [l for l in open(HERE + '/census/%s.probe' % f) if l.startswith('S ')]
    if s: print('      launch summary:', s[-1].strip())

print('\n8. tlm HARDWARE what-if (not built): rows of 32 bytes (half a line) move as line requests (dma_line_elig=2), against the engine as built')
print('   (dma_line_elig=1, the le_elig rule of tnca_tensor.v); both with the default compiler (auto tile), no transposed line buffer. cal_le is NOT')
print('   validated (section 5 note), so the ratio is an expectation of the direction and the order of size, not a prediction of board cycles.')
if 'le1_auto' in tl and 'le2_auto' in tl:
    rs = []
    print('%-58s %10s %10s %8s' % ('point', 'elig=1', 'elig=2', 'ratio'))
    for p in sorted(base):
        a, b2 = tl['le1_auto'][p], tl['le2_auto'][p]
        if a[1] == 'pass' and b2[1] == 'pass':
            rs.append(b2[0] / a[0]); print('%-58s %10d %10d %8.3f' % (p, a[0], b2[0], b2[0] / a[0]))
    print('geometric mean elig=2 / elig=1: %.3f (speed-up %.3f)' % (gm(rs), 1 / gm(rs)))
