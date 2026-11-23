#!/usr/bin/env python3
"""mutation_check.py VECFILE G W OUTDIR -- is the bench able to see a wrong lane? (run on GB10 from the tree root)

Applies one small change at a time to a copy of rtl/core/tnca_kfuse_lane.v (each pattern must occur exactly once),
compiles tb_kfuse_lane against it and counts mismatches on VECFILE. A mutant that survives with 0 mismatches is
reported; the ones that are functionally equivalent by construction (marked EQUIV, with the reason) are expected to.
"""
import os
import re
import subprocess
import sys

VEC, G, W, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../../..'
SRC = open(ROOT + '/rtl/core/tnca_kfuse_lane.v').read()
GEN = ['rtl/gen/tnca_mac_b0_xls.v', 'rtl/gen/tnca_fmad_norm_xls.v', 'rtl/gen/tnca_fma_round_xls.v']
os.makedirs(OUT, exist_ok=True)

M = [
    # name, old, new, expected ('KILL' or 'EQUIV: why'), extra -P overrides
    ('c_offset_0', "c2_q <= {7{sg_d}};", "c2_q <= 7'd0;", 'KILL', ''),
    ('c_offset_126', "c2_q <= {7{sg_d}};", "c2_q <= {sg_d, sg_d, sg_d, sg_d, sg_d, sg_d, 1'b0};",
     'EQUIV: no E4M3 product is 127 mod 128 (127 is prime and above 15x15)', ''),
    ('c_offset_125', "c2_q <= {7{sg_d}};", "c2_q <= {sg_d, sg_d, sg_d, sg_d, sg_d, 1'b0, sg_d};", 'KILL', ''),
    ('shift_plus1', "8'd30 + W", "8'd31 + W", 'KILL', ''),
    ('xpb_minus1', "12'd257 - W", "12'd256 - W", 'KILL', ''),
    ('nz_without_lsb', "wire       nz = |p1_q;", "wire       nz = |p1_q[7:1];", 'KILL', ''),
    ('msb_6_as_5', "p1_q[6] ? 3'd6", "p1_q[6] ? 3'd5", 'KILL', ''),
    ('subnormal_exp_0', "wire [3:0] ea = {a8[6:4], a8[3] | ~ha};", "wire [3:0] ea = {a8[6:4], a8[3]};", 'KILL', ''),
    ('negzero_ignores_sign', "assign nzn = (za | zb) & sg;", "assign nzn = (za | zb);", 'KILL', ''),
    ('hi_threshold_24', "wire       hi   = shv[5] | shv[4];", "wire       hi   = shv[5] | (shv[4] & shv[3]);",
     'EQUIV: any threshold from 8 to 24 keeps A below 2^26 and B within 16 bits (one-hot range 0..25 plus the 8-bit pre-shift)', ''),
    ('drop_6_bits', "assign term = p2[7 +: TW];", "assign term = p2[6 +: TW];", 'KILL', ''),
    ('s_zero_never', "wire             s_zero = ~(|s_sum);", "wire             s_zero = 1'b0;", 'KILL', ''),
    ('c_delay_short', "tnca_kfuse_dly #(.W(32), .N(TPK - N0)) u_dc", "tnca_kfuse_dly #(.W(32), .N(TPK - N0 - 1)) u_dc", 'KILL', ''),
    ('max_is_min', "assign mxv[k] = (mxs[2*k] > mxs[2*k+1])", "assign mxv[k] = (mxs[2*k] < mxs[2*k+1])", 'KILL', ''),
    ('nan_dropped', "{pm, xpb, sp, s_zero, 1'b0, nan_s}", "{pm, xpb, sp, s_zero, 1'b0, 1'b0}", 'KILL', ''),
    ('no_flush_gate', "wire       shok = ~shv[7] & (|p1_d);", "wire       shok = 1'b1;", 'KILL', ''),
    ('abs_without_plus1', "(~s_sum + 1'b1)", "(~s_sum)", 'KILL', ''),
    ('sign_of_zero_sum', "wire             sp     = s_zero ? nzn_s : s_neg;", "wire             sp     = s_zero ? 1'b0 : s_neg;", 'KILL', ''),
]

fails = 0
print('%-24s %-10s %-8s %s' % ('mutant', 'mismatches', 'expected', 'verdict'))
for name, old, new, exp, par in M:
    n = SRC.count(old)
    if n != 1:
        print('%-24s PATTERN OCCURS %d TIMES' % (name, n))
        fails += 1
        continue
    f = '%s/m_%s.v' % (OUT, name)
    open(f, 'w').write(SRC.replace(old, new))
    vvp = '%s/m_%s.vvp' % (OUT, name)
    cmd = ['iverilog', '-g2005', '-o', vvp, '-Ptb_kfuse_lane.G=%d' % G, '-Ptb_kfuse_lane.W=%d' % W,
           '-Ptb_kfuse_lane.NMAX=400000'] + par.split() + [ROOT + '/tools/r42/kfuse_rtl/tb_kfuse_lane.v', f] + [ROOT + '/' + g for g in GEN]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print('%-24s COMPILE ERROR %s' % (name, r.stderr[:200]))
        fails += 1
        continue
    nv = sum(1 for _ in open(VEC))
    r = subprocess.run(['vvp', '-n', vvp, '+VEC=' + VEC, '+N=%d' % nv, '+MAXERR=0'], capture_output=True, text=True)
    m = re.search(r'RESULT .* mismatches=(\d+)', r.stdout)
    mm = int(m.group(1)) if m else -1
    ok = (mm > 0) if exp == 'KILL' else (mm == 0)
    verdict = 'killed' if mm > 0 else 'survived'
    if exp != 'KILL':
        verdict += ' (' + exp + ')'
    print('%-24s %-10d %-8s %s%s' % (name, mm, exp.split(':')[0], verdict, '' if ok else '   <-- UNEXPECTED'))
    if not ok:
        fails += 1
print('unexpected results: %d' % fails)
