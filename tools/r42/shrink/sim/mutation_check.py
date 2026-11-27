#!/usr/bin/env python3
"""mutation_check.py TREE OUT  (GB10) -- are the miters able to see a wrong implementation? (R42-shrink)

Each mutant is one single-point textual change of the new logic in tnca_home_pipe.v / tnca_vrf_banked.v / tnca_rf_bank.v, built into a copy under OUT and run in the
matching miter (tb_home_pipe_miter, tb_vrf_miter) for CYCLES cycles. A mutant is KILLED when the miter reports a mismatch. Survivors are listed and not hidden.
"""
import os
import subprocess
import sys

tree, out = sys.argv[1], sys.argv[2]
cycles = os.environ.get('CYCLES', '120000')
os.makedirs(out, exist_ok=True)
HP = 'rtl/nuca/tnca_home_pipe.v'
VB = 'rtl/core/tnca_vrf_banked.v'
RB = 'rtl/core/tnca_rf_bank.v'
mutants = [
    # name, file, old, new, miter args
    ('hp_no_bypass', HP, "dbyp_v    <= dw_en && (dw_index == dr_index);", "dbyp_v    <= 1'b0;", ['home', 'DR=1 AR=0']),
    ('hp_read_ignores_hold', HP, "wire              dr_en    = (m_state == M_WAIT) || (m_state != M_SCAN && !s3_hold);", "wire              dr_en    = 1'b1;", ['home', 'DR=1 AR=0']),
    ('hp_read_always_victim', HP, "(m_state == M_WAIT) ? m_index : (s2_hit ? s2_hit_index : s2_vic_index);", "(m_state == M_WAIT) ? m_index : s2_vic_index;", ['home', 'DR=1 AR=0']),
    ('hp_maint_read_index', HP, "(m_state == M_WAIT) ? m_index : (s2_hit ?", "(m_state == M_WAIT) ? s3_index : (s2_hit ?", ['home', 'DR=1 AR=0']),
    ('hp_maint_enable', HP, "(m_state == M_WAIT) || (m_state != M_SCAN && !s3_hold);", "(m_state != M_SCAN && !s3_hold);", ['home', 'DR=1 AR=0']),
    ('hp_aux_tgw_no_advance', HP, "wire                tgw_en   = !fu_v || fu_step;", "wire                tgw_en   = !fu_v;", ['home', 'DR=0 AR=1']),
    ('hp_aux_msl_no_chain', HP, "wire                msl_en   = !fu_v || (fu_step && fu_last);", "wire                msl_en   = !fu_v;", ['home', 'DR=0 AR=1']),
    ('hp_aux_tgw_addr_chain', HP, "fu_last ? ms_head[fill_next_pick] : tg_next[fu_tc]", "tg_next[fu_tc]", ['home', 'DR=0 AR=1']),
    ('hp_aux_tgw_addr_head', HP, "!fu_v ? ms_head[fill_pick] :", "!fu_v ? ms_head[fill_next_pick] :", ['home', 'DR=0 AR=1']),
    ('hp_aux_msl_addr', HP, "wire [MSHR_W-1:0]   msl_addr = fu_v ? fill_next_pick : fill_pick;", "wire [MSHR_W-1:0]   msl_addr = fill_pick;", ['home', 'DR=0 AR=1']),
    ('hp_aux_write_second', HP, "if (AUX_RAM == 1) tg_wdata_blk[tg_free_idx]  <= s2_wdata;", "if (AUX_RAM == 1) tg_wdata_blk[tg_free_idx+1'b1]  <= s2_wdata;", ['home', 'DR=0 AR=1']),
    ('vrf_xbar_bank_swap', VB, "assign hit[xb] = bank_rsel[xb*8 + xp];", "assign hit[xb] = bank_rsel[((xb+1)%NUM_BANKS)*8 + xp];", ['vrf', 'XB=1 VLDB=0']),
    ('vrf_xbar_mux_idx', VB, "if (hit[xi]) idx = idx | xi[BANK_W-1:0];", "if (hit[xi]) idx = xi[BANK_W-1:0] ^ 1;", ['vrf', 'XB=2 VLDB=0']),
    ('vrf_sel_not_cleared', RB, "s1_sel      <= (ONEHOT_SEL != 0 && re) ? (8'd1 << dst[2:0]) : 8'd0;", "s1_sel      <= (ONEHOT_SEL != 0) ? (8'd1 << dst[2:0]) : 8'd0;", ['vrf', 'XB=1 VLDB=0']),
    ('vrf_sweep_short', RB, "if (sw_cnt == DEPTH - 1) sweeping <= 1'b0;", "if (sw_cnt == DEPTH - 2) sweeping <= 1'b0;", ['vrf', 'XB=0 VLDB=1']),
    ('vrf_vld_set_always', RB, "{!sweeping, wdata[li*ELEN +: ELEN]};", "{1'b1, wdata[li*ELEN +: ELEN]};", ['vrf', 'XB=0 VLDB=1']),
    ('vrf_write_during_sweep', RB, "assign wr_ok = !swp;", "assign wr_ok = 1'b1;", ['vrf', 'XB=0 VLDB=1']),
    ('vrf_read_during_sweep', RB, "&& !swp;", ";", ['vrf', 'XB=0 VLDB=1']),
]
rows = []
for name, f, old, new, (which, args) in mutants:
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(tree, f)).read()
    if src.count(old) != 1:
        rows.append((name, 'NOT APPLIED (pattern count %d)' % src.count(old)))
        continue
    files = {HP: os.path.join(tree, HP), VB: os.path.join(tree, VB), RB: os.path.join(tree, RB)}
    mut = os.path.join(d, os.path.basename(f))
    open(mut, 'w').write(src.replace(old, new))
    files[f] = mut
    vvp = os.path.join(d, 'm.vvp')
    if which == 'home':
        cmd = ['iverilog', '-g2012', '-I', os.path.join(tree, 'rtl/common'), '-o', vvp, '-s', 'tb', os.path.join(tree, 'tools/r42/shrink/sim/tb_home_pipe_miter.v'), files[HP]]
        for a in args.split():
            k, v = a.split('=')
            cmd += ['-P', 'tb.%s=%s' % (k, v)]
        cmd += ['-P', 'tb.CYCLES=' + cycles, '-P', 'tb.HOT=1', '-P', 'tb.NLINES=300']
    else:
        cmd = ['iverilog', '-g2012', '-I', os.path.join(tree, 'rtl/common'), '-o', vvp, '-s', 'tb', os.path.join(tree, 'tools/r42/shrink/sim/tb_vrf_miter.v'), files[VB], files[RB]]
        if 'VLDB=1' in args:
            cmd[4:4] = ['-DVLDA=,.VLD_RAM(0)', '-DVLDB_=,.VLD_RAM(VLDB)']
        for a in args.split():
            k, v = a.split('=')
            cmd += ['-P', 'tb.%s=%s' % (k, v)]
        cmd += ['-P', 'tb.CYCLES=' + cycles]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        rows.append((name, 'COMPILE ERROR: ' + r.stderr.strip().splitlines()[-1][:100] if r.stderr.strip() else 'COMPILE ERROR'))
        continue
    r = subprocess.run(['nice', '-n', '19', 'vvp', vvp], capture_output=True, text=True)
    txt = r.stdout
    killed = ('MISMATCH' in txt) or ('MITER FAIL' in txt) or ('FAIL' in txt)
    first = next((l for l in txt.splitlines() if 'MISMATCH' in l), '')
    rows.append((name, ('KILLED ' + first[:70]) if killed else 'SURVIVED'))
    os.remove(vvp)
with open(os.path.join(out, 'mutation_result.txt'), 'w') as fh:
    for n, s in rows:
        fh.write('%-28s %s\n' % (n, s))
        print('%-28s %s' % (n, s), flush=True)
print('killed %d of %d' % (sum(1 for _, s in rows if s.startswith('KILLED')), len(rows)))
