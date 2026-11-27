#!/usr/bin/env python3
"""R42 issue: compare the endpoints.csv of two tile screens (tools/r42/area/tile_screen.tcl: every endpoint with slack < 1.5 ns).
  python3 ep_compare.py CTL/endpoints.csv CAND/endpoints.csv [PERIOD_NS]
Per group of endpoints (by the path's end pin): number of endpoints below 0 / 0.5 / 1.0 ns slack and the worst slack, control against candidate.
Groups: the lever's own registers (operand capture x_v*_q / x_v*_f / of_d*, of_got, r_fwd, pf_tag, stall/hazard cone) and every unit of the tile."""
import csv, re, sys

GROUPS = [
    ('operand capture x_v?_q/_f, of_d?', r'/(x_v[abc]_[qf]|of_d[012])_reg'),
    ('of_got / r_fwd / pf_tag / adv_q', r'/(of_got|r_fwd|pf_tag|adv_q|of_acc_q)_reg'),
    ('core front end valids and pc', r'u_core/(r_valid|d_valid|x_valid|f1_valid|fb_valid|pc|r_pc|d_pc|stall)'),
    ('banked VRF', r'u_vrf|g_vrf_banked'),
    ('hybrid LSU', r'u_lsu'),
    ('vector FPU', r'u_vfpu'),
    ('vector ALU', r'u_valu'),
    ('matrix engine', r'u_mtx'),
    ('transfer engine', r'u_txf'),
    ('home slice', r'u_home|u_slice|home'),
    ('core other (u_core/*)', r'u_core/'),
]


def load(p):
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append((float(r['slack']), r['start'], r['end'], int(r['levels'])))
        except (ValueError, KeyError):
            pass
    return rows


def group(rows):
    out = {}
    taken = set()
    for name, rx in GROUPS:
        pat = re.compile(rx)
        sel = [i for i, r in enumerate(rows) if i not in taken and pat.search(r[2])]
        taken.update(sel)
        s = [rows[i][0] for i in sel]
        out[name] = (len(s), sum(1 for x in s if x < 0), sum(1 for x in s if x < 0.5), sum(1 for x in s if x < 1.0), min(s) if s else None)
    s = [r[0] for r in rows]
    out['ALL endpoints below 1.5 ns'] = (len(s), sum(1 for x in s if x < 0), sum(1 for x in s if x < 0.5), sum(1 for x in s if x < 1.0), min(s) if s else None)
    return out


a, b = group(load(sys.argv[1])), group(load(sys.argv[2]))
print(f"{'group':<36}{'ctl: n  <0 <0.5 <1.0  worst':>34}{'cand: n  <0 <0.5 <1.0  worst':>38}")
for k in a:
    fa, fb = a[k], b[k]
    f = lambda t: f"{t[0]:>6}{t[1]:>5}{t[2]:>5}{t[3]:>5}  {('%.3f' % t[4]) if t[4] is not None else '   -  ':>7}"
    print(f"{k:<36}{f(fa):>34}{f(fb):>38}")
