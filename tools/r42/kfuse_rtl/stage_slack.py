#!/usr/bin/env python3
"""stage_slack.py ENDPOINTS.csv -- worst slack per pipeline boundary of tnca_kfuse_lane from the *_endpoints.csv that
tools/r42/kfuse_rtl/ooc.tcl writes (worst path into every register / DSP pin). Boundaries are told by the endpoint's
name; anything unmatched is listed under 'other'."""
import csv
import re
import sys

STAGES = [
    ('in_regs (w_*)',        r'^w_'),
    ('c0 decode regs/DSP1 A,B', r'g_p\[\d+\]\.(ma_q|mb_q|x_q|sg_q)'),
    ('DSP1 P (P1)',          r'g_p\[\d+\]\.p1_q'),
    ('T regs',               r'g_t\.t_r'),
    ('Tmax tree regs',       r'u_max/'),
    ('DSP2 operand regs',    r'g_s[01]\.(a2_q|b2_q|c2_q|neg_q)'),
    ('DSP2 P (terms)',       r'g_s[01]\.p2_q'),
    ('adder tree',           r'u_tree/'),
    ('pkt reg',              r'g_pk\.q'),
    ('B0 reg',               r'g_b0\.q'),
    ('B1 reg',               r'g_b1\.q'),
    ('c/flag delay lines',   r'u_d[cfvtb]|u_d1|u_v'),
    ('y_q / valid_q',        r'^(y_q|valid_q)'),
]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    best = {}
    for r in rows:
        end = r['end']
        name = 'other'
        for n, pat in STAGES:
            if re.search(pat, end):
                name = n
                break
        s = float(r['slack'])
        if name not in best or s < best[name][0]:
            best[name] = (s, r['start'], end, r['levels'], r['delay'])
    print('%-28s %8s %6s %7s  %s' % ('boundary', 'slack', 'levels', 'delay', 'worst path (start -> end)'))
    for n, _ in STAGES + [('other', '')]:
        if n in best:
            s, st, en, lv, d = best[n]
            print('%-28s %8.3f %6s %7s  %s -> %s' % (n, s, lv, d, st, en))


main()
