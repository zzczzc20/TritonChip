#!/usr/bin/env python3
"""bit_families.py PERIOD LABEL=endpoints.csv ...   -- failing endpoints (slack < 0 at PERIOD) of the path families that each CLKFIX bit targets, per screen (OOC tile screens, routed)."""
import csv
import re
import sys

period = float(sys.argv[1])
screens = [a.split('=', 1) for a in sys.argv[2:]]


def leaf(pin):
    p = pin.split('/')
    return re.sub(r'\[\d+\]', '[]', p[-2] if len(p) > 1 else pin)


def unit(pin):
    p = pin.split('/')
    if p[0] == 'u_core':
        return 'u_core/' + (p[1] if len(p) > 3 else '(core own)')
    return p[0] if len(p) > 2 else '(tile own)'


FAM = [
    ('0 PICK_REG: LSU fast done/rr_reg -> out_vector/out_scalar/out_mask', lambda s, e: unit(e) == 'u_core/g_lsu_hybrid.u_lsu' and leaf(s) in ('done_reg[]', 'rr_reg[]') and leaf(e) in ('out_vector_reg[]', 'out_scalar_reg[]', 'out_mask_reg[]')),
    ('1 FREE_WR: home -> ms_addr', lambda s, e: leaf(e) == 'ms_addr_reg[][]'),
    ('2 VRF write phys: w_rctx/w_idx -> vld_reg', lambda s, e: leaf(e) == 'vld_reg[][]'),
    ('3 WREP: SRF endpoints', lambda s, e: unit(e) == 'u_core/u_srf'),
    ('4 S2 split: home s1_addr/ms_state/wbq -> state arrays, s3_*, tag WE, URAM index', lambda s, e: unit(e) == 'g_home_pipe.u_home_pipe' and leaf(e) != 'ms_addr_reg[][]' and leaf(s) in ('s1_addr_reg[]', 'ms_state_reg[][]', 'wbq_addr_reg[][]', 'rq_addr_reg[][]') and leaf(e) != 'g_tag[].read_tag_reg[]'),
    ('5 ARB_SAME: home rq_addr/s1_addr -> g_tag read_tag', lambda s, e: leaf(e) == 'g_tag[].read_tag_reg[]'),
    ('6 FREE_LOAD: LSU command_reg -> lane_*', lambda s, e: unit(e) == 'u_core/g_lsu_hybrid.u_lsu' and leaf(s) == 'g_lsu_hybrid.command_reg[]'),
    ('8 TOKR: x_sd_q_reg -> any', lambda s, e: leaf(s) == 'x_sd_q_reg[]'),
]
rows = {lab: list(csv.DictReader(open(path))) for lab, path in screens}
print('| family (bit) | ' + ' | '.join('%s fail@%.1f / >6.2' % (lab, period) for lab, _ in screens) + ' |')
print('|---|' + '---|' * len(screens))
for name, pred in FAM:
    cells = []
    for lab, _ in screens:
        f = a = 0
        for r in rows[lab]:
            s = float(r['slack'])
            if pred(r['start'], r['end']):
                f += s < 0
                a += (period - s) > 6.2
        cells.append('%d / %d' % (f, a))
    print('| %s | %s |' % (name, ' | '.join(cells)))
