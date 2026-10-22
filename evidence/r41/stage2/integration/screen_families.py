#!/usr/bin/env python3
"""Group a tile screen's worst routed paths into A3 census families.

    screen_families.py --census REPORT_JSON --paths TOP_PATHS_RPT [--paths ...] --label L [--label ...]
                       [--margin 0.5] [--out OUT.json]

TOP_PATHS_RPT is flow/fpga/screen_r41_tile.tcl's top_paths_routed.rpt
(report_timing -max_paths 2000 -nworst 1 -unique_pins), one path per endpoint.
Each path's start and end cells are reduced to the census's family names
(pin dropped, bit indices -> [*], replication suffix __N dropped, prefixed with
the tile's place in the kernel), so a family here is the same key as in the A3
routed census (frequency-preflight-v2 report.json family_groups). A path is
counted when its slack is below MARGIN; the screen is one tile alone in an SLR,
so near-critical families matter, not only failing ones. Each family is also
tagged with the Stage 2 candidate that owns it (by unit, see OWNER).
"""
import argparse, json, re
from collections import defaultdict

TILE = 'system/shell/u_system/g_tile[*].u_tile/'
SOURCE_OWNER = [  # checked on the source first: W's forward/write fan-out into operand capture
    (r'u_core/w_(idx|en|class|rctx)_reg', 'F01/F02/F05 (W forward, operands)'),
]
OWNER = [  # first match wins; patterns on the destination, then the source
    (r'u_txf/', 'transfer engine (u_txf; X8 edits it)'),
    (r'u_vfpu|u_fma/', 'F03/F08 (VFPU, FMA)'),
    (r'u_mtx|u_acc', 'F09 (MTX, accumulator)'),
    (r'g_lsu_pair|u_mesh|core_ni|cache_handoff', 'F10 (LSU, NoC)'),
    (r'nuca|u_slice', 'F07 (NUCA slice)'),
    (r'dispatch', 'F06 (grid dispatch)'),
    (r'w_sdata|w_vdata|w_pdata|w_rctx|x_pm_idx', 'F04 (writeback)'),
    (r'g_vrf_banked|sb_v_reg|sb_s_reg|u_bank', 'F02/F05 (scoreboard, VRF)'),
    (r'x_s[abc]_idx|u_srf|scalar_rf|u_bar', 'F02 (scalar operands)'),
    (r'w_en_reg|stall|pc_reg|/f_|d_valid|r_valid', 'F01 (stall, front end)'),
    (r'u_core', 'other core'),
]


def family(pin):
    cell = pin.rsplit('/', 1)[0]
    cell = re.sub(r'\[\d+\]', '[*]', cell)
    cell = re.sub(r'__\d+$', '', cell)
    return TILE + cell


def owner(src, dst):
    for pat, name in SOURCE_OWNER:
        if re.search(pat, src):
            return name
    for text in (dst, src):
        for pat, name in OWNER:
            if re.search(pat, text):
                return name
    return 'other'


def paths(rpt):
    cur = None
    for line in open(rpt, errors='replace'):
        m = re.match(r'Slack \((MET|VIOLATED)\)\s*:\s*(-?[\d.]+)ns', line)
        if m:
            if cur and 'dst' in cur:
                yield cur
            cur = dict(slack=float(m[2]))
            continue
        if cur is None:
            continue
        m = re.match(r'\s+Source:\s+(\S+)', line)
        if m and 'src' not in cur:
            cur['src'] = m[1]
        m = re.match(r'\s+Destination:\s+(\S+)', line)
        if m and 'dst' not in cur:
            cur['dst'] = m[1]
        m = re.match(r'\s+Data Path Delay:\s+([\d.]+)ns\s+\(logic ([\d.]+)ns.*route ([\d.]+)ns', line)
        if m:
            cur.update(delay=float(m[1]), logic=float(m[2]), route=float(m[3]))
        m = re.match(r'\s+Logic Levels:\s+(\d+)', line)
        if m:
            cur['levels'] = int(m[1])
    if cur and 'dst' in cur:
        yield cur


def screen(rpt, margin, census):
    fam = defaultdict(lambda: dict(paths=0, failing=0, tns=0.0, worst=None))
    total = dict(paths=0, failing=0, tns=0.0, worst=None)
    for p in paths(rpt):
        if p['slack'] >= margin:
            continue
        key = (family(p['src']), family(p['dst']))
        for f in (fam[key], total):
            f['paths'] += 1
            if p['slack'] < 0:
                f['failing'] += 1
                f['tns'] = round(f['tns'] + p['slack'], 3)
            if f['worst'] is None or p['slack'] < f['worst']['slack']:
                f['worst'] = p
    rows = []
    for (s, d), f in fam.items():
        a3 = census.get((s, d))
        rows.append(dict(source_family=s[len(TILE):], destination_family=d[len(TILE):], owner=owner(s, d),
                         paths_below_margin=f['paths'], failing=f['failing'], tns_ns=f['tns'],
                         worst_slack_ns=f['worst']['slack'], worst_levels=f['worst'].get('levels'),
                         worst_delay_ns=f['worst'].get('delay'), worst_route_ns=f['worst'].get('route'),
                         a3=None if a3 is None else dict(endpoints=a3['endpoints'], tns_ns=a3['tns_ns'],
                                                         worst_slack_ns=a3['worst_slack_ns'])))
    rows.sort(key=lambda r: r['worst_slack_ns'])
    owners = defaultdict(lambda: dict(families=0, paths_below_margin=0, failing=0, worst_slack_ns=None))
    for r in rows:
        o = owners[r['owner']]
        o['families'] += 1; o['paths_below_margin'] += r['paths_below_margin']; o['failing'] += r['failing']
        o['worst_slack_ns'] = r['worst_slack_ns'] if o['worst_slack_ns'] is None else min(o['worst_slack_ns'], r['worst_slack_ns'])
    return dict(total=dict(paths_below_margin=total['paths'], failing=total['failing'], tns_ns=total['tns'],
                           worst_slack_ns=None if total['worst'] is None else total['worst']['slack']),
                owners=dict(sorted(owners.items(), key=lambda kv: kv[1]['worst_slack_ns'])), families=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--census', required=True)
    ap.add_argument('--paths', action='append', required=True)
    ap.add_argument('--label', action='append', required=True)
    ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--out')
    a = ap.parse_args()
    census = {(f['source_family'], f['destination_family']): f for f in json.load(open(a.census))['family_groups']}
    doc = dict(schema='tnca-r41s2-screen-families/1', margin_ns=a.margin,
               screens={l: screen(p, a.margin, census) for l, p in zip(a.label, a.paths)})
    if a.out:
        open(a.out, 'w').write(json.dumps(doc, indent=1) + '\n')
    for l, s in doc['screens'].items():
        t = s['total']
        print(f"== {l}: {t['paths_below_margin']} endpoints below +{a.margin} ns, {t['failing']} failing, "
              f"TNS {t['tns_ns']}, worst {t['worst_slack_ns']}")
        for o, v in s['owners'].items():
            print(f"   {o:28s} families {v['families']:3d} endpoints {v['paths_below_margin']:4d} "
                  f"failing {v['failing']:4d} worst {v['worst_slack_ns']:+.3f}")
        for r in s['families'][:8]:
            a3 = r['a3'] and f"A3 {r['a3']['endpoints']} ep worst {r['a3']['worst_slack_ns']}"
            print(f"   {r['worst_slack_ns']:+.3f} n={r['paths_below_margin']:3d} L{r['worst_levels']} "
                  f"{r['source_family']} -> {r['destination_family']}  [{r['owner']}] {a3 or 'not in A3'}")


if __name__ == '__main__':
    main()
