#!/usr/bin/env python3
"""Resource tables for R41 G2 (evidence/r41/hbm_bw/RESOURCES.md).

Every number this prints is read from a Vivado report or a leaf-cell dump
given on the command line; nothing is typed in.

  util_hier.py routed  ROUTED_HIER_RPT            per-tile / per-module tables of the routed image
  util_hier.py compare BASE_RPT CAND_RPT [...]    kernel-synthesis deltas, base vs candidate
  util_hier.py slr     LEAF_CELLS_TSV[.gz]        per-SLR primitive counts per module group
  util_hier.py bytype  HIER_RPT [MIN_LUT]         totals per module type (all instances, kernel)

Report format: `report_utilization -hierarchical` (synthesis or routed; the
column set is taken from the header line).
"""
import gzip
import re
import sys
from collections import OrderedDict, defaultdict

COLS = ['Total LUTs', 'Logic LUTs', 'LUTRAMs', 'SRLs', 'FFs', 'RAMB36', 'RAMB18', 'DSP Blocks']
SHORT = {'Total LUTs': 'LUT', 'Logic LUTs': 'logic', 'LUTRAMs': 'LUTRAM', 'SRLs': 'SRL',
         'FFs': 'FF', 'RAMB36': 'RAMB36', 'RAMB18': 'RAMB18', 'DSP Blocks': 'DSP'}


def parse(path):
    """Rows of (path tuple, module, {col: float}) in report order."""
    rows, header, stack = [], None, []
    op = gzip.open if path.endswith('.gz') else open
    for line in op(path, 'rt', errors='replace'):
        if not line.startswith('|'):
            continue
        cells = line.rstrip('\n').split('|')[1:-1]
        if header is None:
            if cells and cells[0].strip() == 'Instance':
                header = [c.strip() for c in cells]
            continue
        if cells[0].strip() == 'Instance':
            continue
        raw = cells[0]
        name = raw.strip()
        depth = (len(raw) - len(raw.lstrip(' ')) - 1) // 2
        del stack[depth:]
        stack.append(name)
        vals = {}
        for col in COLS:
            if col in header:
                v = cells[header.index(col)].strip()
                vals[col] = float(v) if re.match(r'^-?[0-9.]+$', v) else 0.0
        rows.append((tuple(stack), cells[1].strip(), vals))
    if header is None:
        sys.exit(f'{path}: no hierarchical utilization table')
    return rows


def find(rows, suffix):
    """The row whose path ends with the given components."""
    hits = [r for r in rows if r[0][-len(suffix):] == tuple(suffix)]
    if len(hits) != 1:
        sys.exit(f'expected one row ending in {suffix}, found {len(hits)}')
    return hits[0]


def children(rows, parent):
    n = len(parent)
    return [r for r in rows if len(r[0]) == n + 1 and r[0][:n] == parent]


def fmt(v):
    return f'{int(round(v)):,}'


def table(title, entries, cols=COLS, total=None):
    out = [f'\n#### {title}\n', '| Instance | ' + ' | '.join(SHORT[c] for c in cols) + ' |',
           '|---|' + '---:|' * len(cols)]
    for name, vals in entries:
        out.append(f'| {name} | ' + ' | '.join(fmt(vals.get(c, 0)) for c in cols) + ' |')
    if total is not None:
        out.append('| **total** | ' + ' | '.join(f'**{fmt(total.get(c, 0))}**' for c in cols) + ' |')
    return '\n'.join(out)


def kernel_root(rows):
    # routed image: ... tnca_u50_1/inst/device/system ; kernel synthesis: device/system
    for r in rows:
        if r[0][-2:] == ('device', 'system') and not r[0][-1].startswith('('):
            return r
    sys.exit('no device/system row')


def cmd_routed(path):
    rows = parse(path)
    top = rows[0]
    print(f'Report: `{path}`\n')
    # 1. platform versus kernel
    lvls = [r for r in rows if r[0][-2:] == ('level0_wrapper', 'level0_i')]
    parts = []
    lvl = lvls[0] if lvls else None
    for r in (children(rows, lvl[0]) if lvl else []):
        if r[0][-1].startswith('('):
            continue
        if r[0][-1] == 'ulp':
            for u in children(rows, r[0]):
                if u[0][-1].startswith('('):
                    parts.append(('ulp (own logic)', u[2]))
                else:
                    parts.append((f'ulp/{u[0][-1]}', u[2]))
        else:
            parts.append((r[0][-1], r[2]))
    parts.sort(key=lambda e: -e[1].get('Total LUTs', 0))
    if lvl:
        print(table('Whole design: platform shell versus TNCA kernel (`tnca_u50_1`)', parts, total=top[2]))
    # 2. kernel, non-tile part
    sysrow = kernel_root(rows)
    ent = []
    for r in children(rows, sysrow[0]):
        if r[0][-1] == 'shell':
            for s in children(rows, r[0]):
                if s[0][-1] == 'u_system':
                    for t in children(rows, s[0]):
                        ent.append((f'shell/u_system/{t[0][-1]}', t[2]))
                else:
                    ent.append((f'shell/{s[0][-1]}', s[2]))
        else:
            ent.append((r[0][-1], r[2]))
    ent.sort(key=lambda e: -e[1].get('Total LUTs', 0))
    print(table('TNCA kernel (`device/system`), by block', ent, total=sysrow[2]))
    # 3. per tile, per module (core children, slice, NIs)
    tiles = sorted([r for r in rows if re.fullmatch(r'g_tile\[\d+\]\.u_tile', r[0][-1])],
                   key=lambda r: r[0][-1])
    for col in ['Total LUTs', 'LUTRAMs', 'FFs']:
        names = OrderedDict()
        per = []
        for t in tiles:
            d = {}
            for c in children(rows, t[0]):
                if c[0][-1] == 'u_core':
                    for cc in children(rows, c[0]):
                        d['u_core/' + cc[0][-1]] = cc[2].get(col, 0)
                        names['u_core/' + cc[0][-1]] = None
                else:
                    d[c[0][-1]] = c[2].get(col, 0)
                    names[c[0][-1]] = None
            per.append(d)
        order = sorted(names, key=lambda n: -sum(p.get(n, 0) for p in per))
        hdr = '| Module | ' + ' | '.join(f'{t[0][-1]}' for t in tiles) + ' | share of tile 0 |'
        print(f'\n#### Per tile, per module: {SHORT[col]}\n')
        print(hdr)
        print('|---|' + '---:|' * (len(tiles) + 1))
        t0 = tiles[0][2].get(col, 0) or 1
        for n in order:
            if max(p.get(n, 0) for p in per) == 0:
                continue
            print(f'| {n} | ' + ' | '.join(fmt(p.get(n, 0)) for p in per)
                  + f' | {100.0 * per[0].get(n, 0) / t0:.1f} % |')
        print('| **tile total** | ' + ' | '.join(f'**{fmt(t[2].get(col, 0))}**' for t in tiles) + ' | |')
    # 4. full column set for tile 0's large sub-blocks (depth 3 under the tile)
    t0 = tiles[0]
    ent = []
    for r in rows:
        if r[0][:len(t0[0])] == t0[0] and 1 <= len(r[0]) - len(t0[0]) <= 3 \
                and r[2].get('Total LUTs', 0) >= 2000 and not r[0][-1].startswith('('):
            ent.append(('/'.join(r[0][len(t0[0]):]), r[2]))
    print(table(f'{t0[0][-1]} (SLR0): sub-blocks of at least 2,000 LUTs, three levels deep', ent))


def kernel_tiles(rows):
    return sorted([r for r in rows if re.fullmatch(r'g_tile\[\d+\]\.u_tile', r[0][-1])],
                  key=lambda r: r[0][-1])


def cmd_compare(base, *cands):
    b = parse(base)
    print(f'Base: `{base}`')
    for cand in cands:
        c = parse(cand)
        print(f'\nCandidate: `{cand}`\n')
        print('| Scope | ' + ' | '.join(f'{SHORT[k]} base | {SHORT[k]} cand | Δ' for k in
                                         ['Total LUTs', 'LUTRAMs', 'FFs', 'RAMB36', 'DSP Blocks']) + ' |')
        print('|---|' + '---:|' * 15)
        pairs = [('kernel (top)', b[0][2], c[0][2])]
        for tb, tc in zip(kernel_tiles(b), kernel_tiles(c)):
            pairs.append((tb[0][-1], tb[2], tc[2]))
            for sub in ['u_slice', 'u_core']:
                rb = [r for r in b if r[0][:len(tb[0])] == tb[0] and r[0][len(tb[0]):] == (sub,)]
                rc = [r for r in c if r[0][:len(tc[0])] == tc[0] and r[0][len(tc[0]):] == (sub,)]
                if rb and rc:
                    pairs.append((f'{tb[0][-1]}/{sub}', rb[0][2], rc[0][2]))
        for pat in [r'g_outlet\[\d+\]\.adapter', r'g_local_tmu\.u_tmu', r'g_shared_tmu\.u_tmu', r'u_mesh']:
            rb = {r[0][-1]: r for r in b if re.fullmatch(pat, r[0][-1])}
            rc = {r[0][-1]: r for r in c if re.fullmatch(pat, r[0][-1])}
            for k in sorted(set(rb) | set(rc)):
                pairs.append((k, rb[k][2] if k in rb else {}, rc[k][2] if k in rc else {}))
        for name, vb, vc in pairs:
            cells = []
            for k in ['Total LUTs', 'LUTRAMs', 'FFs', 'RAMB36', 'DSP Blocks']:
                cells += [fmt(vb.get(k, 0)), fmt(vc.get(k, 0)), f'{vc.get(k, 0) - vb.get(k, 0):+,.0f}']
            print(f'| {name} | ' + ' | '.join(cells) + ' |')


GROUPS = [
    ('TNCA tile 0', r'/g_tile\[0\]\.u_tile/'),
    ('TNCA tile 1', r'/g_tile\[1\]\.u_tile/'),
    ('TNCA tile 2', r'/g_tile\[2\]\.u_tile/'),
    ('TNCA DCO/TMU replicas', r'/g_local_tmu\.u_tmu/'),
    ('TNCA outlets (line AXI + links)', r'/system/g_outlet\['),
    ('TNCA mesh', r'/u_system/u_mesh/'),
    ('TNCA other kernel logic', r'/tnca_u50_1/'),
    ('Platform: HBM subsystem (hmss)', r'/ulp/hmss_0/'),
    ('Platform: other dynamic region (ulp)', r'/ulp/'),
    ('Platform: static shell (blp and others)', r''),
]
KINDS = [('LUT cells', lambda rn, pt, c: pt.startswith('CLB.LUT.')),
         ('LUTRAM/SRL cells', lambda rn, pt, c: pt.startswith('CLB.LUTRAM') or pt.startswith('CLB.SRL')),
         ('FF', lambda rn, pt, c: pt.startswith('REGISTER')),
         ('FF replicas (`_rep`)', lambda rn, pt, c: pt.startswith('REGISTER') and '_rep' in c),
         ('CARRY8', lambda rn, pt, c: rn == 'CARRY8'),
         ('RAMB36', lambda rn, pt, c: rn == 'RAMB36E2'),
         ('RAMB18', lambda rn, pt, c: rn == 'RAMB18E2'),
         # a DSP48E2 is several leaf cells; DSP_ALU occurs once per DSP
         ('DSP', lambda rn, pt, c: rn == 'DSP_ALU')]


def cmd_slr(path):
    op = gzip.open if path.endswith('.gz') else open
    pat = [(g, re.compile(p)) for g, p in GROUPS]
    count = defaultdict(lambda: defaultdict(int))
    slrs = set()
    with op(path, 'rt') as f:
        next(f)
        for line in f:
            slr, rn, pt, cell = line.rstrip('\n').split('\t')
            slrs.add(slr)
            grp = next(g for g, p in pat if p.search(cell))
            for k, test in KINDS:
                if test(rn, pt, cell):
                    count[(slr, grp)][k] += 1
    print(f'Leaf-cell dump: `{path}`: placed leaf cells by SLR. LUT and LUTRAM '
          'columns count leaf cells (a LUT site holding two LUT cells, or a LUTRAM macro\'s '
          'RAMD32 pair, counts twice), so they exceed report_utilization\'s site-based LUT '
          'figures; FF, CARRY8, RAMB and DSP counts are exact.\n')
    for slr in sorted(slrs):
        print(f'\n#### {slr}\n')
        print('| Group | ' + ' | '.join(k for k, _ in KINDS) + ' |')
        print('|---|' + '---:|' * len(KINDS))
        tot = defaultdict(int)
        for g, _ in GROUPS:
            v = count.get((slr, g))
            if not v:
                continue
            for k, _ in KINDS:
                tot[k] += v[k]
            print(f'| {g} | ' + ' | '.join(fmt(v[k]) for k, _ in KINDS) + ' |')
        print('| **SLR total** | ' + ' | '.join(f'**{fmt(tot[k])}**' for k, _ in KINDS) + ' |')


def base_module(m):
    m = re.sub(r'^ulp_tnca_u50_1_0_', '', m)
    m = re.sub(r'(__parameterized\d+)?(__\d+)?(_\d+)?$', '', m)
    return re.sub(r'__parameterized\d+.*$', '', m)


def cmd_bytype(path, min_lut='1000'):
    rows = parse(path)
    sysrow = kernel_root(rows)
    n = len(sysrow[0])
    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    for r in rows:
        if r[0][:n] != sysrow[0] or len(r[0]) == n or r[0][-1].startswith('('):
            continue
        t = base_module(r[1])
        # count an instance once: skip it when an ancestor has the same type
        anc = [a for a in rows if False]
        path_types = set()
        for k in range(n + 1, len(r[0])):
            pre = r[0][:k]
            for a in rows:
                if a[0] == pre:
                    path_types.add(base_module(a[1]))
                    break
        if t in path_types:
            continue
        cnt[t] += 1
        for c in COLS:
            agg[t][c] += r[2].get(c, 0)
    tot = sysrow[2].get('Total LUTs', 1)
    print(f'Report: `{path}`; kernel total LUTs {fmt(tot)}. Every instance of a module type, '
          'summed (an instance nested in one of its own type is not counted twice).\n')
    print('| Module type | instances | LUT | % of kernel LUT | LUTRAM | FF | DSP |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for t in sorted(agg, key=lambda t: -agg[t]['Total LUTs']):
        if agg[t]['Total LUTs'] < float(min_lut):
            continue
        a = agg[t]
        print(f'| {t} | {cnt[t]} | {fmt(a["Total LUTs"])} | {100 * a["Total LUTs"] / tot:.1f} % | '
              f'{fmt(a["LUTRAMs"])} | {fmt(a["FFs"])} | {fmt(a["DSP Blocks"])} |')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('routed', 'compare', 'slr', 'bytype'):
        sys.exit(__doc__)
    {'routed': cmd_routed, 'compare': cmd_compare, 'slr': cmd_slr,
     'bytype': cmd_bytype}[sys.argv[1]](*sys.argv[2:])
