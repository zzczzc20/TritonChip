#!/usr/bin/env python3
"""kfuse_report.py -- aggregate the kfuse_study.py records and apply the registered gate.

  python kfuse_report.py DIR [DIR ...] [--out-json agg.json] [--out-txt tables.txt]

A later DIR overrides an earlier one per (job, seed, candidate) (the re-runs that add the noise-free metric, the fine
sweep of W below the registered menu).

Gate (doc/r42_plan.md, "Numerics decision for the wide core", item 2, registered before any result): the point stays
inside its frozen tolerance AND the relative L2 error of its output against the binary64 reference is at most 1.5x
that of the sequential contract on the same inputs. Applied per (point, seed) of a format; a configuration PASSES a
format when every point and every seed passes (the worst of 3 seeds is what is reported).

  meas   lane out_rel_l2 / seq out_rel_l2      the registered ratio as measured: one realisation (the output rounding
                                               turns a binary32 error into a one-ulp flip of a few outputs in a thousand)
  exp    lane out_exp_rel_l2 / seq out_exp_rel_l2   the same ratio with the output rounding's randomness averaged out
                                               (kfuse_study.expected_out_err2): no sampling noise. The VERDICT uses exp
                                               when every (point, seed) of the format has it (basis `exp`), else meas
                                               (marked `*`). The control row shows why: the sequential contract in
                                               decreasing k, which is as accurate as increasing k, has a measured ratio
                                               that strays from 1 by the amount shown.
  sec    DIAGNOSTIC (does not change the verdict): lane/seq error ratio before the output rounding -- the binary32
         accumulator (plain GEMMs) or, for the block-wise FP8 formats, the K-block dot before the scales
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kfuse_model as km  # noqa: E402

GATE = 1.5
FORMATS = ('e4m3', 'e5m2', 'fp16', 'bf16')
MENU_W = (0, 3, 8, 24)
EXPLORE_W = (-16, -8)
FINE_W = (-6, -4, -2)
ALL_W = tuple(sorted(set(MENU_W + EXPLORE_W + FINE_W)))
VAR_NAME = {'acc_in': 'acc_in (accumulator in the tree)', 'acc_after': 'acc_after (group sum rounded, then fp32 add)',
            'sum_fma': 'sum_fma (group sum kept wide, added to C once; = the existing m.mma.dp structure)'}


def load(dirs):
    by = defaultdict(dict)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
            for line in open(f):
                r = json.loads(line)
                by[(r['job'], r['seed'])][r['cfg']] = r
    return by


def ratio(a, b):
    if b == 0:
        return 1.0 if a == 0 else float('inf')
    return a / b


def aggregate(by):
    agg = {}
    for (job, seed), cands in by.items():
        if 'seq' not in cands:
            continue
        seq = cands['seq']
        for tag, r in cands.items():
            if tag in ('seq', 'seqrev'):
                continue
            fmt = r['fmt']
            e = agg.setdefault((fmt, tag), dict(fmt=fmt, cfg=tag, items=[]))
            has_exp = 'out_exp_rel_l2' in r and 'out_exp_rel_l2' in seq
            e['items'].append(dict(job=job, seed=seed, out=ratio(r['out_rel_l2'], seq['out_rel_l2']),
                                   exp=ratio(r['out_exp_rel_l2'], seq['out_exp_rel_l2']) if has_exp else None,
                                   acc=ratio(r['acc_rel_l2'], seq['acc_rel_l2']),
                                   blk=ratio(r['blk_rel_l2'], seq['blk_rel_l2']) if 'blk_rel_l2' in r else None,
                                   tol_ok=r['out_tol'] <= r['tol'], tol=r['out_tol'], bias=r['acc_bias']))
    res = {}
    for (fmt, tag), e in agg.items():
        it = e['items']
        worst = max(it, key=lambda x: x['out'])
        worst_acc = max(it, key=lambda x: x['acc'])
        d = dict(fmt=fmt, cfg=tag, n=len(it),
                 meas_max=worst['out'], meas_med=float(np.median([x['out'] for x in it])),
                 meas_worst=f"{worst['job']}#s{worst['seed']}",
                 acc_max=worst_acc['acc'], acc_med=float(np.median([x['acc'] for x in it])),
                 acc_worst=f"{worst_acc['job']}#s{worst_acc['seed']}",
                 tol_ok=all(x['tol_ok'] for x in it), tol_max=max(x['tol'] for x in it),
                 bias_max=max(abs(x['bias']) for x in it))
        if it[0]['blk'] is not None:
            d['blk_max'] = max(x['blk'] for x in it)
            d['blk_med'] = float(np.median([x['blk'] for x in it]))
        d['gate_meas'] = bool(d['meas_max'] <= GATE and d['tol_ok'])
        if all(x['exp'] is not None for x in it):
            d['exp_max'] = max(x['exp'] for x in it)
            d['exp_med'] = float(np.median([x['exp'] for x in it]))
            d['exp_worst'] = max(it, key=lambda x: x['exp'])['job']
            d['gate'] = bool(d['exp_max'] <= GATE and d['tol_ok'])
            d['basis'] = 'exp'
            d['out_max'] = d['exp_max']
        else:
            d['gate'] = d['gate_meas']
            d['basis'] = 'meas'
            d['out_max'] = d['meas_max']
        d['sec'] = d.get('blk_max', d['acc_max'])
        d['sec_name'] = 'blk' if 'blk_max' in d else 'acc'
        d['gate_sec'] = bool(d['sec'] <= GATE and d['tol_ok'])
        res[(fmt, tag)] = d
    return res


def parse(tag):
    v, g, w, r = tag.split('.')
    return v, int(g[1:]), int(w[1:]), r


def mark(d):
    m = 'P' if (d['gate'] and d['gate_sec']) else ('p' if d['gate'] else 'F')
    return m + ('*' if d['basis'] == 'meas' else '')


def fmtr(x):
    if x == float('inf'):
        return 'inf'
    if x >= 100:
        return f'{x:.0f}'
    if x >= 10:
        return f'{x:.1f}'
    return f'{x:.2f}'


def cell(d):
    if d is None:
        return '-'
    meas = f"{fmtr(d['meas_max'])}" if d['basis'] == 'exp' else ''
    shown = f"{fmtr(d['out_max'])}" + (f" [{meas}]" if meas else '')
    return f"{mark(d)} {shown} ({fmtr(d['sec'])})"


def grids(res, fmt, ws, variants=km.VARIANTS):
    out = []
    sec = 'dot-level error before the scales' if fmt in ('e4m3', 'e5m2') else 'binary32 accumulator error before the output rounding'
    for variant in variants:
        out.append(f'\n{fmt}, {VAR_NAME[variant]}')
        out.append(f'cell: mark  verdict-ratio [measured ratio]  (secondary ratio = {sec}); '
                   f'P passes the registered gate and the secondary diagnostic, p passes the registered gate only, F fails it, '
                   f'* verdict from the measured ratio only\n')
        out.append('| rounding, G | ' + ' | '.join(f'W={w}' for w in ws) + ' |')
        out.append('|' + '---|' * (len(ws) + 1))
        for rnd in km.ROUNDINGS:
            for G in (2, 4, 8, 16):
                out.append(f'| {rnd}, G={G} | ' + ' | '.join(cell(res.get((fmt, f'{variant}.G{G}.W{w}.{rnd}'))) for w in ws) + ' |')
    return '\n'.join(out)


def smallest_w(res, fmt, variant, rnd, G, key):
    """Smallest tested W such that it and every larger tested W pass `key`; None if even the largest does not."""
    ws = [w for w in ALL_W if (fmt, f'{variant}.G{G}.W{w}.{rnd}') in res]
    best = None
    for w in sorted(ws, reverse=True):
        if res[(fmt, f'{variant}.G{G}.W{w}.{rnd}')][key]:
            best = w
        else:
            break
    return best


def min_w_table(res, fmt, key):
    out = ['| variant | rounding | ' + ' | '.join(f'G={g}' for g in (2, 4, 8, 16)) + ' |', '|---|---|---|---|---|---|']
    for variant in km.VARIANTS:
        for rnd in km.ROUNDINGS:
            cells = []
            for G in (2, 4, 8, 16):
                w = smallest_w(res, fmt, variant, rnd, G, key)
                if w is None:
                    cells.append('none')
                else:
                    t = km.tree_width_bits(km.Cfg(G, w, rnd, variant))
                    cells.append(f"W={w}: {t['total']} b" + ('' if w in MENU_W else ' (outside menu)'))
            out.append(f'| {variant} | {rnd} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def tree_width_table():
    out = ['| variant | G | terms | W=0 | W=3 | W=8 | W=24 |', '|---|---|---|---|---|---|---|']
    for variant in km.VARIANTS:
        for G in (2, 4, 8, 16):
            t0 = km.tree_width_bits(km.Cfg(G, 0, 'trunc', variant))
            cells = [f"{km.tree_width_bits(km.Cfg(G, W, 'trunc', variant))['total']}" for W in MENU_W]
            out.append(f"| {variant} | {G} | {t0['terms']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(out)


SUMMARY_ROWS = [('acc_in trunc W=0', 'acc_in', 'trunc', 0), ('acc_in trunc W=3', 'acc_in', 'trunc', 3),
                ('acc_in rn W=0', 'acc_in', 'rn', 0), ('acc_after trunc W=0', 'acc_after', 'trunc', 0),
                ('acc_after rn W=0', 'acc_after', 'rn', 0), ('sum_fma trunc W=0', 'sum_fma', 'trunc', 0),
                ('sum_fma rn W=0', 'sum_fma', 'rn', 0), ('sum_fma trunc W=24 (existing m.mma.dp structure)', 'sum_fma', 'trunc', 24)]


def summary_table(res, fmt):
    """Compact verdict table of one format: cell = mark verdict-ratio / secondary-ratio (worst point and seed)."""
    out = [f'{fmt}   cell: mark  verdict-ratio / secondary-ratio  (worst of all points and seeds)',
           '| configuration | G=2 | G=4 | G=8 | G=16 |', '|---|---|---|---|---|']
    for label, variant, rnd, w in SUMMARY_ROWS:
        cells = []
        for G in (2, 4, 8, 16):
            d = res.get((fmt, f'{variant}.G{G}.W{w}.{rnd}'))
            cells.append('-' if d is None else f"{mark(d)} {fmtr(d['out_max'])} / {fmtr(d['sec'])}")
        out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out-json')
    ap.add_argument('--out-txt')
    a = ap.parse_args()
    by = load(a.dirs)
    res = aggregate(by)
    jobs = defaultdict(set)
    for (job, seed), c in by.items():
        jobs[job.split('/')[0]].add((job, seed))
    md = ['Control: the sequential contract in decreasing k against increasing k (a ratio of 1 expected)\n',
          '| format | point x seed | measured ratio: max / median / min | expected ratio: max / min | accumulator ratio: max |',
          '|---|---|---|---|---|']
    ctrl = {}
    for fmt in FORMATS:
        xs = []
        for (job, seed), c in by.items():
            if job.startswith(fmt + '/') and 'seq' in c and 'seqrev' in c:
                s, r = c['seq'], c['seqrev']
                xs.append((ratio(r['out_rel_l2'], s['out_rel_l2']),
                           ratio(r['out_exp_rel_l2'], s['out_exp_rel_l2']) if 'out_exp_rel_l2' in r and 'out_exp_rel_l2' in s else None,
                           ratio(r['acc_rel_l2'], s['acc_rel_l2'])))
        if xs:
            ex = [x[1] for x in xs if x[1] is not None]
            ctrl[fmt] = dict(n=len(xs), meas_max=max(x[0] for x in xs), meas_med=float(np.median([x[0] for x in xs])),
                             meas_min=min(x[0] for x in xs), acc_max=max(x[2] for x in xs),
                             exp_max=max(ex) if ex else None, exp_min=min(ex) if ex else None, n_exp=len(ex))
            c = ctrl[fmt]
            e = f"{fmtr(c['exp_max'])} / {fmtr(c['exp_min'])} ({c['n_exp']} of {c['n']})" if ex else 'not recorded'
            md.append(f"| {fmt} | {c['n']} | {fmtr(c['meas_max'])} / {fmtr(c['meas_med'])} / {fmtr(c['meas_min'])} | {e} | {fmtr(c['acc_max'])} |")
    for fmt in FORMATS:
        if not any(k[0] == fmt for k in res):
            continue
        md.append(f'\n\n==== {fmt} ====\n')
        md.append(f"{len(jobs[fmt])} (point, seed) jobs over points: " + ', '.join(sorted({j for j, s in jobs[fmt]})))
        md.append(grids(res, fmt, MENU_W))
        explored = [w for w in ALL_W if w not in MENU_W and any(k[0] == fmt and parse(k[1])[2] == w for k in res)]
        if explored:
            md.append(f'\nWindows narrower than a binary32 significand (outside the registered W menu), {fmt}:')
            md.append(grids(res, fmt, explored, ('acc_after', 'sum_fma')))
    md.append('\n\n==== Summary of the verdicts ====\n')
    for fmt in FORMATS:
        if any(k[0] == fmt for k in res):
            md.append(summary_table(res, fmt) + '\n')
    md.append('\n\n==== Smallest W per G that passes the registered gate (verdict ratio <= 1.5, tolerance holds); adder-tree width in bits ====\n'
              '(smallest tested W such that it and every larger tested W pass)\n')
    for fmt in FORMATS:
        if any(k[0] == fmt for k in res):
            md.append(f'\n{fmt}\n')
            md.append(min_w_table(res, fmt, 'gate'))
    md.append('\n\n==== The same with the secondary diagnostic also required (secondary ratio <= 1.5) ====\n')
    for fmt in FORMATS:
        if any(k[0] == fmt for k in res):
            md.append(f'\n{fmt}\n')
            md.append(min_w_table(res, fmt, 'gate_sec'))
    md.append('\n\n==== Adder-tree width in bits = window (24 + W) + carry bits + sign ====\n')
    md.append(tree_width_table())
    text = '\n'.join(md) + '\n'
    if a.out_txt:
        open(a.out_txt, 'w').write(text)
    if a.out_json:
        json.dump(dict(control=ctrl, results=list(res.values())), open(a.out_json, 'w'), indent=1, sort_keys=True)
    print(text)


if __name__ == '__main__':
    main()
