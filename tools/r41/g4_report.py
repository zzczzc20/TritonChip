#!/usr/bin/env python3
"""Tables of G4_DESIGN.md, generated from the recorded G4 runs.

    python3 tools/r41/g4_report.py [--dir evidence/r41/hbm_bw/g4]

Reads issue_bound.json (tools/r41/g4_issue_bound.py, analytic on compiled
code) and model_<size>/suite_model.json (tools/r41/g4_model_suite.py, SystemC
model) and prints the markdown tables quoted in G4_DESIGN.md. Nothing is
typed in: every value is read from those files.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = 320.0 / 3.0
FAMS = ['copy', 'clone', 'add', 'sub', 'mul', 'div', 'relu', 'silu', 'gelu', 'exp', 'sigmoid', 'cast',
        'sum', 'mean', 'max', 'layernorm', 'rmsnorm', 'softmax']


def fmt(x):
    if x is None:
        return '–'
    return f'{x:.1f}' if x < 100 else f'{x:.0f}'


def issue_table(d):
    out = ['| Family | dtype | Layout | data-dependent instr. / 16 elem. | SFU ops | B/program | ceiling, 1 SFU slice | 16 SFU slices | 16 slices + own v.ld/v.st | ≥ 106.7? |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    rows = [r for r in d['rows'] if '-large-' in r['id']]
    key = lambda r: (FAMS.index(r['family']) if r['family'] in FAMS else 99, r['id'])
    for r in sorted(rows, key=key):
        m = re.search(r'/(\w+?)-(?:to-(\w+)-)?large-(\w+)-r', r['id'])
        dt = m.group(1) + (f'→{m.group(2)}' if m.group(2) else '')
        lay = m.group(3)
        k = r.get('kernels') or [{}]
        b = r.get('bound')
        if b is None:
            out.append(f"| {r['family']} | {dt} | {lay} | {k[0].get('k_data','–')} (loop body) | {k[0].get('n_sfu','–')} | – | generic | | | no (see §3.3) |")
            continue
        unl = b['sfu1'] is None
        ok = 'yes (no compute)' if unl else ('yes' if b['sfu16'] >= TARGET else 'no')
        out.append(f"| {r['family']} | {dt} | {lay} | {k[0]['k_data']} | {k[0]['n_sfu']} | {r['bytes_per_program']:.0f} | "
                   f"{'unbounded' if unl else fmt(b['sfu1'])} | {'unbounded' if unl else fmt(b['sfu16'])} | "
                   f"{fmt(b['sfu16_with_lsu_memory_ops'])} | {ok} |")
    n_ok = sum(1 for r in rows if r.get('bound') and (r['bound']['sfu1'] is None or r['bound']['sfu16'] >= TARGET))
    out.append('')
    out.append(f'Large points whose ideal issue ceiling reaches 106.7 B/cycle/core: {n_ok} of {len(rows)}.')
    return '\n'.join(out)


def model_table(path):
    d = json.loads(Path(path).read_text())
    vs = list(d['variants'])
    by = defaultdict(dict)
    status = defaultdict(lambda: defaultdict(int))
    for v in vs:
        for pid, r in d['variants'][v]['points'].items():
            by[pid][v] = r
            status[v][r['status']] += 1
    head = '| Family | dtype | Layout | ' + ' | '.join(f'{v} B/cycle/core' for v in vs) + ' |'
    out = [head, '|---|---|---|' + '---:|' * len(vs)]
    fam_max = defaultdict(lambda: defaultdict(float))
    for pid in sorted(by, key=lambda p: (FAMS.index(p.split('/')[0][3:]) if p.split('/')[0][3:] in FAMS else 99, p)):
        m = re.search(r'/(\w+?)-(?:to-(\w+)-)?\w+?-(contig|strided|broadcast)-r', pid)
        if m is None:      # a point of another suite (the 113-point non-regression set)
            out.append(f"| {pid} | | | " + ' | '.join(
                (fmt(by[pid][v]['b_per_cycle_per_core']) if by[pid].get(v, {}).get('status') == 'pass'
                 else by[pid].get(v, {}).get('status', 'missing')) for v in vs) + ' |')
            continue
        dt = m.group(1) + (f'→{m.group(2)}' if m.group(2) else '')
        cells = []
        for v in vs:
            r = by[pid].get(v)
            val = r and r['status'] == 'pass' and r['b_per_cycle_per_core']
            cells.append(fmt(val) if val else (r['status'] if r else 'missing'))
            if val:
                fam_max[v][pid.split('/')[0][3:]] = max(fam_max[v][pid.split('/')[0][3:]], val)
        out.append(f"| {pid.split('/')[0][3:]} | {dt} | {m.group(3)} | " + ' | '.join(cells) + ' |')
    out.append('')
    out.append('Status counts: ' + '; '.join(f'{v}: ' + ', '.join(f'{k} {n}' for k, n in sorted(status[v].items())) for v in vs))
    best = {v: max(fam_max[v].values()) if fam_max[v] else None for v in vs}
    out.append('Highest B/cycle/core of any point: ' + '; '.join(f'{v} {fmt(b)}' for v, b in best.items()))
    return '\n'.join(out)


def tiled_table(base, arms):
    """Per point and memory configuration: B/cycle/core of the default lowering
    and of each tiled mode, plus a summary (geometric-mean speedup, points
    slower by more than 2 %, numerical failures). Every registered point."""
    import statistics
    b = json.loads(Path(base).read_text())
    t = {name: json.loads(Path(p).read_text()) for name, p in arms}
    out = []
    for v in ('run1', 'g4'):
        if v not in b['variants'] or any(v not in d['variants'] for d in t.values()):
            continue
        pts = b['variants'][v]['points']
        out.append(f'### Memory configuration `{v}`\n')
        summ = ['| Arm | Points | Pass | Geometric-mean speedup | Min | Max | Slower than default by > 2 % |',
                '|---|---:|---:|---:|---:|---:|---:|']
        rows = ['| Family | dtype | Layout | default | ' + ' | '.join(t) + ' |',
                '|---|---|---|---:|' + '---:|' * len(t)]
        for name, d in t.items():
            tp = d['variants'][v]['points']
            sp = [pts[k]['device_cycles'] / tp[k]['device_cycles'] for k in pts
                  if tp.get(k, {}).get('device_cycles') and pts[k].get('device_cycles')]
            npass = sum(1 for k in pts if tp.get(k, {}).get('status') == 'pass')
            summ.append(f'| {name} | {len(pts)} | {npass} | {statistics.geometric_mean(sp):.3f} | '
                        f'{min(sp):.3f} | {max(sp):.3f} | {sum(1 for x in sp if x < 0.98)} |')
        key = lambda p: (FAMS.index(p.split('/')[0][3:]) if p.split('/')[0][3:] in FAMS else 99, p)
        for pid in sorted(pts, key=key):
            m = re.search(r'/(\w+?)-(?:to-(\w+)-)?\w+?-(contig|strided|broadcast)-r', pid)
            dt = m.group(1) + (f'→{m.group(2)}' if m.group(2) else '')
            cells = [fmt(pts[pid]['b_per_cycle_per_core'])]
            for d in t.values():
                r = d['variants'][v]['points'].get(pid, {})
                cells.append(fmt(r.get('b_per_cycle_per_core')) if r.get('status') == 'pass' else r.get('status', 'missing'))
            rows.append(f"| {pid.split('/')[0][3:]} | {dt} | {m.group(3)} | " + ' | '.join(cells) + ' |')
        out += summ + [''] + rows + ['']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(ROOT/'evidence/r41/hbm_bw/g4'))
    a = ap.parse_args()
    base = Path(a.dir)
    print('## Issue ceilings (analytic on compiled code)\n')
    print(issue_table(json.loads((base/'issue_bound.json').read_text())))
    arms = [(f'tiled mode {m}', base/f'model_small_tiled{m}'/'suite_model.json') for m in (1, 2)]
    if all(p.exists() for _n, p in arms):
        print('\n## Tiled lowering, small points (model-prediction-uncalibrated, NON-DEFAULT TNCA_TILED_POINTWISE)\n')
        print(tiled_table(base/'model_small'/'suite_model.json', arms))
    for p in sorted(base.glob('model_*/suite_model.json')):
        if 'tiled' in p.parent.name:
            continue
        d = json.loads(p.read_text())
        nd = d.get('non_default_settings') or []
        label = f", NON-DEFAULT {' '.join(nd)}: does not count for any target" if nd else ''
        print(f"\n## Model, {d['size']} points, {p.parent.name} (model-prediction-uncalibrated{label})\n")
        print(model_table(p))


if __name__ == '__main__':
    main()
