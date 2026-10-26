#!/usr/bin/env python3
"""Directional validation and per-family breakdown of model predictions.

    directional.py --dataset D.json[.gz] --predictions P.jsonl [--predictions ...] \
        --fidelity report.json --out directional.json

A/B effects: two board configurations of the same point and software tree that
differ in exactly one runtime knob of interest (TNCA_ACTIVE_CORES,
TNCA_RTL_MEM_PORTS, TNCA_RTL_OUTSTANDING, TNCA_TRANSFER_POLICY). The board
effect is the ratio of their graph makespans (median over records); the model
effect is the ratio of the two predictions. A direction is reproduced when
both ratios are on the same side of 1 (or both within +-2 %, "no effect").

Per family: the calibrate.py fidelity report's pairs grouped by point family
and split (median/max |error|, mean signed error).
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics

KNOBS = ('TNCA_ACTIVE_CORES', 'TNCA_RTL_MEM_PORTS', 'TNCA_RTL_OUTSTANDING', 'TNCA_TRANSFER_POLICY')
FLAT = 0.02


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def knobs_of(r):
    return {k: v for k, v in sorted(r['knobs'].items()) if k != 'TNCA_OPERATING_CLOCK_EVIDENCE'}


def side(ratio):
    return 0 if abs(ratio - 1) <= FLAT else (1 if ratio > 1 else -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--predictions', action='append', required=True)
    ap.add_argument('--fidelity')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ds = load(a.dataset)
    groups = collections.defaultdict(list)
    family = {}
    for r in ds['rows']:
        if not r.get('graph_makespan_cycles'):
            continue
        family[r['point']] = r.get('family')
        groups[(r['point'], r['software_tree'], json.dumps(knobs_of(r), sort_keys=True))].append(r)
    preds = {}
    for path in a.predictions:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                preds[(p['point'], json.dumps(dict(sorted(p['knobs'].items())), sort_keys=True))] = p
    board = {k: statistics.median(x['graph_makespan_cycles'] for x in rs) for k, rs in groups.items()}
    effects = []
    for (k1, b1), (k2, b2) in itertools.permutations(board.items(), 2):
        if k1[0] != k2[0] or k1[1] != k2[1]:
            continue
        x, y = json.loads(k1[2]), json.loads(k2[2])
        diff = [n for n in set(x) | set(y) if x.get(n) != y.get(n)]
        if len(diff) != 1 or diff[0] not in KNOBS:
            continue
        knob = diff[0]
        lo, hi = sorted([x.get(knob) or '', y.get(knob) or ''])
        if x.get(knob) != lo:
            continue
        m1, m2 = preds.get((k1[0], k1[2])), preds.get((k2[0], k2[2]))
        row = dict(point=k1[0], tree=k1[1][6:14], knob=knob, a=lo, b=hi, split=groups[k1][0]['split'],
                   board_a=b1, board_b=b2, board_ratio=b2 / b1)
        if m1 and m2:
            row.update(model_a=m1['graph_cycles'], model_b=m2['graph_cycles'],
                       model_ratio=m2['graph_cycles'] / m1['graph_cycles'])
            row['direction_reproduced'] = side(row['model_ratio']) == side(row['board_ratio'])
            row['ratio_error'] = row['model_ratio'] / row['board_ratio'] - 1
        else:
            row['missing'] = [n for n, m in (('a', m1), ('b', m2)) if not m]
        effects.append(row)
    summary = {}
    for knob in KNOBS:
        rows = [e for e in effects if e['knob'] == knob and 'model_ratio' in e]
        if rows:
            summary[knob] = dict(pairs=len(rows), reproduced=sum(e['direction_reproduced'] for e in rows),
                                 median_board_ratio=statistics.median(e['board_ratio'] for e in rows),
                                 median_model_ratio=statistics.median(e['model_ratio'] for e in rows),
                                 median_abs_ratio_error=statistics.median(abs(e['ratio_error']) for e in rows),
                                 missing=sum(1 for e in effects if e['knob'] == knob and 'missing' in e))
    out = dict(schema='tnca-systemc-directional/1', flat_band=FLAT, summary=summary,
               effects=sorted(effects, key=lambda e: (e['knob'], e['point'], e['tree'])))
    if a.fidelity:
        fid = load(a.fidelity)
        fam = collections.defaultdict(list)
        for p in fid['pairs']:
            fam[(p['split'], family.get(p['point']) or p['point'].split('/')[0])].append(p)
        out['families'] = [dict(split=s, family=f, n=len(ps),
                                median_abs=statistics.median(abs(p['rel_err']) for p in ps),
                                max_abs=max(abs(p['rel_err']) for p in ps),
                                mean_signed=statistics.mean(p['rel_err'] for p in ps),
                                points=sorted({p['point'] for p in ps}))
                           for (s, f), ps in sorted(fam.items())]
    out['inputs'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.dataset, *a.predictions]}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for knob, s in summary.items():
        print(f"{knob:22s} pairs={s['pairs']:3d} reproduced={s['reproduced']:3d} board~{s['median_board_ratio']:.3f} "
              f"model~{s['median_model_ratio']:.3f} |ratio err|~{s['median_abs_ratio_error']:.3f} missing={s['missing']}")
    for f in out.get('families', []):
        print(f"{f['split']:12s} {f['family']:20s} n={f['n']:3d} median|err|={f['median_abs']:.3f} "
              f"max={f['max_abs']:.3f} mean={f['mean_signed']:+.3f}")


if __name__ == '__main__':
    main()
