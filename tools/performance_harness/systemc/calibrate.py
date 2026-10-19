#!/usr/bin/env python3
"""Judge SystemC model predictions against the board: fidelity on the frozen split.

    calibrate.py evaluate --dataset board_timing_dataset.json[.gz] \
        --predictions predictions.jsonl --out report.json [--fit-report fit.json]

`predictions.jsonl` holds one row per modeled run, from any runner:

    {"point": ID, "knobs": {...}, "graph_cycles": N, "launches": [{"kernel": K, "cycles": N}, ...],
     "model": {"binary_sha256": ..., "config_sha256": ..., "config": PATH}}

A prediction matches a board record of the same point whose knobs agree on
every knob the prediction states (the board records every TNCA_* knob it ran
with). Errors are relative to the board: (model - board) / board, on the device
graph makespan, and per launch where kernels pair up in order.

doc/r40_tasks_and_acceptance.md sec 3.4: freeze at least 10 calibration and an
independent at least 10 holdout points; the model is calibrated when the
holdout graph-makespan median absolute relative error is <= 10 % and its p95
<= 20 %; otherwise it is uncalibrated and its cycles never replace a board
measurement. The splits are spec/r40/suites/R40-04.calibration.json and
R40-04.model-holdout.json. `--fit-report` names the record of how parameters
were chosen; it must declare that only calibration rows were used, or the
holdout judgement is refused.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

MEDIAN_LIMIT = 0.10
P95_LIMIT = 0.20


def read_json(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    return json.loads(raw), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def matches(pred, row):
    return pred['point'] == row['point'] and all(row['knobs'].get(k) == str(v) for k, v in (pred.get('knobs') or {}).items())


def evaluate(dataset, predictions, fit):
    rows = dataset['rows']
    pairs = []
    unmatched = []
    for pred in predictions:
        cands = [r for r in rows if matches(pred, r) and r.get('graph_makespan_cycles')]
        if not cands:
            unmatched.append(pred['point'])
            continue
        # Several board records of one configuration (other identities, other
        # runs): compare with their median, and keep the spread as context.
        board = sorted(r['graph_makespan_cycles'] for r in cands)
        ref = board[len(board) // 2]
        err = (pred['graph_cycles'] - ref) / ref
        launch_err = []
        ref_launches = cands[0]['launches']
        if len(ref_launches) == len(pred.get('launches') or []):
            for b, m in zip(ref_launches, pred['launches']):
                if b.get('cycles') and m.get('cycles') is not None and b['kernel'] == m.get('kernel', b['kernel']):
                    launch_err.append((m['cycles'] - b['cycles']) / b['cycles'])
        pairs.append(dict(point=pred['point'], split=cands[0]['split'], knobs=pred.get('knobs') or {},
                          board_graph_cycles=ref, board_records=len(cands),
                          board_spread=(board[-1] - board[0]) / ref if len(board) > 1 else 0.0,
                          model_graph_cycles=pred['graph_cycles'], rel_err=err,
                          launch_rel_err=dict(n=len(launch_err),
                                              median_abs=quantile([abs(e) for e in launch_err], .5),
                                              p95_abs=quantile([abs(e) for e in launch_err], .95)),
                          model=pred.get('model')))
    out = dict(pairs=pairs, unmatched=sorted(set(unmatched)), splits={})
    for split in ('calibration', 'holdout', 'other'):
        errs = [abs(p['rel_err']) for p in pairs if p['split'] == split]
        signed = [p['rel_err'] for p in pairs if p['split'] == split]
        out['splits'][split] = dict(n=len(errs), median_abs=quantile(errs, .5), p95_abs=quantile(errs, .95),
                                    mean_signed=(sum(signed) / len(signed)) if signed else None)
    hold = out['splits']['holdout']
    frozen = len({p['point'] for p in pairs if p['split'] == 'holdout'})
    if fit is not None and fit.get('fit_rows') != 'calibration-only':
        verdict, why = 'refused', 'the fit report does not declare calibration-only fitting'
    elif frozen < 10:
        verdict, why = 'not-measured', f'{frozen} holdout points matched; sec 3.4 needs at least 10'
    elif hold['median_abs'] <= MEDIAN_LIMIT and hold['p95_abs'] <= P95_LIMIT:
        verdict, why = 'calibrated', 'holdout median and p95 within sec 3.4'
    else:
        verdict, why = 'uncalibrated', (f"holdout median {hold['median_abs']:.3f} / p95 {hold['p95_abs']:.3f} "
                                        f'exceed {MEDIAN_LIMIT} / {P95_LIMIT}')
    out.update(fidelity=verdict, why=why, limits=dict(median_abs=MEDIAN_LIMIT, p95_abs=P95_LIMIT))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--dataset', type=Path, required=True)
    e.add_argument('--predictions', type=Path, required=True)
    e.add_argument('--fit-report', type=Path)
    e.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dataset, dsha = read_json(a.dataset)
    preds = [json.loads(l) for l in a.predictions.read_text().splitlines() if l.strip()]
    fit = read_json(a.fit_report)[0] if a.fit_report else None
    report = evaluate(dataset, preds, fit)
    report.update(schema='tnca-systemc-fidelity/1', dataset=dict(path=str(a.dataset), sha256=dsha),
                  predictions=dict(path=str(a.predictions), sha256=hashlib.sha256(a.predictions.read_bytes()).hexdigest()),
                  fit_report=str(a.fit_report) if a.fit_report else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    for split, s in report['splits'].items():
        if s['n']:
            print(f"{split:12s} n={s['n']:3d} median|err|={s['median_abs']:.3f} p95|err|={s['p95_abs']:.3f} "
                  f"mean err={s['mean_signed']:+.3f}")
    print('fidelity:', report['fidelity'], '-', report['why'])
    if report['unmatched']:
        print('no board record for:', ', '.join(report['unmatched'][:10]))


if __name__ == '__main__':
    main()
