#!/usr/bin/env python3
"""S1-06: fit on dev, calibrate on validation, score held-out ONCE.

    tools/r38/costmodel/evaluate.py            fit + calibrate + report dev/validation
    tools/r38/costmodel/evaluate.py --final    also score the held-out split

`--final` appends to sim/build/r38/dev/heldout_access.log; S1-04's split
check counts those lines, and more than one final evaluation per suite hash
is a protocol failure (doc S1-04 test 5, S1-06 test 4).

Acceptance (doc S1-06): held-out median relative error <= 20%, P90 <= 40%,
Spearman >= 0.7, 90% interval coverage in [80%, 98%]. The thresholds are
constants here and are not arguments.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import model as MODEL                                       # noqa: E402

DEV = ROOT / 'sim' / 'build' / 'r38' / 'dev'
GATE = dict(median=0.20, p90=0.40, spearman=0.70, coverage=(0.80, 0.98))


def load():
    recs = json.loads((DEV / 'records.json').read_text())
    suite = json.loads((ROOT / 'spec' / 'r38' / 'suites' / 'dev.json').read_text())
    meta = {p['id']: p for p in suite['points']}
    rows = []
    for r in recs:
        if r['status'] != 'pass' or r['repeat']['index'] != 0:
            continue
        p = meta[r['workload_id']]
        rows.append(dict(id=r['workload_id'], operator=r['operator'], shape=p['shape'],
                         split=p['split'], family=p['family'], cycles=r['time']['device_cycles']))
    return rows, suite


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def score(m, rows):
    preds, errs, inside, ood = [], [], 0, []
    for r in rows:
        try:
            p, lo, hi = m.predict_interval(r['operator'], r['shape'])
        except MODEL.OutOfDomain as e:
            ood.append(f'{r["id"]}: {e}')
            continue
        preds.append((p, r['cycles'], r['id']))
        errs.append(abs(p - r['cycles']) / r['cycles'])
        inside += lo <= r['cycles'] <= hi
    n = len(errs)
    if not n:
        return dict(n=0, out_of_domain=ood)
    return dict(n=n, median=quantile(errs, .5), p90=quantile(errs, .9),
                spearman=spearman([p for p, _, _ in preds], [t for _, t, _ in preds]),
                coverage=inside / n, out_of_domain=ood,
                worst=sorted(((abs(p - t) / t, i, round(p), t) for p, t, i in preds), reverse=True)[:5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    t0 = time.monotonic()
    rows, suite = load()
    by = {s: [r for r in rows if r['split'] == s] for s in ('dev', 'validation', 'heldout')}
    m = MODEL.Model().fit(by['dev']).calibrate(by['validation'])
    report = {'coefficients': m.coef, 'interval_rel_halfwidth_90': m.interval,
              'fit_seconds': round(time.monotonic() - t0, 3),
              'measurements_used': {'dev': len(by['dev']), 'validation': len(by['validation'])},
              'dev': score(m, by['dev']), 'validation': score(m, by['validation'])}
    verdict = None
    if args.final:
        suite_hash = hashlib.sha256((ROOT / 'spec' / 'r38' / 'suites' / 'dev.json').read_bytes()).hexdigest()
        log = DEV / 'heldout_access.log'
        prior = [l for l in (log.read_text().splitlines() if log.exists() else []) if suite_hash in l]
        with log.open('a') as fh:
            fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {suite_hash} costmodel-final\n')
        h = score(m, by['heldout'])
        report['heldout'] = h
        fails = []
        if prior:
            fails.append(f'held-out already scored {len(prior)} time(s) for this suite')
        if h['n'] == 0:
            fails.append('no held-out points in domain')
        else:
            if h['median'] > GATE['median']:
                fails.append(f'median {h["median"]:.3f} > {GATE["median"]}')
            if h['p90'] > GATE['p90']:
                fails.append(f'P90 {h["p90"]:.3f} > {GATE["p90"]}')
            if not h['spearman'] >= GATE['spearman']:
                fails.append(f'Spearman {h["spearman"]:.3f} < {GATE["spearman"]}')
            lo, hi = GATE['coverage']
            if not lo <= h['coverage'] <= hi:
                fails.append(f'coverage {h["coverage"]:.3f} outside [{lo}, {hi}]')
        verdict = 'ACCEPTED' if not fails else 'NOT ACCEPTED (model-limited)'
        report['verdict'], report['gate_failures'] = verdict, fails
    (DEV / 'costmodel_report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    for split in ('dev', 'validation', 'heldout'):
        s = report.get(split)
        if s and s.get('n'):
            print(f'{split:10} n={s["n"]:3} median={s["median"]:.3f} p90={s["p90"]:.3f} '
                  f'spearman={s["spearman"]:.3f} coverage90={s["coverage"]:.2f} ood={len(s["out_of_domain"])}')
    if verdict:
        print('S1-06:', verdict, *report['gate_failures'])
        return 0 if verdict == 'ACCEPTED' else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
