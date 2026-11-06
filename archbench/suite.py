"""Frozen suites: what each tier contains, by content hash.

    python -m archbench.suite --freeze      # write archbench/suites/<tier>.json
    python -m archbench.suite --check       # exit 1 if the registry drifted

A suite file lists every point of a tier with the hash of its definition
(id, parameters, dtype, tolerance, demand, track). Changing a point -- a
shape, a tolerance, the demand that bounds it -- changes its hash, so a
result can always be traced to the definition it was measured against and a
silent edit cannot pass the check. Results are matched to points by id AND
hash (`point_sha256` in every record).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUITES = HERE / 'suites'
# A tier whose frozen file carries a registered name rather than the tier's.
NAMES = {'membound': 'r41_membound'}
# The R41 memory-bound suite's non-regression set (BANDWIDTH_PLAN.md section 3,
# discipline section 2.3.4): the 113 original points of the R40 HBM suite, among
# them every GEMM point, which cannot show a bandwidth effect but must not slow.
NON_REGRESSION = 'spec/r40/suites/hbm.json'
GEMM_FAMILIES = ('matmul', 'linear', 'gemv', 'bmm')


def path(tier):
    return SUITES / f'{NAMES.get(tier, tier)}.json'


def non_regression():
    f = ROOT / NON_REGRESSION
    doc = json.loads(f.read_text())
    return dict(suite=NON_REGRESSION, file_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                suite_sha256=doc['sha256'], points=len(doc['points']),
                gemm_families=list(GEMM_FAMILIES),
                gemm_points=sorted(p['id'] for p in doc['points'] if p['family'] in GEMM_FAMILIES),
                baseline='s2-original-suite on the run-1 100 MHz image (evidence/r41/stage2/board/results.md)')


def build(tier):
    pts = registry.points(tier=tier)
    entries = [dict(id=p.id, sha256=p.definition_sha256(), kind=p.kind, family=p.family,
                    dtype=p.dtype, track=list(p.track)) for p in pts]
    blob = json.dumps(entries, sort_keys=True)
    doc = dict(schema='archbench-suite/1', tier=tier, points=entries,
               families=sorted({p.family for p in pts}),
               sha256=hashlib.sha256(blob.encode()).hexdigest())
    if tier == 'membound':
        doc.update(name=NAMES[tier], rules='archbench/membound.py (module docstring)',
                   plan='evidence/r41/hbm_bw/BANDWIDTH_PLAN.md section 3',
                   status='frozen: registered before any design change is measured; every point is '
                          'reported, favourable or not; a changed definition is a different suite',
                   non_regression=non_regression())
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)
    SUITES.mkdir(exist_ok=True)
    bad = 0
    for tier in registry.TIERS:
        doc = build(tier)
        file = path(tier)
        if args.freeze:
            file.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
            print(f'{tier}: {len(doc["points"])} points, {len(doc["families"])} families -> {file}')
            continue
        if not file.exists():
            print(f'{tier}: no frozen suite at {file}')
            bad += 1
            continue
        frozen = json.loads(file.read_text())
        if frozen['sha256'] == doc['sha256'] and frozen.get('non_regression') == doc.get('non_regression'):
            print(f'{tier}: matches the frozen suite ({len(doc["points"])} points)')
            continue
        bad += 1
        old = {e['id']: e['sha256'] for e in frozen['points']}
        new = {e['id']: e['sha256'] for e in doc['points']}
        for i in sorted(set(old) | set(new)):
            if old.get(i) != new.get(i):
                what = 'added' if i not in old else 'removed' if i not in new else 'changed'
                print(f'{tier}: {what} {i}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
