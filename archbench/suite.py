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
SUITES = HERE / 'suites'


def build(tier):
    pts = registry.points(tier=tier)
    entries = [dict(id=p.id, sha256=p.definition_sha256(), kind=p.kind, family=p.family,
                    dtype=p.dtype, track=list(p.track)) for p in pts]
    blob = json.dumps(entries, sort_keys=True)
    return dict(schema='archbench-suite/1', tier=tier, points=entries,
                families=sorted({p.family for p in pts}),
                sha256=hashlib.sha256(blob.encode()).hexdigest())


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
        path = SUITES / f'{tier}.json'
        if args.freeze:
            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
            print(f'{tier}: {len(doc["points"])} points, {len(doc["families"])} families -> {path}')
            continue
        if not path.exists():
            print(f'{tier}: no frozen suite at {path}')
            bad += 1
            continue
        frozen = json.loads(path.read_text())
        if frozen['sha256'] == doc['sha256']:
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
