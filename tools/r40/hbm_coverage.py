#!/usr/bin/env python3
"""R40-00: the HBM suite's capacity, compile coverage and measurement, recorded.

    python tools/r40/hbm_coverage.py COMPILE_ONLY.jsonl [--image-heap BYTES] \
        [--out spec/r40/hbm_coverage.json]

doc/r40_tasks_and_acceptance.md R40-00: the HBM suite is frozen before any
R40 optimisation, and for its points "分别记录容量、编译覆盖与实测" --
capacity, compile coverage and measurement, each on its own, without
shrinking an original large shape. The input is an ArchBench compile-only
sweep (`archbench.run --compile-only`: every kernel compiled through the
standard path, none launched) of the large and hbm tiers. For each point
this records whether it compiled (and why not), the largest launch's
argument bytes against the B39 image's heap, and that it was not measured;
R40-04 onward replace `measured` as points run on HBM.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('records', type=Path)
    ap.add_argument('--image-heap', type=int, default=0x80000 - 0x20000,
                    help='the B39 image\'s device heap (archbench/platforms/tnca.py HEAP_BYTES)')
    ap.add_argument('--out', type=Path, default=ROOT / 'spec' / 'r40' / 'hbm_coverage.json')
    args = ap.parse_args(argv)
    from archbench import registry
    recs = {}
    for line in args.records.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r['point_id']] = r
    want = registry.points(tier=('large', 'hbm'))
    rows, missing = [], []
    for p in want:
        r = recs.get(p.id)
        if r is None:
            missing.append(p.id)
            continue
        if r.get('point_sha256') != p.definition_sha256():
            sys.exit(f'{p.id}: the sweep ran another definition')
        h = r.get('host') or {}
        st = r['status']
        # archbench/platforms/tnca.py, compile-only: not-measured when every
        # kernel built, capacity-exceeded when they built and the largest
        # launch is bigger than the heap; anything else did not compile.
        compiled = st in ('not-measured', 'capacity-exceeded')
        biggest = h.get('max_launch_bytes')
        rows.append(dict(
            id=p.id, sha256=p.definition_sha256(), tier=p.tier, family=p.family,
            working_set_bytes=(p.demand.read_bytes or 0) + (p.demand.write_bytes or 0),
            compile=dict(ok=compiled, status=st, launches=h.get('launches'),
                         kernels=len(r.get('kernels') or []),
                         problem=None if compiled else (r.get('problems') or [None])[0]),
            capacity=dict(max_launch_bytes=biggest, max_launch_kernel=h.get('max_launch_kernel'),
                          b39_heap_bytes=args.image_heap,
                          fits_b39=None if biggest is None else biggest <= args.image_heap),
            measured=dict(status='not-measured', why='no TNCA image with external memory yet (R40-04)')))
    if missing:
        sys.exit(f'{len(missing)} large/hbm point(s) missing from the sweep, e.g. {missing[:3]}')
    summary = dict(points=len(rows), compiled=sum(r['compile']['ok'] for r in rows),
                   fits_b39=sum(bool(r['capacity']['fits_b39']) for r in rows),
                   measured=0)
    doc = dict(schema='tnca-r40-hbm-coverage/1',
               what='capacity, compile coverage and measurement of the large and hbm points '
                    '(the existing tiers are measured in B39 itself)',
               sweep=dict(source=args.records.name, sha256=hashlib.sha256(args.records.read_bytes()).hexdigest(),
                          platform=next(iter(recs.values()))['platform'] if recs else None),
               summary=summary, points=rows)
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    print(f"hbm coverage: {summary['compiled']}/{summary['points']} compile, "
          f"{summary['fits_b39']} fit the B39 heap, {summary['measured']} measured -> {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
