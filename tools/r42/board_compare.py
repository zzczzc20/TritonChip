#!/usr/bin/env python3
"""Compare two board runs of the 113 original ArchBench points, from their record files only.

    board_compare.py --new RECORDS.tgz [--old RECORDS.tgz] [--out comparison.json] [--threshold 0.02]

Each side is a records archive (or directory) holding one `*/original/formal.jsonl` and its `batch.json`, as
written by tools/performance_harness/board_batch.py: evidence/r41/hbm_bw/g4/run3/board/original113_records.tgz
(the default --old, the R41 run-3 board records) or evidence/r42/board/NAME/records.tgz (tools/r42/board_ab.sh
fetch). Domain of every number: board, U280, device-cycles-serial-grid at the qualified clock recorded in the
records. Per point the value is the median of the formal samples (3 samples for points at <= 1 s device time,
1 sample otherwise). Nothing is typed in; a point missing or failed on either side is listed, never dropped,
and then the geometric mean is labelled as covering the common passing points only.

When --old is the run-3 archive its medians are also checked against
evidence/r41/hbm_bw/g4/run3/board/original113_comparison.json (`new_cycles`).
"""
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUN3 = ROOT/'evidence/r41/hbm_bw/g4/run3/board/original113_records.tgz'
RUN3_CMP = ROOT/'evidence/r41/hbm_bw/g4/run3/board/original113_comparison.json'


def geo(xs):
    xs = list(xs)
    return math.exp(sum(map(math.log, xs))/len(xs)) if xs else None


def read_side(path):
    """{relative name: bytes} of formal.jsonl and batch.json of the suite batch."""
    path = Path(path)
    found = {}
    if path.is_dir():
        for name in ('formal.jsonl', 'batch.json'):
            hits = sorted(path.glob(f'**/original/{name}'))
            if len(hits) != 1:
                raise SystemExit(f'{path}: expected one original/{name}, found {len(hits)}')
            found[name] = hits[0].read_bytes()
        digest = None
    else:
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                for name in ('formal.jsonl', 'batch.json'):
                    if member.name.endswith('/original/' + name):
                        if name in found:
                            raise SystemExit(f'{path}: more than one original/{name}')
                        found[name] = tar.extractfile(member).read()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(found) != {'formal.jsonl', 'batch.json'}:
        raise SystemExit(f'{path}: no original/formal.jsonl + batch.json')
    batch = json.loads(found['batch.json'])
    points = {}
    for line in io.StringIO(found['formal.jsonl'].decode()):
        if not line.strip():
            continue
        r = json.loads(line)
        t = r['timings']['device-cycles-serial-grid']
        corr = r.get('correctness') or {}
        prov = r.get('provenance') or {}
        if r['point_id'] in points:
            raise SystemExit(f'{path}: duplicate formal record {r["point_id"]}')
        points[r['point_id']] = dict(
            point_sha256=r['point_sha256'], status=r['status'], fallbacks=r.get('fallbacks'),
            family=r['family'], tier=r['tier'], samples=t['samples'], cycles=t['median'], clock_hz=t['clock_hz'],
            max_rel_err=corr.get('max_rel_err'), tol=corr.get('tol'),
            artifacts=sorted(prov.get('artifacts') or []),
            launches=len((prov.get('launch_map') or {}).get('launches') or []),
            software_tree=(r.get('platform') or {}).get('software_tree'),
            image_sha256=(r.get('platform') or {}).get('image_sha256'),
            knobs=(r.get('platform') or {}).get('knobs'))
    return dict(path=str(path), archive_sha256=digest, batch_status=batch['status'],
                configurations=batch['configurations'], failed_configurations=batch['failed_configurations'],
                missing_formal=batch['missing_formal'], software_tree=batch['software_tree'],
                image_sha256=batch['board']['image_sha256'], clock_hz=batch['board']['clock_hz'],
                board_wall_seconds=batch.get('board_wall_seconds'), points=points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', type=Path, default=RUN3)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--threshold', type=float, default=0.02)
    a = ap.parse_args()
    old, new = read_side(a.old), read_side(a.new)
    notes = []
    if a.old.resolve() == RUN3.resolve():
        ref = {r['point']: r['new_cycles'] for r in json.loads(RUN3_CMP.read_text())['rows']}
        mismatch = sorted(p for p in ref if old['points'].get(p, {}).get('cycles') != ref[p])
        if mismatch or len(ref) != len(old['points']):
            raise SystemExit(f'run-3 records disagree with original113_comparison.json: {mismatch[:5]}')
        notes.append('old medians equal original113_comparison.json new_cycles for all 113 points')
    if old['clock_hz'] != new['clock_hz']:
        raise SystemExit('the two runs have different clocks; cycles are not comparable as device time')
    rows, problems = [], []
    for point in sorted(set(old['points']) | set(new['points'])):
        o, n = old['points'].get(point), new['points'].get(point)
        if o is None or n is None:
            problems.append(dict(point=point, problem='no formal record in ' + ('old' if o is None else 'new')))
            continue
        if o['point_sha256'] != n['point_sha256']:
            problems.append(dict(point=point, problem='point definition differs'))
            continue
        bad = [side for side, r in (('old', o), ('new', n))
               if r['status'] != 'pass' or r['fallbacks'] or r['max_rel_err'] is None or r['max_rel_err'] > r['tol']]
        if bad:
            problems.append(dict(point=point, problem='not a passing record in ' + '+'.join(bad)))
            continue
        spread = lambda r: (max(r['samples'])-min(r['samples']))/r['cycles'] if len(r['samples']) > 1 else None
        rows.append(dict(point=point, point_sha256=o['point_sha256'], family=o['family'], tier=o['tier'],
                         old_cycles=o['cycles'], new_cycles=n['cycles'], new_over_old=n['cycles']/o['cycles'],
                         old_samples=o['samples'], new_samples=n['samples'],
                         old_spread=spread(o), new_spread=spread(n),
                         same_artifacts=o['artifacts'] == n['artifacts'],
                         old_launches=o['launches'], new_launches=n['launches'],
                         old_max_rel_err=o['max_rel_err'], new_max_rel_err=n['max_rel_err'], tol=n['tol']))

    def group(key):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[key]].append(r['new_over_old'])
        return {k: dict(points=len(v), geomean_new_over_old=geo(v)) for k, v in sorted(g.items())}

    ratios = [r['new_over_old'] for r in rows]
    differing = sorted((r for r in rows if abs(r['new_over_old']-1) > a.threshold), key=lambda r: -abs(math.log(r['new_over_old'])))
    same = [r for r in rows if r['same_artifacts']]
    changed = [r for r in rows if not r['same_artifacts']]
    spreads = [s for r in rows for s in (r['old_spread'], r['new_spread']) if s is not None]
    complete = (not problems and len(rows) == 113 and old['batch_status'] == new['batch_status'] == 'complete')
    summary = dict(
        schema='r42-board-original113-comparison/1', domain='board (U280), device-cycles-serial-grid',
        scope='complete: all 113 points pass on both sides' if complete else
              'INCOMPLETE: geometric means cover only the points that pass on both sides; see problems',
        old={k: v for k, v in old.items() if k != 'points'}, new={k: v for k, v in new.items() if k != 'points'},
        same_image=old['image_sha256'] == new['image_sha256'], same_software_tree=old['software_tree'] == new['software_tree'],
        knobs_old=sorted({json.dumps(p['knobs'], sort_keys=True) for p in old['points'].values()}),
        knobs_new=sorted({json.dumps(p['knobs'], sort_keys=True) for p in new['points'].values()}),
        points_compared=len(rows), problems=problems, notes=notes,
        geomean_new_over_old=geo(ratios), speedup_geomean_old_over_new=(1/geo(ratios) if ratios else None),
        median_new_over_old=statistics.median(ratios) if ratios else None,
        min_new_over_old=min(ratios, default=None), max_new_over_old=max(ratios, default=None),
        old_total_cycles=sum(r['old_cycles'] for r in rows), new_total_cycles=sum(r['new_cycles'] for r in rows),
        threshold=a.threshold, points_differing_above_threshold=len(differing),
        slower_above_threshold=[r['point'] for r in differing if r['new_over_old'] > 1],
        faster_above_threshold=[r['point'] for r in differing if r['new_over_old'] < 1],
        same_artifact_points=dict(points=len(same), geomean_new_over_old=geo(r['new_over_old'] for r in same),
                                  max_abs_deviation=max((abs(r['new_over_old']-1) for r in same), default=None)),
        changed_artifact_points=dict(points=len(changed), geomean_new_over_old=geo(r['new_over_old'] for r in changed)),
        within_record_sample_spread=dict(records=len(spreads), median=statistics.median(spreads) if spreads else None,
                                         max=max(spreads, default=None),
                                         meaning='(max-min)/median of the 3 samples of one formal record'),
        by_tier=group('tier'), by_family=group('family'), rows=rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(summary['scope'])
    print(f"old tree {old['software_tree']}  new tree {new['software_tree']}  same image {summary['same_image']}")
    print(f"points compared {len(rows)}  problems {len(problems)}")
    if ratios:
        print(f"geomean new/old {summary['geomean_new_over_old']:.4f}  median {summary['median_new_over_old']:.4f}  "
              f"min {min(ratios):.4f}  max {max(ratios):.4f}")
        print(f"same compiled artifacts: {len(same)} points, geomean {summary['same_artifact_points']['geomean_new_over_old']}; "
              f"changed artifacts: {len(changed)} points, geomean {summary['changed_artifact_points']['geomean_new_over_old']}")
        print('by tier: ' + ', '.join(f"{k} {v['geomean_new_over_old']:.4f} ({v['points']})" for k, v in summary['by_tier'].items()))
        print(f"points differing by more than {a.threshold:.0%}: {len(differing)}")
        for r in differing:
            print(f"  {r['new_over_old']:.4f}  {r['old_cycles']:>14} -> {r['new_cycles']:>14}  "
                  f"{'same-art' if r['same_artifacts'] else 'new-art '}  {r['point']}")
    for p in problems:
        print('PROBLEM', p['point'], p['problem'])


if __name__ == '__main__':
    main()
