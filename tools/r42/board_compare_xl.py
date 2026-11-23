#!/usr/bin/env python3
"""Compare two board runs of the xl ArchBench tier, from their record files only.

    board_compare_xl.py --old BASE_RECORDS.tgz --new CAND_RECORDS.tgz [--out comparison.json]

Each side is a records archive (or an unpacked directory) written by `tools/r42/board_ab.sh fetch` for a registration
made with `board_ab.py register --suite xl`: `queue/<id>/xl/{batch.json,state.json,formal.jsonl,runs/*}`.
Domain of every number: board, U280, device-cycles-serial-grid at the qualified clock recorded in the records, the
median of the formal samples of the point's formal record (3 samples for a point at <= 1 s device time, else 1).
Nothing is typed in. Per point and side the outcome is read from the run entries of `state.json` and the raw ArchBench
records:

  pass      a formal record exists and every run of the point passed the batch's checks (numerics inside the frozen
            tolerance, no host fallback, identity of image/clock/tree/knobs, sampling)
  fail      a run of the point failed; the cause is read from the run's problems and its raw record status
            (timeout, numerical, capacity-exceeded, runtime-fault, unsupported, ...). Numerical failures keep their
            measured max_rel_err and tolerance.
  not-run   the point has no run entry (the batch stopped earlier)

A point that is not `pass` on both sides is listed as failed and excluded from the ratios; the geometric means then
cover the common passing points only and the result is labelled NOT CERTIFIED. It is certified only when every
registered point passes on both sides and both batches are complete.
"""
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

FILES = re.compile(r'(?:^|/)queue/([^/]+)/xl/(state\.json|batch\.json|formal\.jsonl|runs/[^/]+)$')


def geo(xs):
    xs = list(xs)
    return math.exp(sum(map(math.log, xs))/len(xs)) if xs else None


def fp8_family(family):
    return family.startswith('fp8_')


def read_files(path):
    """({name: bytes}, queue id, archive sha256)."""
    path = Path(path)
    found, ids = {}, set()
    if path.is_dir():
        for f in sorted(p for p in path.rglob('*') if p.is_file()):
            m = FILES.search(f.as_posix())
            if m:
                ids.add(m.group(1))
                found[m.group(2)] = f.read_bytes()
        digest = None
    else:
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                m = FILES.search(member.name) if member.isfile() else None
                if m:
                    ids.add(m.group(1))
                    found[m.group(2)] = tar.extractfile(member).read()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(ids) != 1 or 'state.json' not in found:
        raise SystemExit(f'{path}: expected the records of exactly one xl batch with a state.json, found {sorted(ids)}')
    return found, ids.pop(), digest


def jsonl(data):
    return [json.loads(l) for l in io.StringIO(data.decode()) if l.strip()]


def classify_failure(entry, raw):
    """(kind, detail) of a failed run from its problems and raw record."""
    problems = ' | '.join(entry.get('problems') or [])
    status = (raw or {}).get('status')
    corr = (raw or {}).get('correctness') or {}
    if 'budget' in problems and 'exhausted' in problems:
        return 'timeout', problems
    if status == 'numeric-error' or (corr.get('max_rel_err') is not None and corr.get('tol') is not None
                                     and corr['max_rel_err'] > corr['tol']):
        return 'numerical', f"max_rel_err {corr.get('max_rel_err')} > tol {corr.get('tol')}; {problems}"
    if status and status != 'pass':
        raw_problems = '; '.join(str(p)[:300] for p in (raw.get('problems') or []))
        return status, raw_problems or problems
    return 'other', problems


def read_side(path):
    files, qid, digest = read_files(path)
    state = json.loads(files['state.json'])
    batch = json.loads(files['batch.json']) if 'batch.json' in files else None
    formal = {}
    if 'formal.jsonl' in files:
        for r in jsonl(files['formal.jsonl']):
            formal[r['point_id']] = r
    runs = collections.defaultdict(list)
    for stem, entry in state['runs'].items():
        runs[entry['config']].append(dict(entry, stem=stem))
    raws = {name.split('/', 1)[1]: data for name, data in files.items() if name.startswith('runs/')}

    def raw_of(entry):
        data = raws.get(Path(entry['raw']).name)
        rows = jsonl(data) if data else []
        return rows[0] if len(rows) == 1 else None

    order = list(batch['formal']) if batch else sorted(runs)
    points = {}
    for cid in order:
        mine = sorted(runs.get(cid, []), key=lambda e: e['started'])
        row = dict(point=cid, runs=[dict(tag=e['tag'], ok=e['ok'], wall_seconds=e['wall_seconds'],
                                         problems=e['problems']) for e in mine])
        failing = [e for e in mine if not e['ok']]
        rec = formal.get(cid)
        shown = rec or (raw_of(failing[-1]) if failing else (raw_of(mine[-1]) if mine else None))
        if shown:
            corr = shown.get('correctness') or {}
            nm = shown.get('native_memory') or {}
            row.update(family=shown.get('family'), tier=shown.get('tier'), status=shown.get('status'),
                       max_rel_err=corr.get('max_rel_err'), tol=corr.get('tol'),
                       eta=shown.get('eta'), bound_cycles=(shown.get('bound') or {}).get('value'),
                       bound_binding=(shown.get('bound') or {}).get('binding'),
                       heap_peak_bytes=((nm.get('session') or {}).get('heap_peak_bytes')),
                       heap_capacity_bytes=((nm.get('session') or {}).get('heap_capacity_bytes')),
                       tensor_peak_bytes=nm.get('tensor_peak_bytes'),
                       launches=len(((shown.get('provenance') or {}).get('launch_map') or {}).get('launches') or []),
                       artifacts=sorted((shown.get('provenance') or {}).get('artifacts') or []),
                       first_call_s=((shown.get('host') or {}).get('first_call_s')))
        if not mine:
            row.update(outcome='not-run')
        elif failing or rec is None:
            last = failing[-1] if failing else mine[-1]
            kind, detail = classify_failure(last, raw_of(last))
            row.update(outcome='fail', failure_kind=kind, failure_detail=detail[:600],
                       failed_run=last['stem'])
        else:
            t = rec['timings']['device-cycles-serial-grid']
            row.update(outcome='pass', cycles=t['median'], samples=t['samples'], clock_hz=t['clock_hz'],
                       point_sha256=rec['point_sha256'], software_tree=(rec.get('platform') or {}).get('software_tree'),
                       image_sha256=(rec.get('platform') or {}).get('image_sha256'))
        points[cid] = row
    meta = dict(path=str(path), archive_sha256=digest, queue_id=qid,
                batch_status=(batch or {}).get('status', 'no batch.json (batch did not finish)'),
                configurations=(batch or {}).get('configurations'),
                software_tree=(batch or state).get('software_tree'),
                image_sha256=(batch or {}).get('board', {}).get('image_sha256') or state.get('image_sha256'),
                clock_hz=(batch or {}).get('board', {}).get('clock_hz') or state.get('clock_hz'),
                runs=len(state['runs']), failed_runs=sum(1 for e in state['runs'].values() if not e['ok']),
                process_wall_seconds=round(sum(e['wall_seconds'] for e in state['runs'].values()), 1),
                sampling_cap=state.get('xl_sampling_cap') or {},
                outcomes=dict(collections.Counter(p['outcome'] for p in points.values())))
    return meta, points


def single(a):
    """One side alone: the per-point table (cycles, eta, tolerance margin, memory) and the failures, same rules."""
    meta, pts = read_side(a.new)
    rows, failed = [], []
    for cid, p in pts.items():
        if p['outcome'] != 'pass':
            failed.append(dict(point=cid, family=p.get('family'), **{k: p.get(k) for k in (
                'outcome', 'failure_kind', 'failure_detail', 'status', 'max_rel_err', 'tol', 'failed_run', 'runs')
                if p.get(k) is not None}))
            continue
        rows.append(dict(point=cid, family=p['family'], cycles=p['cycles'], samples=p['samples'],
                         seconds=p['cycles']/p['clock_hz'], eta=p.get('eta'), bound_cycles=p.get('bound_cycles'),
                         bound_binding=p.get('bound_binding'), max_rel_err=p['max_rel_err'], tol=p['tol'],
                         err_over_tol=(p['max_rel_err']/p['tol'] if p['tol'] else None),
                         launches=p['launches'], heap_peak_bytes=p.get('heap_peak_bytes'),
                         first_call_s=p.get('first_call_s'),
                         wall_seconds=sum(r['wall_seconds'] for r in p['runs'])))
    fams = collections.defaultdict(list)
    for r in rows:
        fams[r['family']].append(r)
    summary = dict(schema='r42-board-xl-single/1', domain='board (U280), device-cycles-serial-grid', meta=meta,
                   scope=('all %d registered points pass' % len(pts)) if not failed else
                         ('%d of %d registered points do not pass' % (len(failed), len(pts))),
                   rows=rows, failed_points=failed,
                   total_device_seconds=sum(r['seconds'] for r in rows),
                   by_family={k: dict(points=len(v), device_seconds=sum(x['seconds'] for x in v),
                                      geomean_eta=geo(x['eta'] for x in v if x['eta']))
                              for k, v in sorted(fams.items())},
                   max_heap_peak_bytes=max((r['heap_peak_bytes'] or 0 for r in rows), default=None),
                   max_err_over_tol=max((r['err_over_tol'] or 0 for r in rows), default=None))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(summary['scope'], '| batch', meta['batch_status'], '| tree', meta['software_tree'])
    print(f"runs {meta['runs']} (failed {meta['failed_runs']}), process wall {meta['process_wall_seconds']} s, "
          f"device seconds of the formal records {summary['total_device_seconds']:.1f}, "
          f"max heap peak {summary['max_heap_peak_bytes']}, max err/tol {summary['max_err_over_tol']:.3f}")
    print('point                                                          cycles    seconds   eta     err/tol  samples')
    for r in rows:
        print(f"  {r['point']:62s} {r['cycles']:>14} {r['seconds']:>9.3f}  {r['eta'] or 0:.3f}  "
              f"{r['err_over_tol'] if r['err_over_tol'] is not None else 0:.3f}  {len(r['samples'])}")
    for f in failed:
        print('FAILED', f['point'], json.dumps({k: v for k, v in f.items() if k not in ('point', 'runs')}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', type=Path, help='baseline records (xl-base); omitted: report the --new side alone')
    ap.add_argument('--new', type=Path, required=True, help='candidate records (xl-cand)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--threshold', type=float, default=0.02)
    a = ap.parse_args()
    if a.old is None:
        return single(a)
    (old_meta, old), (new_meta, new) = read_side(a.old), read_side(a.new)
    if old_meta['clock_hz'] != new_meta['clock_hz'] or old_meta['image_sha256'] != new_meta['image_sha256']:
        raise SystemExit('the two runs have a different image or clock; cycles are not comparable as device time')
    ids = list(old) + [p for p in new if p not in old]
    rows, failed = [], []
    for cid in ids:
        o = old.get(cid, dict(outcome='not-run'))
        n = new.get(cid, dict(outcome='not-run'))
        fam = o.get('family') or n.get('family')
        if o['outcome'] != 'pass' or n['outcome'] != 'pass':
            failed.append(dict(point=cid, family=fam,
                               **{side: {k: r.get(k) for k in ('outcome', 'failure_kind', 'failure_detail', 'status',
                                                               'max_rel_err', 'tol', 'failed_run', 'runs')
                                         if r.get(k) is not None}
                                  for side, r in (('old', o), ('new', n))}))
            continue
        if o['point_sha256'] != n['point_sha256']:
            failed.append(dict(point=cid, family=fam, old=dict(outcome='definition-differs'), new={}))
            continue
        spread = lambda r: (max(r['samples'])-min(r['samples']))/r['cycles'] if len(r['samples']) > 1 else None
        rows.append(dict(point=cid, family=fam, fp8=fp8_family(fam), old_cycles=o['cycles'], new_cycles=n['cycles'],
                         new_over_old=n['cycles']/o['cycles'], old_samples=o['samples'], new_samples=n['samples'],
                         old_spread=spread(o), new_spread=spread(n), old_eta=o.get('eta'), new_eta=n.get('eta'),
                         bound_cycles=o.get('bound_cycles'), bound_binding=o.get('bound_binding'),
                         same_artifacts=o['artifacts'] == n['artifacts'], old_launches=o['launches'],
                         new_launches=n['launches'], old_max_rel_err=o['max_rel_err'], new_max_rel_err=n['max_rel_err'],
                         tol=n['tol'], old_heap_peak_bytes=o.get('heap_peak_bytes'),
                         new_heap_peak_bytes=n.get('heap_peak_bytes')))

    def groups(key, rows):
        g = collections.defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        return {k: dict(points=len(v), geomean_new_over_old=geo(x['new_over_old'] for x in v),
                        old_cycles=sum(x['old_cycles'] for x in v), new_cycles=sum(x['new_cycles'] for x in v))
                for k, v in sorted(g.items())}

    ratios = [r['new_over_old'] for r in rows]
    registered = len(ids)
    certified = (not failed and len(rows) == registered and old_meta['batch_status'] == new_meta['batch_status'] == 'complete')
    fp8 = [r for r in rows if r['fp8'] or r['family'] in ('fp8_linear', 'fp8_act_quant')]
    summary = dict(
        schema='r42-board-xl-comparison/1', domain='board (U280), device-cycles-serial-grid',
        scope=('CERTIFIED: all %d registered points pass on both sides' % registered) if certified else
              ('NOT CERTIFIED: %d of %d registered points do not pass on both sides; geometric means cover the %d '
               'common passing points only' % (len(failed), registered, len(rows))),
        old=old_meta, new=new_meta, registered_points=registered,
        same_image=old_meta['image_sha256'] == new_meta['image_sha256'],
        software_trees=dict(old=old_meta['software_tree'], new=new_meta['software_tree']),
        points_compared=len(rows), failed_points=failed,
        geomean_new_over_old=geo(ratios), speedup_old_over_new=(1/geo(ratios) if ratios else None),
        median_new_over_old=statistics.median(ratios) if ratios else None,
        min_new_over_old=min(ratios, default=None), max_new_over_old=max(ratios, default=None),
        old_total_cycles=sum(r['old_cycles'] for r in rows), new_total_cycles=sum(r['new_cycles'] for r in rows),
        threshold=a.threshold,
        slower_above_threshold=[r['point'] for r in rows if r['new_over_old'] > 1+a.threshold],
        faster_above_threshold=[r['point'] for r in rows if r['new_over_old'] < 1-a.threshold],
        by_family=groups(lambda r: r['family'], rows),
        by_class=groups(lambda r: 'fp8 matmul based' if r['fp8'] else 'fp16', rows),
        max_heap_peak_bytes=dict(old=max((r['old_heap_peak_bytes'] or 0 for r in rows), default=None),
                                 new=max((r['new_heap_peak_bytes'] or 0 for r in rows), default=None)),
        rows=rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(summary['scope'])
    for tag, meta in (('old', old_meta), ('new', new_meta)):
        print(f"{tag}: tree {meta['software_tree']} batch {meta['batch_status']} runs {meta['runs']} "
              f"(failed {meta['failed_runs']}) process wall {meta['process_wall_seconds']} s outcomes {meta['outcomes']}")
    if ratios:
        print(f"points compared {len(rows)}  geomean new/old {summary['geomean_new_over_old']:.4f}  "
              f"median {summary['median_new_over_old']:.4f}  min {min(ratios):.4f}  max {max(ratios):.4f}")
        for k, v in summary['by_class'].items():
            print(f"  class {k}: {v['points']} points, geomean new/old {v['geomean_new_over_old']:.4f}")
        for k, v in summary['by_family'].items():
            print(f"  family {k:20s} {v['points']:2d} points  geomean new/old {v['geomean_new_over_old']:.4f}")
        print('point                                                          old cycles      new cycles   new/old')
        for r in rows:
            print(f"  {r['point']:62s} {r['old_cycles']:>14} {r['new_cycles']:>14}  {r['new_over_old']:.4f}")
    for f in failed:
        print('FAILED', f['point'], json.dumps(f['old']), json.dumps(f['new']))


if __name__ == '__main__':
    main()
