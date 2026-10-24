#!/usr/bin/env python3
"""Physical TNCA_TUNE qualification on the 90 MHz board (R41 S1-03 B).

    tune_board_qualification.py inventory --tune-dir D --out first-pass.json
    tune_board_qualification.py check --tune-dir D --first first-pass.json \
        --first-records FIRST/formal.jsonl --rerun-records RERUN/formal.jsonl \
        --knob TNCA_TUNE=1 --knob TNCA_MAX_BLOCK=32 --knob TNCA_TUNE_CANDIDATES=4 --out report.json
    tune_board_qualification.py deadline-expiry --out DIR [--elements N]

`inventory` names the selection rows the first pass wrote (TNCA_TUNE_DIR/
runs/*/selection.json). `check` judges both passes: every formal record
passes numerically on the board with the registered knobs and contains only
the generated graph's launches (tuning samples isolated); the first pass
compiles at most the registered candidates within the graph budget, ranks at
least one kernel on three independent device-endpoint samples per candidate
and never loses the device; the rerun (a new process on the same persistent
cache) reuses every first-pass selection from the cache with the same
winner. `deadline-expiry` forces a device deadline inside a bounded
measurement scope on a real standard Torch graph, then requires a poisoned
Session with unchanged epoch, restored limits, refused stale handles after
explicit recovery, and a numerically correct graph on the new epoch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]


def selections(tune_dir, only=None, exclude=()):
    rows = {}
    for path in sorted((Path(tune_dir)/'runs').glob('*/selection.json')):
        name = path.parent.name
        if (only is None or name in only) and name not in exclude:
            rows[name] = json.loads(path.read_text())
    return rows


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def judge(first, rerun, first_records, rerun_records, *, knobs, max_candidates, max_seconds):
    """(problems, summary) for the two passes; problems empty means qualified."""
    problems = []
    for label, rows in (('first', first_records), ('rerun', rerun_records)):
        if not rows:
            problems.append(f'no {label}-pass formal records')
        for r in rows:
            plat = r.get('platform') or {}
            where = f'{label} {r.get("point_id")}'
            if r.get('evidence_level') != 'measured-board' or not str(plat.get('backend', '')).startswith('fpga-'):
                problems.append(f'{where}: not a board measurement')
            if r.get('status') != 'pass' or r.get('fallbacks'):
                problems.append(f'{where}: status {r.get("status")} fallbacks {r.get("fallbacks")}')
            if any((plat.get('knobs') or {}).get(k) != v for k, v in knobs.items()):
                problems.append(f'{where}: registered tuning knobs not in effect')
            graph = len(((r.get('provenance') or {}).get('launch_map') or {}).get('launches') or [])
            if not graph or (r.get('time') or {}).get('launches') != graph:
                problems.append(f'{where}: formal samples contain launches outside the generated graph')
    if sorted(r.get('point_id') for r in first_records) != sorted(r.get('point_id') for r in rerun_records):
        problems.append('the rerun did not repeat exactly the first-pass points')
    if not first:
        problems.append('the first pass wrote no selection rows: the public tuning consumer did not run')
    ranked = []
    for name, s in first.items():
        compiled = 1 + sum(1 for c in s.get('compilation') or [] if c.get('status') == 'compiled')
        if compiled > max_candidates:
            problems.append(f'{name}: {compiled} candidates exceed the registered {max_candidates}')
        if (s.get('tuning_seconds') or 0) > max_seconds:
            problems.append(f'{name}: tuning took {s["tuning_seconds"]:.1f}s, over the {max_seconds}s graph budget')
        if s.get('status') == 'device-lost':
            problems.append(f'{name}: tuning lost the device')
        if s.get('status') == 'selected':
            selection = s.get('selection') or {}
            measured = [row for row in selection.get('rows') or [] if row.get('status') == 'measured']
            if selection.get('cache_hit') is not False:
                problems.append(f'{name}: first pass did not measure (stale persistent cache)')
            if any(len(row.get('device_samples') or []) != 3 for row in measured):
                problems.append(f'{name}: a candidate lacks three independent device samples')
            if len(measured) >= 2:
                ranked.append(selection.get('identity_key'))
    if first and not ranked:
        problems.append('no kernel was ranked on qualified device endpoints (all conservative): '
                        + '; '.join(sorted({str(s.get('reason')) for s in first.values()})))
    first_keys = {s['selection']['identity_key']: s for s in first.values() if s.get('status') == 'selected'}
    hits = {}
    for name, s in rerun.items():
        if s.get('status') != 'selected':
            continue
        selection = s.get('selection') or {}
        key = selection.get('identity_key')
        if key not in first_keys:
            problems.append(f'{name}: rerun selected a kernel identity the first pass never tuned')
            continue
        if selection.get('cache_hit') is not True:
            problems.append(f'{name}: rerun measured again instead of using the persistent cache')
        elif selection.get('selected') != first_keys[key]['selection'].get('selected'):
            problems.append(f'{name}: cached winner differs from the first-pass winner')
        hits[key] = name
    for key in first_keys:
        if key not in hits:
            problems.append(f'first-pass selection {key[:12]} was not reused by the rerun')
    summary = dict(first_pass_rows=len(first), rerun_rows=len(rerun), ranked_kernels=len(ranked),
                   selected_first=len(first_keys), cache_hits=len(hits),
                   first_statuses=sorted(s.get('status') for s in first.values()),
                   conservative_reasons=sorted({str(s.get('reason')) for s in first.values()
                                                if s.get('status') != 'selected'}))
    return problems, summary


def inventory(args):
    rows = selections(args.tune_dir)
    args.out.write_text(json.dumps(dict(schema='tnca-tune-inventory/1', tune_dir=str(args.tune_dir),
        runs={name: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()
              for name, s in rows.items()}), indent=2)+'\n')
    print(f'{len(rows)} first-pass selection rows')


def check(args):
    known = json.loads(args.first.read_text())['runs']
    first = selections(args.tune_dir, only=set(known))
    if {n: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for n, s in first.items()} != known:
        raise ValueError('first-pass selection rows changed after their inventory')
    rerun = selections(args.tune_dir, exclude=set(known))
    knobs = dict(kv.split('=', 1) for kv in args.knob)
    problems, summary = judge(first, rerun, records(args.first_records), records(args.rerun_records),
                              knobs=knobs, max_candidates=args.max_candidates, max_seconds=args.max_seconds)
    status = 'tuning-board-qualified' if not problems else 'fail'
    report = dict(schema='tnca-tune-board-qualification/1', status=status, problems=problems, summary=summary,
                  knobs=knobs, max_candidates=args.max_candidates, max_seconds=args.max_seconds,
                  result=f'{summary["ranked_kernels"]} kernels ranked on device endpoints; '
                         f'{summary["cache_hits"]}/{summary["selected_first"]} selections reused from cache',
                  residual=None,
                  conclusion=('physical selection, cache reuse and sample isolation qualified on this image'
                              if not problems else 'tuning qualification failed: ' + '; '.join(problems[:3])),
                  next_step=('deadline-expiry poison/recovery check' if not problems
                             else 'fix and rerun; conservative default stays in effect'),
                  performance_gate='not-measured')
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=status, problems=problems[:5], summary=summary), indent=2))
    return 0 if not problems else 1


def expiry_scope(cycles, clock_hz):
    """Scope wall time whose device deadline (clock*wall/4) is 1/8 of the
    graph's measured cycles, while the host wait (wall/2) is twice that."""
    if type(cycles) is not int or cycles <= 0 or type(clock_hz) is not int or clock_hz <= 0:
        raise ValueError('positive measured cycles and clock required')
    wall = cycles/clock_hz/2
    return wall, int(clock_hz*wall/4)


def classify(error):
    text = f'{type(error).__name__}: {error}'
    if 'device deadline expired' in text:
        return 'device-deadline'
    if 'could not drain' in text:
        return 'bounded-drain-timeout'
    return 'other'


def deadline_expiry(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tnca-tune-deadline-expiry/1', status='running', elements=args.elements,
                  performance_gate='not-measured')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    save()
    try:
        backend = os.environ.get('TNCA_BACKEND', '')
        if not backend.startswith('fpga-'):
            raise ValueError('the deadline-expiry qualification runs only on a physical board backend')
        clock = json.loads(Path(os.environ['TNCA_OPERATING_CLOCK_EVIDENCE']).read_text())['clock_hz']
        os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
        os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_TENSOR_DESCRIPTORS='1',
                          TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
        sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]
        import torch
        import triton  # noqa: F401
        import tnca.torch_device as device
        from tnca.triton_backend.driver import launch_log
        from archbench.provenance import software_tree
        tree = software_tree(ROOT, refresh=True)
        device.register()
        torch.manual_seed(4011)
        a, b = torch.randn(args.elements), torch.randn(args.elements)
        reference = (a+b)*.5
        fn = torch.compile(lambda x, y: (x+y)*.5, backend='inductor', fullgraph=True, dynamic=False)
        da, db = a.to('tnca'), b.to('tnca')
        torch.testing.assert_close(fn(da, db).cpu(), reference, rtol=1e-5, atol=1e-5)
        session = device.MEMORY.session()
        session.drain()
        with launch_log(isolated=True) as launches:
            steady = fn(da, db)
        torch.testing.assert_close(steady.cpu(), reference, rtol=1e-5, atol=1e-5)
        cycles = sum(e['device_cycles'] for e in launches)
        wall, deadline = expiry_scope(cycles, clock)
        limits = (session.deadline_cycles, session.dev.deadline_cycles, session.dev.host_timeout)
        epoch = session.epoch
        report.update(software_tree=tree, backend=backend, clock_hz=clock, graph_cycles=cycles,
                      scope_wall_seconds=wall, scope_deadline_cycles=deadline, epoch_before=epoch,
                      limits_before=limits)
        save()
        error = None
        try:
            with session.measurement_scope(wall_seconds=wall, clock_hz=clock):
                fn(da, db)
        except Exception as caught:                          # noqa: BLE001
            error = caught
        report.update(forced_error=None if error is None else f'{type(error).__name__}: {error}',
                      expiry=None if error is None else classify(error), poisoned=session.poisoned,
                      epoch_after_expiry=session.epoch,
                      limits_after=(session.deadline_cycles, session.dev.deadline_cycles, session.dev.host_timeout))
        problems = []
        if error is None:
            problems.append('the forced deadline did not expire')
        if not session.poisoned:
            problems.append('the Session was not poisoned')
        if session.epoch != epoch:
            problems.append('the epoch changed without explicit recovery')
        if tuple(report['limits_after']) != tuple(limits):
            problems.append('bounded launch limits were not restored')
        new_epoch = session.recover()
        try:
            da.cpu()
            problems.append('a stale handle was accepted after recovery')
            stale = None
        except Exception as refused:                         # noqa: BLE001
            stale = f'{type(refused).__name__}: {refused}'
        da, db = a.to('tnca'), b.to('tnca')
        recovered = fn(da, db).cpu()
        torch.testing.assert_close(recovered, reference, rtol=1e-5, atol=1e-5)
        if new_epoch != epoch+1 or session.poisoned:
            problems.append('recovery did not open a new usable epoch')
        if software_tree(ROOT, refresh=True) != tree:
            problems.append('source changed during the check')
        report.update(epoch_after_recovery=new_epoch, stale_handle_refusal=stale, problems=problems,
                      recovered_output_sha256=hashlib.sha256(recovered.numpy().tobytes()).hexdigest(),
                      status='poison-and-recovery-pass' if not problems else 'fail',
                      result=f'{report["expiry"]} expiry after {deadline} of {cycles} cycles',
                      residual=None,
                      conclusion=('device deadline poisons the Session; explicit recovery restores correct execution'
                                  if not problems else '; '.join(problems)),
                      next_step='none' if not problems else 'fix the drain/recovery path before enabling tuning')
    except Exception:
        report.update(status='failed-preserved', error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({k: report.get(k) for k in ('status', 'expiry', 'poisoned', 'problems')}, indent=2))
    return 0 if report['status'] == 'poison-and-recovery-pass' else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory')
    p.add_argument('--tune-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('check')
    p.add_argument('--tune-dir', type=Path, required=True)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--first-records', type=Path, required=True)
    p.add_argument('--rerun-records', type=Path, required=True)
    p.add_argument('--knob', action='append', default=[])
    p.add_argument('--max-candidates', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('deadline-expiry')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--elements', type=int, default=1 << 25)
    args = ap.parse_args()
    if args.command == 'inventory':
        return inventory(args) or 0
    if args.command == 'check':
        return check(args)
    return deadline_expiry(args)


if __name__ == '__main__':
    raise SystemExit(main())
