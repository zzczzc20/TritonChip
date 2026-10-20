#!/usr/bin/env python3
"""Run every frozen multi-kernel transfer audit in isolated child processes.

The default is the entire R40-02.graphs3 inventory. --point selects a
diagnostic subset; its report cannot claim full coverage. These audits
establish transfer correctness, not request latency or KV growth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / 'spec/r40/suites/R40-02.graphs3.json'
sys.path.insert(0, str(ROOT))
from archbench.provenance import software_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', action='append')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=7 * 24 * 3600)
    parser.add_argument('--host-timeout', type=float, default=172800,
                        help='host seconds per launch; device cycle deadline is unchanged')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'sim/build/r40/frozen_residency_suite')
    args = parser.parse_args()
    if args.jobs < 1 or args.steps < 2 or args.timeout <= 0 or args.host_timeout <= 0:
        parser.error('jobs >= 1, steps >= 2 and positive timeouts are required')
    frozen = json.loads(SUBSET.read_text())
    by_id = {p['id']: p for p in frozen['points']}
    if len(by_id) != len(frozen['points']):
        raise RuntimeError('duplicate frozen point')
    if args.point and (len(set(args.point)) != len(args.point) or
                       set(args.point) - by_id.keys()):
        parser.error('--point must name distinct frozen points')
    points = [by_id[p] for p in args.point] if args.point else frozen['points']
    tiers = dict(tiny=0, small=1, medium=2)
    points = sorted(points, key=lambda p: (tiers[p['tier']], p['id']))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be empty; retain prior attempts in separate directories')
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='tnca-r40-frozen-residency-suite/1',
                  subset_sha256=hashlib.sha256(SUBSET.read_bytes()).hexdigest(),
                  selected=[p['id'] for p in points],
                  full_inventory=set(p['id'] for p in points) == set(by_id),
                  profile=os.environ.get('TNCA_PROFILE', 'core-vrf-sync-v5'),
                  software_tree=software_tree(),
                  sources={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                           for f in ('tools/r40/frozen_residency_suite.py',
                                     'tools/r40/frozen_residency_tests.py')},
                  steps=args.steps, host_timeout_s=args.host_timeout,
                  child_timeout_s=args.timeout, measured_performance=False, results=[])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(point):
        log = args.out / (point['id'].replace('/', '__') + '.log')
        started = time.monotonic()
        with log.open('w') as stream:
            child = subprocess.Popen(
                [sys.executable, '-u', str(ROOT / 'tools/r40/frozen_residency_tests.py'),
                 '--point', point['id'], '--steps', str(args.steps),
                 '--source-tree', report['software_tree']],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, TNCA_HOST_TIMEOUT=str(args.host_timeout)))
            timed_out = False
            try:
                code = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Include the simulator descendant, which otherwise survives
                # a killed Python worker and consumes a CPU indefinitely.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
        passed = (not timed_out and code == 0 and
                  f'r40-frozen-residency-tests PASS ({args.steps * 2}/{args.steps * 2})'
                  in log.read_text())
        return dict(point_id=point['id'], definition_sha256=point['sha256'],
                    status='pass' if passed else 'timeout' if timed_out else 'fail',
                    returncode=code, log=str(log.resolve()),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                    host_elapsed_s=time.monotonic() - started)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(run, point) for point in points]
        for future in as_completed(pending):
            result = future.result()
            report['results'].append(result)
            # Preserve a reviewable checkpoint even during long RTL jobs.
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f"{result['status'].upper()} {result['point_id']}", flush=True)
    passed = sum(r['status'] == 'pass' for r in report['results'])
    print(f'r40-frozen-residency-suite {"PASS" if passed == len(points) else "FAIL"} '
          f'({passed}/{len(points)}; full_inventory={report["full_inventory"]})', flush=True)
    return int(passed != len(points))


if __name__ == '__main__':
    raise SystemExit(main())
