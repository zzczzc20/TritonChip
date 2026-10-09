#!/usr/bin/env python3
"""S1-04: measure a suite on the candidate in front of you.

    tools/r38/run.py --suite smoke

Builds the device from this working tree, runs each point's cases through
compile_aot -> .tnca -> Session(counters=True) -> RTL with the numbers
checked, `repeats` times, and writes one archbench-measurement/1 record per
case per repeat to sim/build/r38/<suite>/records.json. Exits non-zero if a
case fails, a record is invalid, a repeat differs on the fast memory model,
eta exceeds 1, or the suite overran its wall-clock budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402
import manifest                                             # noqa: E402
from archbench import bounds as B, peaks as P, r37_workloads   # noqa: E402
from tnca import profiles                                   # noqa: E402

BOUND_VERSION = 'r38-bounds-v0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(row, point, cand, profile, level, repeat, workloads, peak):
    wl = workloads.get(row['case'])
    strict = scoped = binding = eta = required = None
    if wl is not None and wl.bound_available:
        strict, binding, _ = B.lower_bound(wl, peak)
        scoped, _, _ = B.lower_bound(wl, peak, scoped=True)
        required = wl.required_bytes
    ok = row['ok']
    cycles = row.get('cycles') if ok else None
    if strict is not None and cycles:
        eta = strict / cycles
    c = row.get('counters') if ok else None
    measured = None
    if c and c.get('available'):
        t = c['total']
        measured = 8 * (t['perf_mem_rd_beats'] + t['perf_mem_wr_beats'])
    src = row.get('source')
    return {
        'schema': 'archbench-measurement/1',
        'candidate_hash': cand['candidate_hash'], 'parent_hash': None,
        'workload_id': row['case'], 'operator': row['operator'], 'point': point,
        'source_sha256': sha(src) if src else '0' * 64,
        'input_sha256': None,
        'artifact_sha256': sha(ROOT / row['artifact']) if ok else '0' * 64,
        'profile': profile, 'config_hash': profiles.config_hash(profile), 'opt_level': level,
        'evidence_level': 'measured-rtl',
        'status': 'pass' if ok else ('numeric-error' if 'AssertionError' in row.get('detail', '')
                                     else 'runtime-fault'),
        'correctness': {'detail': row.get('detail', '')},
        'time': {'roi': 'launch_accept_to_core_idle', 'timing': 'serial-grid',
                 'device_cycles': cycles, 'program_cycles': row.get('program_cycles') if ok else None,
                 'host_seconds': row.get('seconds') or None,
                 'achieved_clock_hz': None, 'device_seconds': None},
        'counters': c,
        'traffic': {'required_lower_bound_bytes': required, 'measured_bytes': measured},
        'bounds': {'version': BOUND_VERSION, 'strict_cycles': strict, 'scoped_cycles': scoped,
                   'binding': binding, 'eta': eta,
                   'unavailable': (wl.note if wl is not None and not wl.bound_available else None)},
        'resources': {'lut': None, 'ff': None, 'dsp': None, 'bram': None, 'power_w': None},
        'repeat': {'seed': 0, 'index': repeat},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', default='smoke')
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--rtl', type=Path,
                    help='a directory holding a candidate rtl/ and tb/ (default: this working tree)')
    ap.add_argument('--out', type=Path, help='output directory (default sim/build/r38/<suite>)')
    ap.add_argument('--fail-fast', action='store_true',
                    help='stop at the first failing case (gates: the verdict, not the full table)')
    ap.add_argument('--splits', nargs='+', default=['dev', 'validation', 'heldout'],
                    help='generated suites only; held-out results are recorded but not printed')
    args = ap.parse_args()
    suite = json.loads((ROOT / 'spec' / 'r38' / 'suites' / f'{args.suite}.json').read_text())
    out = args.out or ROOT / 'sim' / 'build' / 'r38' / args.suite
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cand = manifest.candidate(hardware_root=args.rtl)
    tree = dict(rtl=args.rtl / 'rtl', tb=args.rtl / 'tb') if args.rtl else {}
    bench = H.build_bench(args.profile, out / f'bench-{args.profile}.vvp', **tree)
    level = suite['level']
    repeats = args.repeats or suite['repeats']
    peak = P.spec_peak(args.profile)
    generated = 'generator' in suite
    if generated:
        import dev_cases
        points = dev_cases.build(H.TILEBENCH)
        registered = {p['id']: p for p in suite['points']}
        if set(points) != set(registered):
            print('FAIL the generator and the frozen suite disagree about which points exist; '
                  'run tools/r38/dev_cases.py --write and review the diff')
            return 1
        for pid, p in points.items():
            if p['input_sha256'] != registered[pid]['input_sha256'] or p['split'] != registered[pid]['split']:
                print(f'FAIL {pid}: inputs or split differ from the frozen suite')
                return 1
        points = {k: v for k, v in points.items() if v['split'] in args.splits}
        workloads = {k: v['workload'] for k, v in points.items()}
        point_of = {k: k for k in points}
    else:
        workloads = r37_workloads.build()
        point_of = {c: p['id'] for p in suite['points'] for c in p['cases']}
    records, problems = [], []
    for rep in range(repeats):
        kw = dict(profile=args.profile, session_kwargs=dict(bench=str(bench), counters=True), progress=generated,
                  stop_on_fail=args.fail_fast)
        rows = (H.run_points(points, level, out / 'artifacts', **kw) if generated
                else H.run_cases(level, out / 'artifacts', only=list(point_of), **kw))
        for row in rows:
            if row['case'] not in point_of:
                continue                     # a feeding stage outside the suite
            rec = record(row, point_of[row['case']], cand, args.profile, level, rep, workloads, peak)
            if generated:
                rec['input_sha256'] = points[row['case']]['input_sha256']
            problems += [f'{row["case"]}: {p}' for p in manifest.validate(rec)]
            if rec['status'] != 'pass':
                problems.append(f'{row["case"]}: {rec["status"]} ({row.get("detail")})')
            records.append(rec)
        print(f'repeat {rep}: {sum(r["ok"] for r in rows)}/{len(rows)} cases pass', flush=True)
        if args.fail_fast and problems:
            break
    by_case = {}
    for r in records:
        by_case.setdefault(r['workload_id'], []).append(r['time']['device_cycles'])
    for case, cyc in by_case.items():
        if len(set(cyc)) != 1:
            problems.append(f'{case}: repeats differ {cyc} on the fast memory model')
    missing = set(point_of) - set(by_case)
    problems += [f'{c}: no record' for c in sorted(missing)]
    wall = round(time.monotonic() - started, 1)
    if wall > suite['wall_budget_seconds']:
        problems.append(f'suite took {wall}s, budget {suite["wall_budget_seconds"]}s')
    summary = dict(suite=args.suite, candidate=cand, wall_seconds=wall, repeats=repeats,
                   problems=problems, records=len(records))
    (out / 'records.json').write_text(json.dumps(records, indent=1) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f'{"case":28} {"cycles":>8} {"L_strict":>9} {"eta":>7} {"binding":10} {"bytes meas/req":>16}')
    split_of = {p['id']: p.get('split') for p in suite['points']} if generated else {}
    for r in records:
        if r['repeat']['index'] or split_of.get(r['workload_id']) == 'heldout':
            continue
        b, t = r['bounds'], r['traffic']
        print(f'{r["workload_id"]:28} {r["time"]["device_cycles"] or "-":>8} '
              f'{b["strict_cycles"] if b["strict_cycles"] is not None else "-":>9} '
              f'{"%.4f" % b["eta"] if b["eta"] is not None else "-":>7} {b["binding"] or "-":10} '
              f'{t["measured_bytes"]}/{t["required_lower_bound_bytes"]}')
    for p in problems:
        print('FAIL', p)
    print(f'{args.suite}: {"PASS" if not problems else "FAIL"} -- {len(records)} records, {wall}s wall, '
          f'candidate {cand["candidate_hash"][:12]}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
