#!/usr/bin/env python3
"""S1-01 -- the cycle baseline every later R38 change is compared with.

doc/r38_stage1_tasks_and_acceptance.md, S1-01 and rule R2.

This does not measure anything itself. It runs the R3.7 public-AOT gate
(tools/r37/aot_run.py: compile_aot -> .tnca -> Session -> RTL, with the
numbers checked) and keeps what that gate already knows about each case,
plus what a baseline needs and the gate does not record: the artifact's
own hash, the profile's configuration hash and the tool versions.

    --run      run the gate at --levels and write sim/build/r38/baseline_O<n>.json
    --freeze   write spec/r38/baseline_cycles.json from the O2 report
    --check    run O2 twice; every case must pass, both runs must give the same
               cycles case for case, and both must equal the frozen file

The cycles are serial-grid: the runtime launches a grid one program at a
time and adds the per-program counts. They are the right thing to compare
one change against another on this bench, and they are not a device's
throughput. Each record says so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca import profiles                                  # noqa: E402

OUT = ROOT / 'sim' / 'build' / 'r38'
FROZEN = ROOT / 'spec' / 'r38' / 'baseline_cycles.json'
AOT_RUN = ROOT / 'tools' / 'r37' / 'aot_run.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tools():
    def first_line(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (out.stdout or out.stderr).strip().splitlines()[0]
        except (OSError, IndexError, subprocess.TimeoutExpired):
            return None
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    git = first_line(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
                           capture_output=True, text=True).stdout.strip() != ''
    return dict(iverilog=first_line(['iverilog', '-V']), python=platform.python_version(),
                triton=triton_version, git_commit=git, git_dirty=dirty)


def run_level(level, source, profile, path):
    cmd = [sys.executable, str(AOT_RUN), '--source', str(source), '--levels', str(level),
           '--profile', profile, '--json', str(path)]
    subprocess.run(cmd, check=False)
    report = json.loads(path.read_text())
    config = profiles.config_hash(profile)
    for row in report['cases']:
        row['timing'] = 'serial-grid'
        row['config_hash'] = config
        art = row.get('artifact')
        row['artifact_sha256'] = sha256(ROOT / art) if art and (ROOT / art).exists() else None
    report['tools'] = tools()
    report['config_hash'] = config
    path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def cycles_of(report):
    return {r['case']: r.get('cycles') for r in report['cases']}


def problems_with(report):
    bad = []
    for r in report['cases']:
        if not r['ok']:
            bad.append(f"{r['case']}: FAILED ({r['detail']})")
        for field in ('artifact_sha256', 'source_sha256', 'config_hash'):
            if not r.get(field):
                bad.append(f"{r['case']}: no {field}")
        if r['ok'] and (r.get('cycles') is None or r.get('program_cycles') is None
                        or sum(r['program_cycles']) != r['cycles']):
            bad.append(f"{r['case']}: cycles and per-program cycles disagree")
    if not report.get('completed'):
        bad.append('the gate did not complete')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=ROOT.parent / 'Tilebench')
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.run:
        failed = []
        for level in args.levels:
            report = run_level(level, args.source, args.profile, OUT / f'baseline_O{level}.json')
            failed += problems_with(report)
        for line in failed:
            print('BASELINE', line)
        print(f'baseline: {"FAIL" if failed else "PASS"} at levels {args.levels}')
        return 1 if failed else 0

    if args.freeze:
        report = json.loads((OUT / 'baseline_O2.json').read_text())
        bad = problems_with(report)
        if bad:
            print('\n'.join(bad))
            print('refusing to freeze a baseline that does not pass')
            return 1
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(dict(
            schema='r38-baseline-cycles/1', level=2, profile=args.profile,
            config_hash=report['config_hash'], timing='serial-grid',
            source_commit=report['tools']['git_commit'],
            note='device cycles per case, fast memory model; rule R2 of doc/r38_stage1_tasks_and_acceptance.md',
            cycles=dict(sorted(cycles_of(report).items()))), indent=2) + '\n')
        print(f'froze {len(report["cases"])} cases to {FROZEN.relative_to(ROOT)}')
        return 0

    # --check
    frozen = json.loads(FROZEN.read_text())
    runs = [run_level(2, args.source, args.profile, OUT / f'baseline_check_{i}.json') for i in (1, 2)]
    bad = problems_with(runs[0]) + problems_with(runs[1])
    a, b = cycles_of(runs[0]), cycles_of(runs[1])
    if frozen['config_hash'] != runs[0]['config_hash']:
        bad.append('profile configuration differs from the frozen baseline')
    for case in sorted(set(a) | set(b) | set(frozen['cycles'])):
        if a.get(case) != b.get(case):
            bad.append(f'{case}: not deterministic ({a.get(case)} vs {b.get(case)})')
        if a.get(case) != frozen['cycles'].get(case):
            bad.append(f'{case}: {a.get(case)} cycles, frozen baseline says {frozen["cycles"].get(case)}')
    for line in bad:
        print('CHECK', line)
    print(f'baseline check: {"FAIL" if bad else "PASS"} ({len(a)} cases, two runs)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
