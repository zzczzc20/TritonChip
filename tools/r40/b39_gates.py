#!/usr/bin/env python3
"""R40-00 acceptance 2: R39's gates, as they actually are, reproducible.

    python3 tools/r40/b39_gates.py --run [--only NAME ...] [--out sim/build/r40/b39_gates_run.json]
    python3 tools/r40/b39_gates.py --freeze sim/build/r40/b39_gates_run.json
    python3 tools/r40/b39_gates.py --check sim/build/r40/b39_gates_run.json [--evidence]

doc/r40_tasks_and_acceptance.md sec 3.4: the merge gates are R39's --
`make r37`, core / core-vrf / core-slowmem / core-ooo / top / nuca,
`r38-baseline-check`, `r38-profiler`, `r38-torch`, `r38-tdesc` and the R39
role/pair regressions -- "with the list and the counts as they are when
frozen, not the counts an older document quotes". `--run` runs each on the
build host (Icarus, TORCH_PYTHON and TRITON_PYTHON as the Makefile has
them) and records its exit status, wall time and the pass/total counts its
output reports; `--freeze` writes the run to spec/r40/b39/gates.json;
`--check` compares a later run with it: every gate exits 0 and reports the
same counts. A gate added after the freeze is an amendment.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'spec' / 'r40' / 'b39' / 'gates.json'
TP = '$(TORCH_PYTHON)'

# name, command (run in ROOT with bash), what it covers
GATES = [
    ('r37', 'make r37', 'AOT + Session (R3.7 cases at O0/O1/O2)'),
    ('core', 'make core', 'core ISA tests, fast memory'),
    ('core-vrf', 'make core-vrf', 'core tests on the banked VRF'),
    ('core-slowmem', 'make core-slowmem', 'core tests, slow memory'),
    ('core-ooo', 'make core-ooo', 'core tests, out-of-order memory'),
    ('top', 'make top', 'the 2x2 mesh system'),
    ('nuca', 'make nuca', 'the NUCA slice and DCO'),
    ('core-r39hw', 'make core-r39hw', 'R39 v2 hardware switches, fast and out-of-order memory'),
    ('core-r39sw', 'make core-r39sw', 'R39 v3 (RCTX_SWITCH, RSLICE, LSU_INFLIGHT, FPU_OVERLAP)'),
    ('core-r39sw4', 'make core-r39sw4', 'R39 v4, four Role Contexts'),
    ('core-r39sw5', 'make core-r39sw5', 'R39 v5 (RPAIR, MTX_ABUF, RCTX_SWITCH=2, transposed transfer)'),
    ('r38-xpose-rtl', 'make r38-xpose-rtl', 'the transposed transfer at the ISA level'),
    ('r1-profile-tests', 'make r1-profile-tests', 'every profile reads its capabilities back'),
    ('r38-l0', 'make r38-l0', 'counters, evidence and ArchBench self-checks'),
    ('r38-baseline-check', 'make r38-baseline-check', 'TileBench at O2 against the R38 baseline'),
    ('r38-profiler', 'make r38-profiler', 'the profiler'),
    ('r38-identity', 'make r38-identity', 'artifacts byte-identical across compile-path changes'),
    ('r38-tdesc', 'make r38-tdesc', 'tensor descriptors rank 1-5 through the Triton front end'),
    ('r38-torch', 'make r38-torch', 'plugin identity O0-O2, native Triton launch, torch tests with and '
                                    'without descriptors'),
    ('archbench-selftest', 'make archbench-selftest', 'ArchBench registry, suites and scoring'),
    ('rctx-roles-v5', f'{TP} tools/r39/rctx_roles_tests.py --profile core-vrf-sync-v5',
     'R39-12 Mode A on v5'),
    ('rctx-roles-v4', f'{TP} tools/r39/rctx_roles_tests.py --profile core-vrf-sync-v4',
     'R39-12 Mode A refused on v4'),
    ('rctx-spmd-v4', f'{TP} tools/r39/rctx_spmd_tests.py --profile core-vrf-sync-v4', 'R39-05 Mode B'),
    ('rctx-spmd-v5', f'{TP} tools/r39/rctx_spmd_tests.py --profile core-vrf-sync-v5', 'R39-05 Mode B'),
    ('r39-schedule', f'{TP} tools/r39/schedule_tests.py', 'R39-01 list scheduling'),
    ('r39-ablate', f'{TP} tools/r39/ablate_tests.py', 'the R39 A/B tool'),
    ('r39-vrf-fairness', f'{TP} tools/r39/vrf_fairness_test.py', 'B5: VRF read arbitration'),
]

SUMMARY = [
    # "447/447", "84 passed", "PASS (27/27)", "10/10 ok" -- the last such count wins
    re.compile(r'(\d+)\s*/\s*(\d+)\b'),
]
PASS_LINE = re.compile(r'^\s*(PASS|ok|pass|\[PASS\])\b')
FAIL_LINE = re.compile(r'^\s*(FAIL|fail|\[FAIL\]|ERROR)\b')


def counts(text):
    """What the output says it passed: per-test PASS/FAIL lines, and the
    last n/m summary it prints."""
    lines = text.splitlines()
    out = dict(pass_lines=sum(1 for l in lines if PASS_LINE.match(l)),
               fail_lines=sum(1 for l in lines if FAIL_LINE.match(l)))
    last = None
    for l in lines[-40:]:
        for rx in SUMMARY:
            for m in rx.finditer(l):
                last = f'{m.group(1)}/{m.group(2)}'
    out['summary'] = last
    return out


def expand(cmd):
    env_py = os.environ.get('TORCH_PYTHON') or str(Path.home() / 'src' / 'torchenv' / 'bin' / 'python')
    return cmd.replace('$(TORCH_PYTHON)', env_py)


def run(only, out_path):
    rows = []
    for name, cmd, what in GATES:
        if only and name not in only:
            continue
        t = time.monotonic()
        p = subprocess.run(['bash', '-c', expand(cmd)], cwd=ROOT, capture_output=True, text=True)
        text = p.stdout + p.stderr
        log = ROOT / 'sim' / 'build' / 'r40' / 'b39_gates' / f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text)
        row = dict(name=name, command=cmd, what=what, exit=p.returncode,
                   seconds=round(time.monotonic() - t, 1), **counts(text))
        rows.append(row)
        print(f"  {'PASS' if p.returncode == 0 else 'FAIL'}  {name:<20s} exit {p.returncode}  "
              f"pass-lines {row['pass_lines']:>4d}  fail-lines {row['fail_lines']:>3d}  "
              f"summary {row['summary']}  [{row['seconds']}s]", flush=True)
    sys.path.insert(0, str(ROOT))
    from archbench.provenance import software_tree, git_commit
    doc = dict(schema='tnca-r40-b39-gates/1', software_tree=software_tree(), git=git_commit(),
               host=os.uname().nodename, gates=rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=1) + '\n')
    return 0 if all(r['exit'] == 0 for r in rows) else 1


def check(run_path, frozen_path, evidence):
    now = json.loads(Path(run_path).read_text())
    frozen = json.loads(Path(frozen_path).read_text())
    got = {r['name']: r for r in now['gates']}
    checks = []
    for f in frozen['gates']:
        g = got.get(f['name'])
        if g is None:
            checks.append(dict(name=f['name'], ok=False, why='not run'))
            continue
        same = all(g[k] == f[k] for k in ('pass_lines', 'fail_lines', 'summary'))
        ok = g['exit'] == 0 and same
        checks.append(dict(name=f['name'], ok=ok,
                           why='' if ok else f"exit {g['exit']}, counts {g['pass_lines']}/"
                                             f"{g['fail_lines']}/{g['summary']} vs frozen "
                                             f"{f['pass_lines']}/{f['fail_lines']}/{f['summary']}"))
    extra = sorted(set(got) - {f['name'] for f in frozen['gates']})
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']:<20s} {c['why']}")
    ok = all(c['ok'] for c in checks)
    print(f"b39-gates  {'PASS' if ok else 'FAIL'}  ({sum(c['ok'] for c in checks)}/{len(checks)})"
          + (f'; not frozen: {extra}' if extra else ''))
    if evidence:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import evidence as EV
        EV.record('R40-00.b39-gates', 'capability-pass' if ok else 'fail', checks,
                  detail=f"{sum(c['ok'] for c in checks)}/{len(checks)} gates as frozen",
                  run=str(run_path), software_tree=now.get('software_tree'))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--run', action='store_true')
    g.add_argument('--freeze', type=Path, metavar='RUN.json')
    g.add_argument('--check', type=Path, metavar='RUN.json')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', type=Path, default=ROOT / 'sim' / 'build' / 'r40' / 'b39_gates_run.json')
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args(argv)
    if args.run:
        return run(args.only, args.out)
    if args.freeze:
        doc = json.loads(args.freeze.read_text())
        bad = [r['name'] for r in doc['gates'] if r['exit'] != 0]
        if bad:
            sys.exit(f'not freezing a run with failing gates: {bad}')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(doc, indent=1) + '\n')
        print(f'{len(doc["gates"])} gates -> {FROZEN}')
        return 0
    return check(args.check, FROZEN, args.evidence)


if __name__ == '__main__':
    sys.exit(main())
