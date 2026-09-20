#!/usr/bin/env python3
"""
The R1 acceptance gate -- doc/phase1_tasks_and_acceptance.md R1-10.

Runs every entry point the contract requires, maps what actually executed onto
the frozen case inventory, and returns non-zero unless the whole contract is
satisfied. It is expected to FAIL until R1 is finished: a gate that passes on
partial delivery is the thing the contract exists to prevent.

What it does NOT do is guess. A case in spec/r1_required_cases.json that no
entry point executed is reported as NOT RUN and fails the verdict; it is never
marked "not applicable", and its absence never reduces the required set.

strict=1, retry=0, and none of the timeout caps can be raised from the
environment. See tools/r1/r1lib.py.
"""
import argparse
import json
import hashlib
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inventory as INV   # noqa: E402
import r1lib as R         # noqa: E402

ROOT = R.ROOT

# One line of run_test.sh output: a name, then a verdict. The gap between them
# is whatever is left of a fixed-width column, so a long name leaves exactly
# one space -- which is why this is `\s+` and not `\s{2,}`. Requiring two
# spaces silently dropped every case whose name ran long, and a case the gate
# never saw is indistinguishable from one that never ran.
RESULT_RE = re.compile(r'^\s{2}(\S.*?)\s+(PASS|FAIL|SKIP)\b')

# `t17_descfault (slow mem)` and `t17_descfault` are the same case run in two
# profiles; the profile is already in the ID template, so it is stripped here.
QUALIFIER_RE = re.compile(r'\s*\([^)]*\)\s*$')


def case_key(name):
    name = QUALIFIER_RE.sub('', name.strip())
    return re.sub(r'[^0-9a-zA-Z_]+', '-', name).strip('-')

# Entry points, in the order the contract's batches imply. `kind` picks the
# timeout cap; `maps` turns this entry's per-case lines into inventory IDs.
ENTRIES = [
    ('lint',         ['make', 'lint'],            'build', None),
    ('unit',         ['make', 'unit'],            'build', 'BASE-UNIT-{}'),
    ('core-fast',    ['make', 'core'],            'build', 'BASE-CORE-{}-fast'),
    ('core-slow',    ['make', 'core-slowmem'],    'build', 'BASE-CORE-{}-slow'),
    ('core-ooo',     ['make', 'core-ooo'],        'build', 'BASE-CORE-{}-ooo'),
    ('nuca',         ['make', 'nuca'],            'build', None),
    ('mesh',         ['make', 'mesh'],            'build', None),
    ('top',          ['make', 'top'],             'build', None),
    # `make compiler` runs several checks and several suites. Mapping its
    # exit code onto BASE-CC-UNIT reported that suite as failing when a
    # *different* step in the same target failed, which is a mapping bug
    # wearing the shape of a test failure. Its per-case lines are parsed.
    ('compiler',     ['make', 'compiler'],        'build', '{}'),
    ('compiler-rtl', ['make', 'compiler-rtl'],    'build', None),
    # Not a make target: it is one line inside `make compiler`. Calling it
    # directly is what lets the gate see its exit code on its own.
    ('compiler-ir',  [sys.executable, 'compiler/test/IR/run_ir_tests.py'],
                                                  'build', '{}'),
    ('triton-e2e',   ['make', 'triton-e2e'],      'build', None),
    ('e2e',          ['make', 'e2e'],             'build', None),
]

# Suites that report per-case results as JSON rather than as output lines.
# Their case IDs are the contract's own, so no mapping is needed -- which is
# the point of having them emit structured results in the first place.
CASE_SUITES = [
    ('gate-selftest', [sys.executable, 'tools/r1/selftest.py', '--json'],
     'build'),
    ('build-tests',   [sys.executable, 'tools/r1/build_tests.py', '--json'],
     'build'),
    ('profile-tests', [sys.executable, 'tools/r1/profile_tests.py', '--json'],
     'build'),
]

# Entries whose verdict covers a fixed set of inventory IDs rather than
# emitting one line per case.
# Entry output line name -> inventory ID, for entries that emit named lines.
LINE_IDS = {
    'compiler-unit': 'BASE-CC-UNIT',
    'compiler-ir':   'BASE-CC-IR',
}

COVERS = {
    'nuca':         ['BASE-NUCA-fast', 'BASE-NUCA-slow', 'BASE-NUCA-ooo'],
    'mesh':         ['BASE-MESH-2x2'],
    'top':          ['BASE-TOP-t10_mesh-single-role', 'BASE-TOP-t10_mesh-all-roles',
                     'BASE-TOP-t24_amo_mesh-single-role',
                     'BASE-TOP-t24_amo_mesh-all-roles'],
    'compiler-rtl': ['BASE-CC-RTL'],
    'triton-e2e':   ['BASE-TRITON-CORE'],
    'e2e':          ['BASE-TRITON-SYSTEM'],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_results(text):
    """Per-case verdicts from the standard two-column output."""
    out = {}
    for line in text.splitlines():
        m = RESULT_RE.match(line)
        if m:
            name, verdict = m.group(1).strip(), m.group(2)
            out[name] = verdict
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', help='run directory (default: a fresh one)')
    ap.add_argument('--only', nargs='*', help='limit to these entries (dev use)')
    args = ap.parse_args()

    run_dir = args.evidence or R.fresh_run_dir(tag='acceptance')
    for sub in ('logs', 'cases', 'profiles', 'build', 'mutations'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    started = time.time()
    spec = INV.load_spec()
    required = INV.required_ids(spec)

    run_meta = dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    git=R.git_state(), strict=True, retry=0,
                    caps=R.CAPS, acceptance_cap=R.ACCEPTANCE_CAP,
                    contract='doc/phase1_tasks_and_acceptance.md')

    # R1-01: the frozen inventory has to be well formed before anything runs.
    spec_problems = INV.spec_problems(spec)
    pipe_problems = INV.unguarded_pipelines()

    statuses = {}
    discovered = []
    steps = []

    # An artefact from another profile is newer than every prerequisite, so
    # make accepts it; this is the check that does not (BUILD-07).
    art = R.run_step(['make', 'r1-check-artifacts'], kind='build',
                     case_dir=os.path.join(run_dir, 'build'),
                     name='check-artifacts')
    steps.append(art)

    entries = [e for e in ENTRIES if not args.only or e[0] in args.only]
    for name, argv, kind, tmpl in entries:
        step = R.run_step(argv, kind=kind,
                          case_dir=os.path.join(run_dir, 'logs', name),
                          name=name,
                          env={'R1_STRICT': '1'})
        steps.append(step)
        text = R.read_text(step['stdout']) + R.read_text(step['stderr'])
        per_case = parse_results(text)

        if tmpl:
            for case_name, verdict in per_case.items():
                key = case_key(case_name)
                cid = LINE_IDS.get(key) or tmpl.format(key)
                if cid not in required:
                    continue     # a line this entry prints that is not a case
                discovered.append(cid)
                statuses[cid] = verdict
        for cid in COVERS.get(name, []):
            discovered.append(cid)
            statuses[cid] = 'PASS' if step.ok else 'FAIL'
        if not tmpl and name not in COVERS and not step.ok:
            statuses[f'ENTRY-{name}'] = 'FAIL'
            discovered.append(f'ENTRY-{name}')

    # Suites that report their own case IDs.
    for name, argv, kind in CASE_SUITES:
        if args.only and name not in args.only:
            continue
        case_dir = os.path.join(run_dir, 'logs', name)
        os.makedirs(case_dir, exist_ok=True)
        out_json = os.path.join(case_dir, 'cases.json')
        step = R.run_step(argv + [out_json], kind=kind, case_dir=case_dir,
                          name=name, env={'R1_STRICT': '1'})
        steps.append(step)
        if os.path.exists(out_json):
            with open(out_json) as fh:
                for rec in json.load(fh):
                    cid = rec['id']
                    if cid not in required:
                        continue
                    discovered.append(cid)
                    statuses[cid] = 'PASS' if rec.get('passed') else 'FAIL'

    # Everything the contract requires that no entry point ran.
    for cid in sorted(required):
        if cid not in statuses:
            statuses[cid] = 'NOT RUN'
            discovered.append(cid)

    rec = INV.reconcile(required, discovered, statuses)

    failing_steps = [s['name'] for s in steps if not s.ok]
    problems = list(rec['problems'])
    problems += [f"spec: {p}" for p in spec_problems]
    problems += [f"unguarded pipeline: {p}" for p in pipe_problems]
    problems += [f"entry failed: {n}" for n in failing_steps]

    verdict = 'ACCEPTED' if not problems else 'NOT ACCEPTED'

    R.write_json(os.path.join(run_dir, 'run.json'), run_meta)
    R.write_json(os.path.join(run_dir, 'inventory.json'), rec)
    R.write_json(os.path.join(run_dir, 'results.json'),
                 dict(statuses=statuses, steps=steps))
    R.write_json(os.path.join(run_dir, 'acceptance.json'),
                 dict(verdict=verdict, problems=problems,
                      required=len(required),
                      passed=sum(1 for v in statuses.values() if v == 'PASS'),
                      not_run=sum(1 for v in statuses.values() if v == 'NOT RUN'),
                      failed=sum(1 for v in statuses.values()
                                 if v in ('FAIL', 'ERROR', 'TIMEOUT')),
                      seconds=round(time.time() - started, 1)))

    # checksums last, over everything else, so the verifier can re-derive them.
    lines = []
    for base, _dirs, files in os.walk(run_dir):
        for f in sorted(files):
            p = os.path.join(base, f)
            if os.path.basename(p) == 'checksums.sha256':
                continue
            lines.append(f"{sha256(p)}  {os.path.relpath(p, run_dir)}")
    with open(os.path.join(run_dir, 'checksums.sha256'), 'w') as fh:
        fh.write('\n'.join(sorted(lines)) + '\n')

    npass = sum(1 for v in statuses.values() if v == 'PASS')
    nrun = sum(1 for v in statuses.values() if v == 'NOT RUN')
    nbad = sum(1 for v in statuses.values() if v in ('FAIL', 'ERROR', 'TIMEOUT'))
    print()
    print(f"  required={len(required)}  pass={npass}  not-run={nrun}  fail={nbad}")
    print(f"  evidence: {os.path.relpath(run_dir, ROOT)}")
    print(f"  verdict:  {verdict}")
    if problems:
        print("  first problems:")
        for p in problems[:12]:
            print('    ' + p)
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more")
    return 0 if verdict == 'ACCEPTED' else 1


if __name__ == '__main__':
    sys.exit(main())
