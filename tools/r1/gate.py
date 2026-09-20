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
import shutil
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
    # Per-seed lines, named with their inventory IDs directly.
    ('nuca',         ['make', 'nuca'],            'build', '{}'),
    ('mesh',         ['make', 'mesh'],            'build', '{}'),
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
    # One ID per test, printed by compiler/test/run_suite.py. These used to
    # map onto a single ID each, which is how five cases and two cases could
    # be "covered" by two rows in the inventory.
    ('triton-e2e',   ['make', 'triton-e2e'],      'build', '{}'),
    ('e2e',          ['make', 'e2e'],             'build', '{}'),
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
    ('resource-tests', [sys.executable, 'tools/r1/resource_tests.py', '--json'],
     'build'),
    ('ir-tests',      [sys.executable, 'tools/r1/ir_tests.py', '--json'],
     'build'),
    # The regression matrix re-runs several of the entries above on purpose:
    # its cases are about the combinations and the seeds, not about whether a
    # suite passes once.
    ('regression',    [sys.executable, 'tools/r1/regression.py', '--json'],
     'build'),
    # It makes its own small evidence directory rather than tampering with
    # this run's, which is still being written.
    ('evidence-tests', [sys.executable, 'tools/r1/evidence_tests.py', '--json'],
     'build'),
    # Last, and deliberately so: it copies the tree once per mutation and
    # re-runs a suite twice, so it is the most expensive thing here and the
    # only one whose value depends on everything above it already working.
    ('mutation-tests', [sys.executable, 'tools/r1/mutation_tests.py', '--json'],
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
    'top':          ['BASE-TOP-t10_mesh-single-role', 'BASE-TOP-t10_mesh-all-roles',
                     'BASE-TOP-t24_amo_mesh-single-role',
                     'BASE-TOP-t24_amo_mesh-all-roles'],
    'compiler-rtl': ['BASE-CC-RTL'],
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


def work_packages(spec, statuses):
    """Roll every case up to the work package that owns it.

    The contract's final verdict ends with "R1-01 ... R1-12 全部 PASS", and
    until now nothing computed that: the gate counted cases and the sign-off
    table asserted packages, with no arithmetic between them. Three packages
    own no cases of their own -- R1-01 is the inventory itself, R1-09 is the
    environment, R1-12 is this table -- and those are decided by their own
    checks below rather than by an empty average, which would otherwise read
    as a pass.
    """
    out = {}
    by_wp = {}
    for c in spec['cases']:
        by_wp.setdefault(c['work_package'], []).append(c['id'])
    for wp in [f"R1-{i:02d}" for i in range(1, 13)]:
        ids = by_wp.get(wp, [])
        bad = sorted(i for i in ids
                     if statuses.get(i, 'NOT RUN') in INV.NOT_A_PASS)
        out[wp] = dict(cases=len(ids), failing=bad,
                       verdict=('PASS' if ids and not bad else
                                'NO CASES' if not ids else 'NOT PASS'))
    return out


def junit(path, statuses, steps, seconds):
    """results.json in the shape CI systems read.

    The contract asks for `results.json + junit.xml`, not one or the other:
    the JSON is what `r1-verify-evidence` re-derives its set relations from,
    and the XML is what a CI run turns into a red or green square. Writing
    only the second is how a dashboard comes to disagree with the archive.
    """
    def esc(t):
        return (str(t).replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))

    by_step = {s['name']: s for s in steps}
    bad = [k for k, v in statuses.items() if v in INV.NOT_A_PASS]
    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<testsuites name="r1-acceptance" tests="{len(statuses)}" '
           f'failures="{len(bad)}" time="{seconds}">',
           f'  <testsuite name="r1" tests="{len(statuses)}" '
           f'failures="{len(bad)}">']
    for cid in sorted(statuses):
        verdict = statuses[cid]
        out.append(f'    <testcase classname="r1" name="{esc(cid)}">')
        if verdict in INV.NOT_A_PASS:
            out.append(f'      <failure message="{esc(verdict)}">'
                       f'{esc(cid)} is {esc(verdict)}; the contract admits no '
                       f'status but PASS</failure>')
        out.append('    </testcase>')
    out.append('  </testsuite>')
    out.append('  <testsuite name="entries" '
               f'tests="{len(steps)}" '
               f'failures="{sum(1 for s in steps if not s.ok)}">')
    for st in steps:
        out.append(f'    <testcase classname="entry" name="{esc(st["name"])}" '
                   f'time="{st.get("seconds", 0)}">')
        if not st.ok:
            out.append(f'      <failure message="exit {esc(st.get("returncode"))}">'
                       f'{esc(st.get("stdout", ""))}</failure>')
        out.append('    </testcase>')
    out.append('  </testsuite>')
    out.append('</testsuites>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(out) + '\n')


def archive_profiles(run_dir):
    """The configuration inputs, the generated targets and the readback.

    The contract's `profiles/` is not a copy of the spec for decoration: it
    is what makes "this run was on THAT machine" checkable afterwards, and
    what a later run is compared against when someone asks whether the
    configuration moved.
    """
    dest = os.path.join(run_dir, 'profiles')
    os.makedirs(dest, exist_ok=True)
    written = []
    src = os.path.join(ROOT, 'spec', 'r1_profiles.json')
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dest, 'r1_profiles.json'))
        written.append('r1_profiles.json')
    tdir = os.path.join(ROOT, 'compiler', 'targets')
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if f.endswith('.json'):
                shutil.copy2(os.path.join(tdir, f), os.path.join(dest, f))
                written.append(f)
    try:
        sys.path.insert(0, HERE)
        import profiles as P
        man = {name: P.make_manifest(name) for name in P.load()['profiles']}
        R.write_json(os.path.join(dest, 'manifests.json'), man)
        written.append('manifests.json')
    except Exception as e:
        R.write_json(os.path.join(dest, 'manifests.json'),
                     dict(error=f"{type(e).__name__}: {e}"))
    return written


def archive_build(run_dir):
    """Fingerprints and provenance for every artefact the run used."""
    dest = os.path.join(run_dir, 'build')
    os.makedirs(dest, exist_ok=True)
    build = os.path.join(ROOT, 'sim', 'build')
    records = {}
    if os.path.isdir(build):
        for f in sorted(os.listdir(build)):
            full = os.path.join(build, f)
            if f.endswith(('.fp', '.provenance.json')):
                shutil.copy2(full, os.path.join(dest, f))
            if f.endswith('.vvp'):
                records[f] = dict(sha256=sha256(full),
                                  bytes=os.path.getsize(full))
    R.write_json(os.path.join(dest, 'artifacts.json'), records)
    return records


def archive_cases(run_dir):
    """The per-case JSON each structured suite produced.

    `logs/` holds what was printed; this holds what was decided, so a
    disagreement between the two is visible rather than averaged.
    """
    dest = os.path.join(run_dir, 'cases')
    os.makedirs(dest, exist_ok=True)
    n = 0
    for base, _dirs, files in os.walk(os.path.join(run_dir, 'logs')):
        for f in files:
            if f == 'cases.json':
                suite = os.path.basename(base)
                shutil.copy2(os.path.join(base, f),
                             os.path.join(dest, f"{suite}.json"))
                n += 1
    return n


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
    wps = work_packages(spec, statuses)

    # The three packages that own no cases, decided by their own evidence.
    #
    # R1-01 is the inventory and the support matrix; the spec check above
    # already ran, and a stale matrix is a stale claim about the machine.
    sm = R.run_step([sys.executable, 'tools/isa/gen_support_matrix.py',
                     '--check'], kind='meta',
                    case_dir=os.path.join(run_dir, 'build'),
                    name='support-matrix')
    steps.append(sm)
    wps['R1-01']['verdict'] = ('PASS' if sm.ok and not spec_problems
                               else 'NOT PASS')
    wps['R1-01']['evidence'] = 'spec/r1_required_cases.json, doc/support_matrix.md'

    # R1-09 is the environment. One environment can say whether IT is an
    # acceptance environment; it cannot say whether two agree, so this is
    # PASS only where the cross-environment comparison has been made, which
    # is .github/workflows/r1-acceptance.yml's `compare` job.
    doc = R.run_step([sys.executable, 'tools/r1/doctor.py'], kind='meta',
                     case_dir=os.path.join(run_dir, 'build'), name='doctor')
    steps.append(doc)
    import platform as _plat
    is_linux_x86 = (_plat.system() == 'Linux'
                    and _plat.machine() in ('x86_64', 'amd64'))
    # Deliberately not PASS, and deliberately not a defect. R1-09 asks
    # whether two environments agree, and one run is not entitled to answer
    # that; tools/r1/compare_envs.py resolves it over both, and the R1
    # verdict is the combined one.
    wps['R1-09']['verdict'] = 'NOT PASS'
    wps['R1-09']['deferred_to'] = 'tools/r1/compare_envs.py'
    wps['R1-09']['evidence'] = (
        'this environment: ' + ('an R1 acceptance platform' if is_linux_x86
                                else f"{_plat.system()} {_plat.machine()}, "
                                     "which R1-09 does not accept")
        + '; the A/B comparison is tools/r1/compare_envs.py and is not part '
          'of a single run')

    # R1-12 is the sign-off table: complete, and not claiming a PASS this run
    # contradicts.
    sg = R.run_step([sys.executable, 'tools/r1/signoff.py',
                     '--evidence', run_dir], kind='meta',
                    case_dir=os.path.join(run_dir, 'build'), name='signoff')
    steps.append(sg)
    all_twelve = all(v['verdict'] == 'PASS' for k, v in wps.items()
                     if k != 'R1-12')
    wps['R1-12']['verdict'] = 'PASS' if (sg.ok and all_twelve) else 'NOT PASS'
    wps['R1-12']['deferred_to'] = 'tools/r1/compare_envs.py'
    wps['R1-12']['evidence'] = 'doc/r1_signoff.md, doc/r1_status.md'

    failing_steps = [s['name'] for s in steps if not s.ok]
    problems = list(rec['problems'])
    problems += [f"work package {k}: {v['verdict']}"
                 + (f" ({len(v['failing'])} case(s) not passing)"
                    if v['failing'] else '')
                 for k, v in sorted(wps.items()) if v['verdict'] != 'PASS']
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
                      work_packages=wps,
                      required=len(required),
                      passed=sum(1 for v in statuses.values() if v == 'PASS'),
                      not_run=sum(1 for v in statuses.values() if v == 'NOT RUN'),
                      failed=sum(1 for v in statuses.values()
                                 if v in ('FAIL', 'ERROR', 'TIMEOUT')),
                      seconds=round(time.time() - started, 1)))

    junit(os.path.join(run_dir, 'junit.xml'), statuses, steps,
          round(time.time() - started, 1))
    archive_profiles(run_dir)
    archive_build(run_dir)
    archive_cases(run_dir)
    # mutations/ is the mutation suite's own report: the contract wants the
    # failure injections' ACTUAL results archived, not a claim that they ran.
    mut = os.path.join(run_dir, 'logs', 'mutation-tests', 'cases.json')
    if os.path.exists(mut):
        shutil.copy2(mut, os.path.join(run_dir, 'mutations', 'results.json'))
    for name in ('gate-selftest',):
        sel = os.path.join(run_dir, 'logs', name, 'cases.json')
        if os.path.exists(sel):
            shutil.copy2(sel, os.path.join(run_dir, 'mutations',
                                           f'{name}.json'))

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
    deferred = [k for k, v in wps.items()
                if v.get('deferred_to') and v['verdict'] != 'PASS']
    if deferred and verdict != 'ACCEPTED':
        print(f"  {', '.join(sorted(deferred))} cannot be settled by one run: "
              f"they are about the two environments agreeing.")
        print("  Run tools/r1/compare_envs.py over both runs' evidence for "
              "the R1 verdict.")
    if problems:
        print("  first problems:")
        for p in problems[:12]:
            print('    ' + p)
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more")
    return 0 if verdict == 'ACCEPTED' else 1


if __name__ == '__main__':
    sys.exit(main())
