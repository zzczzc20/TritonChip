#!/usr/bin/env python3
"""
R36-TEST-02 -- phase3_6_tasks_and_acceptance.md section R36-08.

The frozen impact-scope regression, run here rather than cited.

Section 5.2 is explicit: R3/R3.5's sign-off proves things about their
frozen commits and implementations, and quoting an old ACCEPTED verdict
does not show that a refactor did not regress. So spec/r36_regression.json
names the suites the interface work could reach, frozen at the start of
the stage, and this runs every one of them on the candidate.

Three things beyond pass or fail:

  A SKIP is a failure unless it is one of the two inherited open items
  the scope file names, each with a reason and with `in_scope: false`.
  R1-02 exists to stop a missing case becoming a skip, and this is the
  same rule one level up.

  Section 5.2's machine-code and cycle fidelity is part of the verdict,
  not a separate claim: tools/r36/fidelity.py's result is required here.

  The budget is accounted subprocess by subprocess and compared with the
  frozen numbers. Over budget stays a failure; the contract forbids
  re-running until it fits.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

SCOPE = json.load(open(os.path.join(ROOT, 'spec',
                                    'r36_regression.json')))


def run(cmd, log, timeout=7200):
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault('TRITON_PYTHON',
                   os.environ.get('TRITON_PYTHON', sys.executable))
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout, env=env)
    return r, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--skip-fidelity', action='store_true',
                    help='for a quick pass; the gate never passes this')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R36-TEST-02: the frozen impact scope, run on this candidate")
    print(f"  frozen at {SCOPE['frozen_at']}, "
          f"{len(SCOPE['suites'])} suite(s) in scope")
    rows, ok, wall = [], True, 0.0
    for s in SCOPE['suites']:
        if args.only and s['id'] not in args.only:
            continue
        r, secs = run(s['command'], None)
        wall += secs
        out = (r.stdout + r.stderr)
        passed = r.returncode == 0
        skipped = 'SKIP' in out
        rows.append({'id': s['id'], 'phase': s['phase'],
                     'command': s['command'], 'passed': passed,
                     'skipped': skipped, 'seconds': secs})
        verdict = 'PASS' if passed else 'FAIL'
        if passed and skipped:
            verdict = 'PASS+SKIP'
        print(f"  {s['id']:<18s} {verdict:<10s} {secs:>6.1f}s  "
              f"{s['command']}", flush=True)
        if not passed:
            for ln in [x for x in out.splitlines() if x.strip()][-4:]:
                print(f"      {ln.strip()[:96]}", flush=True)
        if skipped:
            for ln in [x for x in out.splitlines() if 'SKIP' in x]:
                print(f"      SKIPPED: {ln.strip()[:88]}", flush=True)
            print("      a skip in a suite that is in scope is a failure: "
                  "R1-02 keeps a missing case from becoming one")
        ok &= passed and not skipped

    print()
    print(f"  the two inherited open items, deliberately not in scope:")
    for o in SCOPE['known_open']:
        print(f"    {o['id']:<12s} {o['state']}")
        print(f"      {o['why'][:88]}")
        print(f"      not in scope because: {o['why_not_in_scope'][:76]}")

    fid = None
    if not args.skip_fidelity:
        print()
        print("  section 5.2, machine code and cycles against the frozen "
              "baseline:")
        r, secs = run(f"{os.environ.get('TRITON_PYTHON', sys.executable)} "
                      f"tools/r36/fidelity.py --modes B0 O2 "
                      f"--json {os.path.join(ROOT, 'sim', 'build', 'r36_fidelity.json')}",
                      None)
        wall += secs
        for ln in [x for x in r.stdout.splitlines() if x.strip()][-6:]:
            print(f"    {ln.strip()[:96]}")
        fid = r.returncode == 0
        ok &= fid
    else:
        print("  section 5.2 fidelity: NOT RUN (--skip-fidelity). This "
              "cannot be a passing gate result.")
        ok = False

    b = SCOPE['budget']
    print()
    print(f"  budget: {wall:.0f}s of {b['elapsed_seconds']}s "
          f"({'within' if wall <= b['elapsed_seconds'] else 'OVER'})")
    within = wall <= b['elapsed_seconds']
    if not within:
        print("    over budget stays a failure: the contract forbids "
              "re-running until it fits")
    ok &= within

    if args.json:
        R.write_json(args.json, {'suites': rows, 'fidelity': fid,
                                 'wall_seconds': round(wall, 1),
                                 'budget': b, 'ok': ok})
    bad = [r['id'] for r in rows if not r['passed'] or r['skipped']]
    print()
    print(f"  r36-regression         {'PASS' if ok else 'FAIL':<6s} "
          + (', '.join(bad) if bad
             else f"{len(rows)} suite(s), no skip, fidelity {fid}"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
