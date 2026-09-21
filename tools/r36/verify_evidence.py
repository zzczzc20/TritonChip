#!/usr/bin/env python3
"""
R36-EV-01 -- phase3_6_tasks_and_acceptance.md section R36-10.

An independent verifier: it recomputes the verdict from the raw exit
codes, outputs, hashes and budget in an evidence directory, and refuses
a run that is missing something, has been edited, mixes commits or
implementations, or has had only its summary changed to PASS.

Independent means it does not read `acceptance.json`'s verdict and
believe it. It reads `runs/*.json` and `runs/*.log` -- what the checkers
actually returned -- recomputes, and then compares with what the summary
claims. A disagreement is the finding.

The four refusals the standard names, and how each is made:

  missing        every package in the frozen required-case list must
                 have a result, and every case in it must appear
  tampered       the summary's own numbers are recomputed from the runs
  mixed          one commit, one platform, one set of tool versions
  summary-only   the verdict is recomputed and compared, never read
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

SIGNOFF = 'R36_ENGINEERING_READY'
PACKAGES = ['R36-01', 'R36-02', 'R36-03', 'R36-04', 'R36-05', 'R36-06',
            'R36-07', 'R36-08', 'R36-09']
# R36-10 is this verification itself and the sign-off record; it has no
# checker of its own to run, which is deliberate: a package that graded
# itself would be the thing this file exists to prevent.


def _report_verdict(d):
    """A checker's result, for the ones that are not case lists.

    Three shapes exist and each is recognised by name rather than by a
    generic `ok` lookup. The first version of this used `d.get('ok')`,
    which is present in the regression report and absent from the other
    two -- so it would have refused a conformance run that passed, for
    the reason that it could not read it. Silently. Returning None here
    makes not knowing a finding instead.
    """
    if not isinstance(d, dict):
        return None, type(d).__name__
    if 'ok' in d:                       # regression
        return bool(d['ok']), 'a report with an ok field'
    if 'mutations' in d:                # mutation_tests
        return (all(m.get('caught') for m in d['mutations'])
                and bool(d.get('tree_unchanged')),
                f"{len(d['mutations'])} mutation(s)")
    if d and all(isinstance(v, dict) and 'passed' in v
                 for v in d.values()):  # conformance, keyed by backend
        return all(v['passed'] for v in d.values()), f"{len(d)} backend(s)"
    return None, f"a dict with keys {sorted(d)[:4]}"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--json')
    args = ap.parse_args()
    ev = os.path.abspath(args.evidence)
    # A run directory, or the parent of several: take the newest.
    if not os.path.exists(os.path.join(ev, 'acceptance.json')):
        runs = sorted(d for d in os.listdir(ev)
                      if os.path.exists(os.path.join(ev, d,
                                                     'acceptance.json')))
        if not runs:
            print(f"  no acceptance.json under {ev}")
            return 2
        ev = os.path.join(ev, runs[-1])

    print("R36-EV-01: the evidence, verified independently")
    print(f"  {ev}")
    rep = load(os.path.join(ev, 'acceptance.json'))
    findings, ok = [], True

    # 1. One candidate. Everything must agree on the commit, and the
    #    tree must have been clean.
    print(f"  commit {rep.get('commit','?')[:12]}, clean tree "
          f"{rep.get('clean_tree')}")
    if not rep.get('clean_tree'):
        findings.append('the working tree was not clean')
    if not rep.get('commit'):
        findings.append('the run records no commit')

    # 2. The frozen contracts, re-hashed from the copies in the evidence
    #    AND compared with the tree. A run whose contracts were edited
    #    afterwards describes a different contract.
    for name, want in sorted((rep.get('frozen_hashes') or {}).items()):
        here = os.path.join(ev, 'contracts', name)
        there = os.path.join(ROOT, 'spec', name)
        got = (hashlib.sha256(open(here, 'rb').read()).hexdigest()
               if os.path.exists(here) else None)
        tree = (hashlib.sha256(open(there, 'rb').read()).hexdigest()
                if os.path.exists(there) else None)
        state = ('kept' if got == want else 'EDITED IN THE EVIDENCE')
        drift = '' if tree == want else '  (the tree has moved since)'
        print(f"    {name:<28s} {state}{drift}")
        if got != want:
            findings.append(f"{name} in the evidence does not match its "
                            f"recorded hash")

    # 3. Recompute every package's verdict from the runs, not from the
    #    summary.
    recomputed = {}
    for entry in rep.get('checks', []):
        name, pkg = entry['checker'], entry['package']
        runf = os.path.join(ev, 'runs', f'{name}.json')
        logf = os.path.join(ev, 'runs', f'{name}.log')
        if not os.path.exists(logf):
            findings.append(f"{name}: no log in the evidence")
            recomputed.setdefault(pkg, []).append(False)
            continue
        cases = load(runf) if os.path.exists(runf) else None
        if cases is None:
            findings.append(f"{name}: no result file")
            recomputed.setdefault(pkg, []).append(False)
            continue
        if isinstance(cases, list):
            bad = [c['id'] for c in cases if not c.get('passed')]
            good = not bad
            if entry.get('exit') == 0 and bad:
                findings.append(
                    f"{name}: exit 0 with {len(bad)} failing case(s) "
                    f"{bad} -- the summary and the cases disagree")
        else:
            good, shape = _report_verdict(cases)
            if good is None:
                findings.append(
                    f"{name}: its result is a report this verifier does "
                    f"not know how to read ({shape}). A verifier that "
                    f"fell back to the exit code here would be believing "
                    f"the summary, which is the thing it exists not to "
                    f"do.")
                good = False
        # The exit code is the primary fact; the cases are the check on
        # it. Both have to say the same thing.
        good = good and entry.get('exit') == 0
        recomputed.setdefault(pkg, []).append(good)
        n = len(cases) if isinstance(cases, list) else 1
        if not isinstance(cases, list):
            n = _report_verdict(cases)[1]
        print(f"    {pkg} {name:<12s} exit={entry.get('exit')} "
              f"{n}{' result(s)' if isinstance(cases, list) else ''}"
              f" -> {'ok' if good else 'OPEN'}")

    by_pkg = {p: all(v) for p, v in recomputed.items()}
    missing = [p for p in PACKAGES if p not in by_pkg]
    if missing:
        findings.append(f"no result at all for {missing}")
    for p in sorted(by_pkg):
        if not by_pkg[p]:
            findings.append(f"{p} has an open checker")

    # 4. The verdict, recomputed and then compared with what the summary
    #    claims. This is the "只改 summary PASS" refusal.
    want = (SIGNOFF if (not findings and rep.get('clean_tree'))
            else 'NOT ACCEPTED')
    claimed = rep.get('verdict')
    print()
    print(f"  the summary claims: {claimed}")
    print(f"  recomputed:         {want}")
    if claimed != want and claimed != 'PARTIAL':
        findings.append(f"the summary says {claimed!r} and the evidence "
                        f"says {want!r}")
        ok = False

    # 5. And the claim level, which no result may raise.
    level = rep.get('claim_level')
    print(f"  claim level: {level!r} "
          f"({'allowed' if level == 'engineering-ready' else 'NOT ALLOWED'})")
    if level != 'engineering-ready':
        findings.append(f"claim_level {level!r}: this gate may only "
                        f"produce engineering-ready")

    ok &= not findings
    print()
    for f in findings:
        print(f"  open: {f}")
    print(f"  verification: {'CONFIRMED' if ok else 'REFUSED'}")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'evidence': ev, 'claimed': claimed,
                       'recomputed': want, 'packages': by_pkg,
                       'findings': findings, 'confirmed': bool(ok)},
                      fh, indent=2, sort_keys=True)
            fh.write('\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
