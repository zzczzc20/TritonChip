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
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

SIGNOFF = 'R36_ENGINEERING_READY'

REQUIRED = json.load(open(os.path.join(ROOT, 'spec',
                                       'r36_required_cases.json')))
# R36-10 is this verification itself and the sign-off record; it has no
# checker of its own to run, which is deliberate: a package that graded
# itself would be the thing this file exists to prevent. Its three cases
# are answered here, at the bottom.
SELF_GRADED = 'R36-10'
PACKAGES = [p['id'] for p in REQUIRED['packages']
            if p['id'] != SELF_GRADED]


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
        # The OUTPUT, which R36-EV-01 names beside the exit codes and
        # the hashes and which this used to check only for existence.
        # Every checker prints a verdict line; it has to agree with the
        # exit code and with the case records, because three ways of
        # saying the same thing is how a disagreement becomes visible.
        text = open(logf, encoding='utf-8', errors='replace').read()
        said_fail = bool(re.search(r'^\s{2}\S.*\bFAIL\b', text, re.M))
        said_pass = bool(re.search(r'^\s{2}\S.*\bPASS\b', text, re.M))
        if entry.get('exit') == 0 and said_fail and not said_pass:
            findings.append(
                f"{name}: exit 0 and its own output says FAIL")
        if entry.get('exit') not in (0, None) and said_pass \
                and not said_fail:
            findings.append(
                f"{name}: a non-zero exit and its own output says PASS")
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

    # 3b. R36-EV-02: every one of the thirty standards has evidence.
    #
    # Package-level agreement is not enough. A checker that ran, exited
    # zero and quietly stopped emitting one of its cases would leave a
    # package looking complete, so the case IDs the contract names are
    # matched against the case IDs that actually appear.
    seen_cases = set()
    for entry in rep.get('checks', []):
        runf = os.path.join(ev, 'runs', f"{entry['checker']}.json")
        if not os.path.exists(runf):
            continue
        cases = load(runf)
        if isinstance(cases, list):
            seen_cases |= {c.get('id') for c in cases if isinstance(c, dict)}
    print()
    for pk in REQUIRED['packages']:
        want = set(pk['cases'])
        if pk['id'] == SELF_GRADED:
            print(f"    {pk['id']} {sorted(want)} answered by this "
                  f"verification")
            continue
        gap = sorted(want - seen_cases)
        print(f"    {pk['id']} {len(want - set(gap))}/{len(want)} case(s) "
              f"present{'' if not gap else '   MISSING ' + str(gap)}")
        if gap:
            findings.append(f"{pk['id']}: no evidence for {gap}")

    # 3c. Every standard has an owner as well as evidence (R36-EV-02).
    for pk in REQUIRED['packages']:
        if not (pk.get('owner') or '').strip():
            findings.append(f"{pk['id']}: no owner recorded")

    # 3d. The budget, recomputed (R36-EV-01 names it alongside the exit
    #     codes, the counts and the hashes). Summed from what each
    #     checker actually took, not from a number the run wrote down.
    budget = json.load(open(os.path.join(
        ROOT, 'spec', 'r36_regression.json')))['budget']
    cap = budget['new_runs']['elapsed_seconds']
    # The cap is on what this stage ADDED. `regression` re-runs suites
    # that existed before it, and charging their runtime here would fail
    # the gate for the size of a suite this stage did not create.
    new_spent = sum(float(e.get('elapsed_s') or 0)
                    for e in rep.get('checks', [])
                    if e.get('checker') != 'regression')
    inherited = sum(float(e.get('elapsed_s') or 0)
                    for e in rep.get('checks', [])
                    if e.get('checker') == 'regression')
    print()
    print(f"  new runs: {new_spent:.0f}s of {cap}s across "
          f"{len([e for e in rep.get('checks', []) if e.get('checker') != 'regression'])}"
          f" checker(s) ({'within' if new_spent <= cap else 'OVER'})")
    print(f"  inherited: {inherited:.0f}s, recorded and not capped")
    if new_spent > cap:
        findings.append(
            f"over the frozen new-run budget: {new_spent:.0f}s of "
            f"{cap}s. The contract keeps this a failure rather than "
            f"something to re-run until it fits.")

    # 3e. One candidate, one implementation. A run whose pieces came
    #     from different trees is what "混提交/实现" names.
    build = os.path.join(ev, 'packages', 'build-report.json')
    if os.path.exists(build):
        br = load(build)
        bc = br.get('commit')
        print(f"  the build report's commit: {(bc or '?')[:12]} "
              f"({'the same candidate' if bc == rep.get('commit') else 'A DIFFERENT ONE'})")
        if bc and bc != rep.get('commit'):
            findings.append(
                f"the deliverables were built at {bc[:12]} and the "
                f"checkers ran at {rep.get('commit', '?')[:12]}")
        lvl = (br.get('hardware') or {}).get('claim_level')
        if lvl and lvl != 'engineering-ready':
            findings.append(f"the export claims {lvl!r}")
    else:
        findings.append('no build report: the deliverables are not '
                        'bound to this run')

    # 3f. A failure and a re-run keep separate run IDs (R36-EV-02).
    root = os.path.dirname(ev)
    runs = [d for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
    print(f"  run directories under {os.path.basename(root)}: "
          f"{len(runs)}, all distinct: {len(runs) == len(set(runs))}")
    if len(runs) != len(set(runs)):
        findings.append('two runs share an id')

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

    # R36-10's own three cases, answered here rather than by a checker
    # that would be grading itself.
    ev01 = not [f for f in findings
                if 'does not match' in f or 'disagree' in f
                or 'over the frozen budget' in f]
    ev02 = not [f for f in findings
                if 'no evidence for' in f or 'no result' in f
                or 'no owner' in f or 'different' in f
                or 'share an id' in f or 'not bound' in f] \
        and bool(rep.get('commit')) and bool(rep.get('clean_tree'))
    rel01 = ok and claimed == SIGNOFF
    tenth = [
        {'id': 'R36-EV-01', 'passed': bool(ev01),
         'detail': 'the verdict recomputed from the raw runs, and the '
                   'frozen contracts re-hashed'},
        {'id': 'R36-EV-02', 'passed': bool(ev02),
         'detail': 'every package bound to one commit, on a clean tree, '
                   'with evidence for all thirty cases'},
        {'id': 'R36-REL-01', 'passed': bool(rel01),
         'detail': f"ten packages, and the only sign-off this may be is "
                   f"{SIGNOFF}; FPGA and ASIC physical verification are "
                   f"an external follow-on"},
    ]
    for c in tenth:
        print(f"    {c['id']:<14s} {'ok' if c['passed'] else 'OPEN'}")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'evidence': ev, 'claimed': claimed,
                       'recomputed': want, 'packages': by_pkg,
                       'cases_seen': sorted(seen_cases),
                       'r36_10': tenth,
                       'findings': findings, 'confirmed': bool(ok)},
                      fh, indent=2, sort_keys=True)
            fh.write('\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
