#!/usr/bin/env python3
"""
The R1 input R2 stands on -- doc/phase2_tasks_and_acceptance.md R2-01.

    "缺失/伪造/不完整 R1 验收证据，R2 gate 非零；R1 的实时 PARTIAL 状态不能被
     当作前置通过"

R2 is accepted on top of a *passed* R1, and the way that claim usually rots is
not forgery -- it is a stale directory. Someone points R2 at last week's
evidence, R1 has moved on, and the baseline quietly describes a machine that
no longer exists. So this checks four separate things, and a failure in any
one of them is a failure:

  the evidence exists and is INTERNALLY consistent
      -- by running R1's own verifier, which recomputes hashes and re-derives
         the required/discovered/executed relation. R2 does not re-implement
         that; re-implementing a verifier is how two verifiers come to
         disagree.

  it says ACCEPTED, and says it about all twelve work packages
      -- "PARTIAL" is the state R1 is in while it is being worked on, and the
         contract names it: a partial R1 is not a prerequisite that has been
         met. A single environment's run also cannot be ACCEPTED, because
         R1-09 and R1-12 are settled over two; so a baseline is either a
         combined verdict from compare_envs.py, or two runs plus that
         comparison.

  it is about THIS candidate, or a stated ancestor of it
      -- R2 may evolve the R1 fixes (the contract allows it) but the
         relationship has to be recorded, not assumed.

  the things R2 reads out of it are actually there
      -- profile fingerprints, the toolchain lock, the normalisation rules
         and the case inventory. R2 binds artifacts to a profile fingerprint;
         a baseline without one cannot support that binding, and guessing the
         fingerprint is exactly what ART-03 exists to catch.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))

R1_VERIFY = os.path.join(ROOT, 'tools', 'r1', 'verify_evidence.py')
R1_PACKAGES = [f"R1-{i:02d}" for i in range(1, 13)]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def check(evidence, *, candidate=None, combined=None):
    """Returns (problems, facts). Facts are what R2 may then rely on."""
    problems, facts = [], {}

    if not evidence or not os.path.isdir(evidence):
        return ([f"no R1 evidence directory at {evidence!r}: R2 cannot be "
                 "accepted without a passed R1 to stand on"], facts)

    acc_path = os.path.join(evidence, 'acceptance.json')
    if not os.path.exists(acc_path):
        return ([f"{evidence} has no acceptance.json; it is not an R1 "
                 "evidence directory"], facts)

    # 1. R1's own verifier, not a second implementation of it.
    r = subprocess.run([sys.executable, R1_VERIFY, evidence],
                       capture_output=True, text=True, timeout=600)
    facts['verifier_exit'] = r.returncode
    if r.returncode != 0:
        problems.append(
            "R1's evidence verifier rejected this directory: "
            + ' / '.join(l.strip() for l in r.stdout.splitlines()[:4]))

    acc = load(acc_path)
    run = load(os.path.join(evidence, 'run.json'))
    facts['r1_commit'] = (run.get('git') or {}).get('commit')
    facts['r1_verdict'] = acc.get('verdict')
    facts['r1_required'] = acc.get('required')
    facts['r1_passed'] = acc.get('passed')

    # 2. ACCEPTED, over all twelve packages.
    #
    # A single run leaves R1-09 and R1-12 open by construction, so a single
    # run is not a baseline on its own; the combined verdict from
    # compare_envs.py is. Accepting one run here would be accepting exactly
    # the thing R1's own gate refuses to conclude.
    wps = acc.get('work_packages') or {}
    if not wps:
        problems.append(
            "this R1 run recorded no work-package roll-up; it predates the "
            "roll-up and cannot show that all twelve passed")
    deferred = {k for k, v in wps.items() if v.get('deferred_to')}
    unmet = sorted(k for k in R1_PACKAGES
                   if (wps.get(k) or {}).get('verdict') != 'PASS')
    facts['r1_unmet'] = unmet
    facts['r1_deferred'] = sorted(deferred)

    if combined:
        if not os.path.exists(combined):
            problems.append(f"no combined verdict at {combined!r}")
        else:
            comb = load(combined)
            facts['combined_verdict'] = comb.get('verdict')
            if comb.get('verdict') != 'ACCEPTED':
                problems.append(
                    f"the two-environment comparison says "
                    f"{comb.get('verdict')}: "
                    + ' / '.join(comb.get('problems', [])[:3]))
            resolved = set(comb.get('resolved') or {})
            still = [k for k in unmet if k not in resolved]
            if still:
                problems.append(
                    "work package(s) not passing and not resolved by the "
                    "comparison: " + ', '.join(still))
    else:
        if unmet:
            problems.append(
                "R1 work package(s) not PASS in this run: "
                + ', '.join(unmet)
                + (f" (of which {', '.join(sorted(deferred & set(unmet)))} "
                   "are settled by the two-environment comparison -- pass "
                   "--combined to supply it)" if deferred & set(unmet) else ""))
        if acc.get('verdict') != 'ACCEPTED':
            problems.append(
                f"R1 verdict is {acc.get('verdict')!r}, not ACCEPTED. "
                "A PARTIAL or in-progress R1 is not a prerequisite that has "
                "been met (DEP-01)")

    # 3. About this candidate, or a recorded ancestor of it.
    if candidate:
        facts['r2_candidate'] = candidate
        if facts['r1_commit'] and facts['r1_commit'] != candidate:
            rc = subprocess.run(
                ['git', 'merge-base', '--is-ancestor',
                 facts['r1_commit'], candidate],
                cwd=ROOT, capture_output=True, text=True)
            facts['r1_is_ancestor'] = (rc.returncode == 0)
            if rc.returncode != 0:
                problems.append(
                    f"the R1 evidence is for {facts['r1_commit'][:12]}, which "
                    f"is not an ancestor of the R2 candidate "
                    f"{candidate[:12]}: the baseline describes a different "
                    "line of development")
        else:
            facts['r1_is_ancestor'] = True

    # 4. The specific inputs R2 reads out of it.
    for rel, why in (
            ('profiles/manifests.json',
             'R2 binds an artifact to a profile fingerprint'),
            ('inventory.json',
             'R2 must show the R1 case set did not shrink'),
            ('results.json',
             'R2 compares per-case verdicts before and after its changes')):
        p = os.path.join(evidence, rel)
        if not os.path.exists(p):
            problems.append(f"missing {rel} -- {why}")
    man = os.path.join(evidence, 'profiles', 'manifests.json')
    if os.path.exists(man):
        fps = {k: v.get('config_hash') for k, v in load(man).items()}
        facts['profile_fingerprints'] = fps
        if not all(fps.values()):
            problems.append(
                "a profile in the baseline has no config hash; an artifact "
                "cannot be bound to a fingerprint that does not exist")
    for rel in ('spec/r1_toolchain.lock.json', 'spec/r1_normalization.json',
                'spec/r1_required_cases.json'):
        if not os.path.exists(os.path.join(ROOT, rel)):
            problems.append(f"the candidate has no {rel}")
    return problems, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', help='an R1 run directory')
    ap.add_argument('--combined', help='compare_envs.py output for A and B')
    ap.add_argument('--candidate', help='the R2 candidate commit')
    ap.add_argument('--json')
    args = ap.parse_args()

    candidate = args.candidate
    if candidate is None:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                           capture_output=True, text=True)
        candidate = r.stdout.strip() if r.returncode == 0 else None

    print("R2-01 baseline: the R1 result R2 stands on")
    problems, facts = check(args.evidence, candidate=candidate,
                            combined=args.combined)
    for k in sorted(facts):
        print(f"    {k}: {facts[k]}")
    if args.json:
        import r1lib as R
        R.write_json(args.json, dict(ok=not problems, problems=problems,
                                     facts=facts))
    if problems:
        print(f"  r2-baseline            FAIL   {len(problems)} problem(s)")
        for p in problems[:8]:
            print('    ' + p)
        return 1
    print("  r2-baseline            PASS   a passed R1, about this candidate")
    return 0


if __name__ == '__main__':
    sys.exit(main())
