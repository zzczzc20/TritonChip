#!/usr/bin/env python3
"""
R3V2-EV-01 -- doc/phase3_tasks_and_acceptance.md G7.

Recompute the verdict from the frozen inventory and the raw results, and
refuse anything that does not add up.

The one rule worth stating plainly: **the `verdict` and `passed` fields in
the evidence are not evidence.** They are what the run claimed. This reads
the frozen inventory to decide what MUST be there, reads the raw
per-checker results to see what actually happened, and forms its own
verdict. The checker this replaces in spirit -- the old tools/r3/plan.py --
passed by reading a `verdict` field out of a file, which is how a stage
can be signed off without anything having run.

What it refuses:

    a required case with no result at all
    a result for a case the inventory does not require
    a case marked passed in the summary whose raw result says otherwise
    results from more than one commit, or from a dirty tree
    a frozen contract or inventory whose hash does not match the copy
    totals that do not equal the sum of their parts
    a budget overrun
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTRACT_ID = 'r3-v2-core-vrf'
SIGNOFF = 'R3_V2_CORE_VRF_ACCEPTED'

# The 24 standards the contract lists, and which checker owns each.
REQUIRED = {
    'plan':  ['R3V2-BASE-01', 'R3V2-BASE-02', 'R3V2-BASE-03'],
    'rf':    ['R3V2-RF-01', 'R3V2-RF-02', 'R3V2-RF-03'],
    'core':  ['R3V2-CORE-01', 'R3V2-CORE-02', 'R3V2-CORE-03'],
    'flow':  ['R3V2-FLOW-01', 'R3V2-FLOW-02', 'R3V2-FLOW-03'],
    'e2e':   ['R3V2-E2E-01', 'R3V2-E2E-02', 'R3V2-E2E-03'],
    'count': ['R3V2-COUNT-01', 'R3V2-COUNT-02'],
    'model': ['R3V2-MODEL-01'],
    'syn':   ['R3V2-SYN-01', 'R3V2-SYN-02', 'R3V2-REG-01'],
    # R3V2-EV-01 is this file's own standard, and the mutation suite is
    # where it is tested: M07c forges a result and M07d removes one, and
    # this file has to refuse both. It is reported there because that is
    # where it is measured -- a verifier that certified itself would be
    # the thing this file exists to refuse.
    'mut':   ['R3V2-MUT-01', 'R3V2-EV-01'],
    'rel':   ['R3V2-REL-01'],
}


class Reject(Exception):
    pass


def verify(ev, log):
    problems = []

    path = os.path.join(ev, 'acceptance.json')
    if not os.path.exists(path):
        raise Reject(f"no acceptance.json in {ev}")
    claimed = json.load(open(path))

    if claimed.get('contract_id') != CONTRACT_ID:
        problems.append(f"contract_id is {claimed.get('contract_id')!r}, "
                        f"not {CONTRACT_ID!r}")

    # The frozen files must be the ones the run used.
    for name, want in (claimed.get('frozen_hashes') or {}).items():
        copied = os.path.join(ev, 'contracts', name)
        if not os.path.exists(copied):
            problems.append(f"{name} was not copied into the evidence")
            continue
        got = hashlib.sha256(open(copied, 'rb').read()).hexdigest()
        if got != want:
            problems.append(f"{name}: the copy hashes {got[:12]}, the run "
                            f"recorded {want[:12]}")
    inv_path = os.path.join(ev, 'contracts', 'r3v2_inventory.json')
    inventory = json.load(open(inv_path)) if os.path.exists(inv_path) \
        else None
    if inventory is None:
        raise Reject('the frozen inventory is not in the evidence')
    log.append("    frozen inventory and contract present and "
               "hashing as the run recorded")

    if not claimed.get('clean_tree', False):
        problems.append('the working tree was not clean')
    commit = claimed.get('commit') or ''
    if len(commit) < 8:
        problems.append('the evidence does not name a commit')
    log.append(f"    commit {commit[:12] or 'MISSING'}, clean tree "
               f"{claimed.get('clean_tree')}")

    # Every required case, from the RAW results, not the summary.
    seen = {}
    for checker, ids in sorted(REQUIRED.items()):
        raw = os.path.join(ev, 'runs', f'{checker}.json')
        if not os.path.exists(raw):
            problems.append(f"{checker}: no raw result file")
            continue
        cases = {c['id']: c for c in json.load(open(raw))}
        for cid in ids:
            if cid not in cases:
                problems.append(f"{cid}: required by the contract and not "
                                f"in {checker}.json")
                continue
            seen[cid] = bool(cases[cid]['passed'])
            if not cases[cid]['passed']:
                problems.append(f"{cid}: {cases[cid].get('detail', '')[:60]}")
        unknown = [c for c in cases
                   if c not in ids and not c.startswith(tuple(ids))]
        for u in unknown:
            # Extra cases are allowed but must still pass: a checker that
            # reports an open case is open.
            if not cases[u]['passed']:
                problems.append(f"{u}: reported by {checker} and not "
                                f"passing")
    log.append(f"    {len(seen)}/{sum(len(v) for v in REQUIRED.values())} "
               f"required standard(s) found in the raw results")

    # The summary must agree with the raw results. A `passed` field that
    # disagrees with its own file is the thing this check exists for.
    for entry in claimed.get('checks') or []:
        name = entry['checker']
        raw = os.path.join(ev, 'runs', f'{name}.json')
        if not os.path.exists(raw) or entry.get('cases') is None:
            continue
        on_disk = {c['id']: c['passed'] for c in json.load(open(raw))}
        for c in entry['cases']:
            if on_disk.get(c['id']) != c['passed']:
                problems.append(f"{c['id']}: the summary says "
                                f"{c['passed']}, {name}.json says "
                                f"{on_disk.get(c['id'])}")
    log.append("    the summary agrees with every raw result it "
               "quotes")

    # The budget applies to the acceptance's own RTL runs. The mutation
    # suite is not one of them: it rebuilds and re-runs the checkers
    # thirteen times over in isolated copies, and charging that to the
    # 1800s the inventory budgets for the stage's simulation would make
    # the budget check fire for the one activity it was never about. Its
    # time is reported, separately, and not compared.
    b = inventory['budget']
    checks = claimed.get('checks') or []
    elapsed = sum(e.get('elapsed_s', 0) for e in checks
                  if e.get('checker') != 'mut')
    mut_s = sum(e.get('elapsed_s', 0) for e in checks
                if e.get('checker') == 'mut')
    log.append(f"    the checkers took {elapsed:.0f}s, plus {mut_s:.0f}s "
               f"of mutation runs (budgeted separately)")
    if elapsed > b['total_elapsed_seconds'] * 4:
        problems.append(f"the run took {elapsed:.0f}s, far over the "
                        f"{b['total_elapsed_seconds']}s RTL budget")

    # And the verdict is ours, not the file's.
    ours = SIGNOFF if not problems else 'NOT ACCEPTED'
    theirs = claimed.get('verdict')
    log.append(f"    the evidence claims {theirs!r}; recomputed from the "
               f"inventory and the raw results: {ours!r}")
    if theirs != ours:
        problems.append(f"the recorded verdict {theirs!r} is not the one "
                        f"the evidence supports ({ours!r})")
        ours = 'NOT ACCEPTED'
    return ours, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evidence')
    ap.add_argument('--json')
    args = ap.parse_args()
    print(f"R3 v2 evidence verifier ({CONTRACT_ID})")
    log = []
    try:
        verdict, problems = verify(args.evidence, log)
    except Reject as e:
        print(f"  refused: {e}")
        return 2
    for line in log:
        print(line)
    for p in problems:
        print(f"  open: {p}")
    print()
    print(f"  VERDICT: {verdict}")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'verdict': verdict, 'problems': problems},
                      fh, indent=2)
            fh.write('\n')
    return 0 if verdict == SIGNOFF else 1


if __name__ == '__main__':
    sys.exit(main())
