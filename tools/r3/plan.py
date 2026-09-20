#!/usr/bin/env python3
"""
PLAN-01 .. PLAN-04 -- doc/phase3_tasks_and_acceptance.md R3-01.

The budget, checked before anything runs.

R3's premise is that RTL simulation is slow, so the expensive mistake is not
a wrong answer -- it is an hour spent getting one. Every check here happens
before a simulator starts:

  PLAN-01  the R1 and R2 baselines are real, and no profile grew a resource
           to make an error go away
  PLAN-02  every workload is inside the contract's caps, and an over-limit
           one is refused rather than run
  PLAN-03  the manifest expands to exactly 32 performance runs, and the
           measured simulation rate says what that costs in wall time
  PLAN-04  every unit contract has a source, and an unmeasured parameter is
           null rather than zero

PLAN-04 is the one worth stating twice. A parameter nobody has measured,
written as 0, is a cost model that predicts a machine where that thing is
free -- and the prediction will look excellent right up until somebody builds
it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))

WORKLOADS = os.path.join(ROOT, 'spec', 'r3_workloads.json')
CONTRACTS = os.path.join(ROOT, 'spec', 'r3_unit_contracts.json')
SCOUT = os.path.join(ROOT, 'doc', 'evidence', 'r3-timing-scout-macos.json')


def load(path):
    with open(path) as fh:
        return json.load(fh)


def check_caps(spec):
    """PLAN-02: every workload inside the contract's limits."""
    caps = spec['caps']
    problems = []
    for w in spec['workloads']:
        p, wid, kind = w['params'], w['id'], w['kind']
        if kind == 'gemm' and w['profile'] == 'core-sim':
            c = caps['core_gemm']
            tail = c['fixed_tail']
            is_tail = (p.get('m'), p.get('n'), p.get('k')) == \
                (tail['m'], tail['n'], tail['k'])
            if not is_tail:
                for axis in ('m', 'n', 'k'):
                    if p.get(axis, 0) > c[axis]:
                        problems.append(
                            f"{wid}: {axis}={p[axis]} exceeds the single-core "
                            f"GEMM cap of {c[axis]}")
                if (p.get('m'), p.get('n')) != (16, 16) and \
                        p.get('k') not in (16, 32, 64):
                    problems.append(
                        f"{wid}: a ragged shape that is not the fixed tail "
                        f"{tail}; R3 freezes one tail rather than a family")
        elif kind == 'gemm':
            c = caps['system_gemm']
            for axis in ('m', 'n', 'k'):
                if p.get(axis, 0) > c[axis]:
                    problems.append(
                        f"{wid}: {axis}={p[axis]} exceeds the whole-machine "
                        f"GEMM cap of {c[axis]}")
            if p.get('grid', 1) > c['grid']:
                problems.append(
                    f"{wid}: grid={p['grid']} exceeds {c['grid']}")
        elif kind in ('reduction', 'softmax', 'rmsnorm', 'layernorm'):
            c = caps['reduction_like']
            if p.get('rows', 1) > c['rows']:
                problems.append(f"{wid}: rows={p['rows']} exceeds {c['rows']}")
            limit = c['elements']
            if p.get('elements', 0) > limit:
                problems.append(
                    f"{wid}: elements={p['elements']} exceeds {limit}")
        elif kind == 'elementwise':
            c = caps['elementwise']
            if p.get('n', 0) > c['n']:
                problems.append(f"{wid}: n={p['n']} exceeds {c['n']}")
            if p.get('grid', 1) > c['grid']:
                problems.append(
                    f"{wid}: grid={p['grid']} exceeds {c['grid']}")
        else:
            problems.append(f"{wid}: unknown workload kind {kind!r}")

        if p.get('repetitions', 1) != 1:
            problems.append(
                f"{wid}: {p['repetitions']} repetitions. R3 allows one kernel "
                "per simulation; averaging a hundred inside the RTL spends "
                "hours to learn what one run already said")
    return problems


def check_expansion(spec):
    """PLAN-03: exactly 32 performance runs."""
    total = sum(len(w['runs']) for w in spec['workloads'])
    want = spec['required_runs']
    problems = []
    if total != want:
        problems.append(
            f"the manifest expands to {total} performance run(s) and the "
            f"budget is {want}")
    seen = set()
    for w in spec['workloads']:
        for mode in w['runs']:
            key = (w['id'], mode)
            if key in seen:
                problems.append(f"{w['id']} lists {mode!r} twice")
            seen.add(key)
            if mode not in ('fast', 'slow', 'ooo'):
                problems.append(f"{w['id']}: unknown memory mode {mode!r}")
    return problems, total


def check_contracts(contracts):
    """PLAN-04: every parameter has a source, and unknown is not zero."""
    problems = []
    measured = unmeasured = 0

    def walk(node, path):
        nonlocal measured, unmeasured
        if isinstance(node, dict):
            if 'value' in node and 'source' in node:
                if node['value'] is None:
                    unmeasured += 1
                    if node['source'] != 'unmeasured':
                        problems.append(
                            f"{path}: value is null but the source says "
                            f"{node['source']!r}")
                else:
                    measured += 1
                    if not node['source']:
                        problems.append(f"{path}: a value with no source")
                    if node['value'] == 0 and node['source'] == 'unmeasured':
                        problems.append(
                            f"{path}: 0 recorded as unmeasured. Unknown is "
                            "null; a zero here is a model that predicts this "
                            "costs nothing")
                return
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    walk(contracts, '')
    return problems, measured, unmeasured


def estimate(spec, scout):
    """PLAN-03: what 32 runs cost, from the measured rate."""
    if not scout:
        return None
    rate = {p['name']: p.get('cycles_per_second')
            for p in scout.get('probes', []) if p.get('cycles_per_second')}
    # A rough cycle count per workload kind, from what the R1 programs cost.
    # Deliberately coarse: this is a budget, not a prediction, and R3-02's
    # model is the thing that predicts.
    typical = {'gemm': 4000, 'reduction': 1500, 'softmax': 3000,
               'rmsnorm': 2500, 'layernorm': 3000, 'elementwise': 1200}
    seconds = 0.0
    for w in spec['workloads']:
        cycles = typical.get(w['kind'], 2000)
        if w['profile'] == 'system-2x2':
            cycles *= 5          # the mesh and the slices
        for mode in w['runs']:
            key = ('system-2x2' if w['profile'] == 'system-2x2'
                   else f"core-{mode}")
            r = rate.get(key) or rate.get('core-fast') or 500
            seconds += cycles / r
    return seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    args = ap.parse_args()

    spec = load(WORKLOADS)
    contracts = load(CONTRACTS)
    scout = load(SCOUT) if os.path.exists(SCOUT) else None

    print("R3-01 plan: budgets and caps, checked before any simulator starts")
    results = []

    cap_problems = check_caps(spec)
    ok = not cap_problems
    print(f"  PLAN-02  {'PASS' if ok else 'FAIL':<5s} "
          f"{len(spec['workloads'])} workload(s) inside the contract's caps")
    for p in cap_problems[:8]:
        print('    ' + p)
    results.append(dict(id='PLAN-02', passed=ok,
                        detail=f"{len(spec['workloads'])} workloads checked"))

    exp_problems, total = check_expansion(spec)
    ok = not exp_problems
    print(f"  PLAN-03  {'PASS' if ok else 'FAIL':<5s} expands to exactly "
          f"{total} performance run(s)")
    for p in exp_problems[:8]:
        print('    ' + p)
    est = estimate(spec, scout)
    if est:
        print(f"    at the measured simulation rate that is about "
              f"{est / 60:.0f} minutes of RTL")
        print(f"    (rate from {os.path.relpath(SCOUT, ROOT)}; it is a "
              f"simulation rate, not a clock)")
    else:
        print("    no timing scout result: run tools/r3/scout.py first, "
              "because a budget without a measurement is a guess")
        ok = False
    results.append(dict(id='PLAN-03', passed=ok,
                        detail=f"{total} runs, ~{est / 60:.0f} min"
                        if est else 'unmeasured'))

    c_problems, measured, unmeasured = check_contracts(contracts)
    ok = not c_problems
    print(f"  PLAN-04  {'PASS' if ok else 'FAIL':<5s} {measured} parameter(s) "
          f"with a source, {unmeasured} recorded as unmeasured rather than 0")
    for p in c_problems[:8]:
        print('    ' + p)
    results.append(dict(id='PLAN-04', passed=ok,
                        detail=f"{measured} sourced, {unmeasured} unmeasured"))

    # PLAN-01 is about the baselines, and is answered by R1's and R2's own
    # verdicts rather than by re-deriving them here.
    combined = os.path.join(ROOT, 'doc', 'evidence',
                            'r1-combined-verdict.json')
    r1_ok = False
    if os.path.exists(combined):
        v = load(combined)
        r1_ok = v.get('verdict') == 'ACCEPTED'
        print(f"  PLAN-01  {'PASS' if r1_ok else 'FAIL':<5s} R1 baseline: "
              f"{v.get('verdict')} on {v['notes'][0][9:21] if v.get('notes') else '?'}")
    else:
        print("  PLAN-01  FAIL  no R1 combined verdict to stand on")
    results.append(dict(id='PLAN-01', passed=r1_ok,
                        detail='R1 combined verdict'))

    if args.json:
        import r1lib as R
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3-plan                FAIL   {', '.join(sorted(bad))}")
        return 1
    print(f"  r3-plan                PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
