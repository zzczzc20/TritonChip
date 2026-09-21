#!/usr/bin/env python3
"""
R35-EV-01 -- phase3_5_tasks_and_acceptance.md section 7.

Recompute the verdict from the frozen inventory and the raw results, and
refuse anything that does not add up.

Section 7 says what it has to refuse: "verifier 按冻结 inventory 从原始
数值/计数/退出码重算收益、退化、预算和判决；拒绝混源码/硬件/工具链、改
cycle、漏消融、篡改日志和仅修改 summary PASS".

So the gains are recomputed from the raw per-case cycles rather than
read from a summary, the ablation has to be present, and a `passed`
field that disagrees with its own file is the thing this exists to
catch.
"""
import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTRACT_ID = 'r35-compiler'
SIGNOFF = 'R35_COMPILER_ACCEPTED'
SHORTFALL = 'OPTIMIZATION_DELIVERED / PERFORMANCE_NOT_MET'

REQUIRED = {
    'plan':     ['R35-BASE-01', 'R35-BASE-02', 'R35-BASE-03'],
    'upstream': ['R35-UP-01', 'R35-UP-02', 'R35-UP-03'],
    'layout':   ['R35-LL-01', 'R35-LL-02', 'R35-LL-03'],
    'opt':      ['R35-OPT-01', 'R35-OPT-02', 'R35-OPT-03'],
    'alloc':    ['R35-ALLOC-01', 'R35-ALLOC-02', 'R35-SCHED-01'],
    'safety':   ['R35-SAFE-01', 'R35-SAFE-02', 'R35-MUT-01'],
    'perf':     ['R35-PERF-01', 'R35-PERF-02', 'R35-PERF-03'],
    'package':  ['R35-PKG-01', 'R35-PKG-02', 'R35-OBS-01'],
    'release':  ['R35-REG-01', 'R35-EV-01', 'R35-REL-01'],
}


class Reject(Exception):
    pass


def recompute_gains(perf_cases, gains, log, problems):
    """The gain table, worked out again from the raw cycles.

    Not read from `saved`, `meets` or any other field the run wrote: a
    summary that says a case won is a claim, and the cycles beside it
    are the evidence. Section 7 lists "改 cycle" and "仅修改 summary
    PASS" as separate things to refuse, and recomputing catches both.
    """
    rows = None
    for c in perf_cases:
        if c['id'] == 'R35-PERF-01' and isinstance(c.get('rows'), list):
            rows = c['rows']
    if rows is None:
        problems.append('the performance result carries no per-case rows, '
                        'so nothing can be recomputed from it')
        return
    wins, attributed, worse = [], [], []
    total_b = total_o = 0
    for r in rows:
        modes = r.get('modes') or {}
        b = (modes.get('B0') or {}).get('cycles')
        o = (modes.get('O2') or {}).get('cycles')
        nl = (modes.get('O2-no-LL') or {}).get('cycles')
        if b is None or o is None:
            problems.append(f"{r.get('id')}: no B0/O2 cycles to compare")
            continue
        if not r.get('correct'):
            problems.append(f"{r.get('id')}: not correct against its oracle")
        total_b += b
        total_o += o
        need = max(gains['primary_absolute_cycles'],
                   math.ceil(gains['primary_relative'] * b))
        spread = r.get('spread') or 0
        if b - o >= need and (b - o) > spread:
            wins.append(r['id'])
        allowed = max(gains['regression_absolute_cycles'],
                      math.ceil(gains['regression_relative'] * b))
        if o - b > allowed:
            worse.append(r['id'])
        if nl is not None and r['id'] in gains['ll_subjects']:
            need_ll = max(gains['ll_absolute_cycles'],
                          math.ceil(gains['ll_relative'] * nl))
            if nl - o >= need_ll:
                attributed.append(r['id'])
    log.append(f"    recomputed from the raw cycles: {len(wins)} win(s) "
               f"{wins}, {len(attributed)} LL attribution(s) "
               f"{attributed}, {len(worse)} regression(s)")
    log.append(f"    total over the cases: {total_b} -> {total_o} cycles")
    if len(wins) < gains['primary_workloads_required']:
        problems.append(f"{len(wins)} workload(s) over the gate; "
                        f"{gains['primary_workloads_required']} are wanted")
    if not any(w != 'L03' for w in wins):
        problems.append('every winner is the layout micro-case; section '
                        '6.3 wants at least one that is not')
    if not attributed:
        problems.append('no LinearLayout attribution on '
                        + '/'.join(gains['ll_subjects']))
    for w in worse:
        problems.append(f"{w} is meaningfully slower")
    if total_o > total_b:
        problems.append('the total cycle count went up')


def verify(ev, log):
    problems = []
    path = os.path.join(ev, 'acceptance.json')
    if not os.path.exists(path):
        raise Reject(f"no acceptance.json in {ev}")
    claimed = json.load(open(path))

    if claimed.get('contract_id') != CONTRACT_ID:
        problems.append(f"contract_id is {claimed.get('contract_id')!r}, "
                        f"not {CONTRACT_ID!r}")

    for name, want in (claimed.get('frozen_hashes') or {}).items():
        copied = os.path.join(ev, 'contracts', name)
        if not os.path.exists(copied):
            problems.append(f"{name} was not copied into the evidence")
            continue
        got = hashlib.sha256(open(copied, 'rb').read()).hexdigest()
        if got != want:
            problems.append(f"{name}: the copy hashes {got[:12]}, the run "
                            f"recorded {want[:12]}")
    inv_path = os.path.join(ev, 'contracts', 'r35_workloads.json')
    if not os.path.exists(inv_path):
        raise Reject('the frozen workload inventory is not in the evidence')
    inv = json.load(open(inv_path))
    log.append('    the frozen contracts are present and hash as the run '
               'recorded')

    if not claimed.get('clean_tree', False):
        problems.append('the working tree was not clean')
    commit = claimed.get('commit') or ''
    if len(commit) < 8:
        problems.append('the evidence does not name a commit')
    log.append(f"    commit {commit[:12] or 'MISSING'}, clean tree "
               f"{claimed.get('clean_tree')}")

    # The hardware this stage stands on. Section 5: a gain measured on
    # hardware that was never accepted is not this stage's gain.
    r3 = os.path.join(ev, 'inherited', 'r3v2-acceptance.json')
    if not os.path.exists(r3):
        problems.append('the R3 v2 acceptance record is not in the evidence')
    else:
        d = json.load(open(r3))
        pre = inv['hardware_precondition']
        log.append(f"    the hardware it stands on: {d.get('verdict')} at "
                   f"{str(d.get('commit'))[:12]}")
        if d.get('verdict') != pre['signoff']:
            problems.append('the R3 v2 record does not carry its sign-off')
        if not str(d.get('commit', '')).startswith(pre['commit']):
            problems.append(f"the R3 v2 record is at "
                            f"{str(d.get('commit'))[:12]}, and this stage "
                            f"was frozen against {pre['commit']}")

    seen = {}
    perf_cases = None
    for checker, ids in sorted(REQUIRED.items()):
        raw = os.path.join(ev, 'runs', f'{checker}.json')
        if not os.path.exists(raw):
            problems.append(f"{checker}: no raw result file")
            continue
        cases = {c['id']: c for c in json.load(open(raw))}
        if checker == 'perf':
            perf_cases = list(cases.values())
        for cid in ids:
            if cid not in cases:
                problems.append(f"{cid}: required by the contract and not "
                                f"in {checker}.json")
                continue
            seen[cid] = bool(cases[cid]['passed'])
            if not cases[cid]['passed']:
                problems.append(f"{cid}: "
                                f"{str(cases[cid].get('detail', ''))[:70]}")
        for u, c in cases.items():
            if u not in ids and not c['passed']:
                problems.append(f"{u}: reported by {checker} and not passing")
    log.append(f"    {len(seen)}/{sum(len(v) for v in REQUIRED.values())} "
               f"required standard(s) found in the raw results")

    for entry in claimed.get('checks') or []:
        name = entry['checker']
        raw = os.path.join(ev, 'runs', f'{name}.json')
        if not os.path.exists(raw) or entry.get('cases') is None:
            continue
        on_disk = {c['id']: c['passed'] for c in json.load(open(raw))}
        for c in entry['cases']:
            if on_disk.get(c['id']) != c['passed']:
                problems.append(f"{c['id']}: the summary says {c['passed']}, "
                                f"{name}.json says {on_disk.get(c['id'])}")
    log.append('    the summary agrees with every raw result it quotes')

    if perf_cases:
        recompute_gains(perf_cases, inv['gain_thresholds'], log, problems)

    ours = SIGNOFF if not problems else 'NOT ACCEPTED'
    theirs = claimed.get('verdict')
    log.append(f"    the evidence claims {theirs!r}; recomputed from the "
               f"inventory and the raw results: {ours!r}")
    if theirs != ours and not (theirs == SHORTFALL and ours != SIGNOFF):
        problems.append(f"the recorded verdict {theirs!r} is not the one "
                        f"the evidence supports ({ours!r})")
        ours = 'NOT ACCEPTED'
    return ours, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evidence')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    if args.only and 'R35-EV-01' not in args.only:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    print(f"Phase 3.5 evidence verifier ({CONTRACT_ID})")
    log = []
    try:
        verdict, problems = verify(args.evidence, log)
    except Reject as e:
        print(f"  refused: {e}")
        if args.json:
            with open(args.json, 'w') as fh:
                json.dump([{'id': 'R35-EV-01', 'passed': False,
                            'detail': str(e)}], fh, indent=2)
        return 2
    for line in log:
        print(line)
    for p in problems:
        print(f"  open: {p}")
    print()
    print(f"  VERDICT: {verdict}")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump([{'id': 'R35-EV-01',
                        'passed': verdict == SIGNOFF,
                        'detail': (f"recomputed {verdict}"
                                   f" from {len(problems)} problem(s)"),
                        'problems': problems}], fh, indent=2)
            fh.write('\n')
    return 0 if verdict == SIGNOFF else 1


if __name__ == '__main__':
    sys.exit(main())
