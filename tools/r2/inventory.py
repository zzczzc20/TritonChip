#!/usr/bin/env python3
"""
The R2 required-case inventory -- doc/phase2_tasks_and_acceptance.md R2-01.

Same job as R1's, with one addition the contract asks for explicitly:

    "required/discovered/executed 的比较以每个环境/profile/seed 展开后的唯一
     执行 ID 为准；共享语义 case ID 保留用于追踪。不能把同一 case 跑两次当成
     另一个遗漏 case。"

So an entry here is a *semantic* case, and the thing compared is its
expansion: `MEM-01` over profile `system-2x2` and seed 7 is one execution ID,
and running it twice on seed 7 does not stand in for seed 42. The expansion
rule lives in the spec next to the case, rather than in whichever runner
happens to execute it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import inventory as R1INV    # noqa: E402  (NOT_A_PASS, reconcile)

SPEC = os.path.join(ROOT, 'spec', 'r2_required_cases.json')

# Families whose cases run once per seed. The rest run once per profile.
SEEDED = {'MEM', 'E2E', 'GRID'}


def load_spec(path=SPEC):
    with open(path) as fh:
        return json.load(fh)


def expand(case, seeds):
    """The unique execution IDs one semantic case becomes.

    A case with profiles ['n/a'] is not a device case and expands to itself:
    inventing `PKG-01@n/a` would be a longer name for the same thing.
    """
    fam = case['id'].split('-')[0]
    profiles = case.get('profiles') or ['n/a']
    out = []
    for p in profiles:
        base = case['id'] if p == 'n/a' else f"{case['id']}@{p}"
        if fam in SEEDED and p != 'n/a':
            out.extend(f"{base}#seed{s}" for s in seeds)
        else:
            out.append(base)
    return out


def required_ids(spec=None):
    spec = spec or load_spec()
    seeds = spec.get('seeds') or [1]
    out = []
    for case in spec['cases']:
        out.extend(expand(case, seeds))
    return sorted(set(out))


def semantic_of(execution_id):
    """The semantic case an execution ID belongs to, for tracking."""
    return execution_id.split('@')[0].split('#')[0]


def spec_problems(spec=None):
    """The same well-formedness rules R1 froze, on R2's inventory."""
    spec = spec or load_spec()
    problems = []
    seen = set()
    for case in spec['cases']:
        cid = case.get('id')
        if not cid:
            problems.append('a case with no id')
            continue
        if cid in seen:
            problems.append(f"duplicate case id {cid}")
        seen.add(cid)
        for field in ('group', 'entry', 'expect', 'source', 'work_package',
                      'baseline_status'):
            value = case.get(field)
            if value is None or value == '':
                problems.append(f"{cid}: {field} is empty")
                continue
            if not isinstance(value, str):
                continue
            # A field that IS a placeholder, not one that mentions the word.
            # REL-05's acceptance text says "不得留下 ... TBD/例外" -- a rule
            # ABOUT placeholders -- and a substring search called that a
            # placeholder, which is the check failing the case it was written
            # to protect.
            head = value.strip().upper()
            if head in ('TBD', 'TODO', 'FIXME', 'N/A', 'NA', '-', '?'):
                problems.append(f"{cid}: {field} is the placeholder {value!r}")
            elif field in ('group', 'entry', 'source', 'work_package',
                           'baseline_status') and any(
                    t in head for t in ('TBD', 'TODO', 'FIXME')):
                problems.append(f"{cid}: {field} says {value!r}")
        if not case.get('profiles'):
            problems.append(f"{cid}: no profiles, so it cannot be expanded")
        if case.get('baseline_status') in ('PASS',):
            problems.append(
                f"{cid}: baseline_status PASS -- nothing in R2 has run yet, "
                "and a case cannot start out passing")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    args = ap.parse_args()
    spec = load_spec()
    problems = spec_problems(spec)
    ids = required_ids(spec)
    by_wp = {}
    for c in spec['cases']:
        by_wp[c['work_package']] = by_wp.get(c['work_package'], 0) + 1
    print(f"spec: {len(spec['cases'])} semantic case(s) -> {len(ids)} "
          f"execution ID(s) after expanding profile and seed, "
          f"{len(problems)} problem(s)")
    for wp in sorted(by_wp):
        print(f"    {wp}: {by_wp[wp]}")
    for p in problems[:10]:
        print('  ' + p)
    if args.json:
        import r1lib as R
        R.write_json(args.json, dict(ok=not problems, problems=problems,
                                     required=ids))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
