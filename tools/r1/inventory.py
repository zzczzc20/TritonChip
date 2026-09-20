#!/usr/bin/env python3
"""
The required / discovered / executed sets -- R1-01 and R1-02.

The contract's final verdict includes `required IDs == discovered IDs ==
executed IDs`. That is a stronger statement than "all tests passed", and it is
the one that catches the failure modes a pass count cannot: a suite that
discovered fewer cases than it should have, a case renamed out of existence, a
case counted twice, or a case that reported SKIP and was read as success.

Everything here is a pure function over sets and statuses so the gate's
self-test can inject each of those without running anything.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SPEC = os.path.join(ROOT, 'spec', 'r1_required_cases.json')

# Statuses some frameworks report as success and the contract does not.
NOT_A_PASS = ('SKIP', 'XFAIL', 'XPASS', 'FAIL', 'ERROR', 'TIMEOUT', 'NOT RUN')


def load_spec(path=SPEC):
    with open(path) as fh:
        return json.load(fh)


def required_ids(spec=None):
    spec = spec or load_spec()
    return {c['id'] for c in spec['cases']}


def reconcile(required, discovered, statuses):
    """Compare the three sets and return every way they disagree.

    `discovered` is a list, not a set, so a duplicate ID is visible -- running
    one case twice must not be able to stand in for a case that never ran.
    """
    problems = []
    seen = {}
    for cid in discovered:
        seen[cid] = seen.get(cid, 0) + 1
    dupes = sorted(k for k, n in seen.items() if n > 1)
    disc = set(seen)

    if not discovered:
        problems.append('no cases were discovered')
    if dupes:
        problems.append(f"duplicate case IDs: {dupes}")
    missing = sorted(set(required) - disc)
    if missing:
        problems.append(f"required but not discovered: {missing}")
    extra = sorted(disc - set(required))
    if extra:
        problems.append(f"discovered but not required: {extra}")

    executed = {cid for cid, st in statuses.items() if st == 'PASS'}
    for cid in sorted(disc):
        st = statuses.get(cid, 'NOT RUN')
        if st in NOT_A_PASS:
            problems.append(f"{cid}: {st}")

    return dict(required=sorted(required), discovered=sorted(disc),
                executed=sorted(executed), duplicates=dupes,
                problems=problems)


def require_fresh_artifact(path, *, min_bytes=1):
    """An output this run produced, not one left behind by the last one.

    RUN-08: the gate must fail when its own dump is missing, even though a
    perfectly good dump from the previous run is sitting next to it.
    """
    if not os.path.exists(path):
        return False, f"{os.path.basename(path)} was not produced by this run"
    if os.path.getsize(path) < min_bytes:
        return False, f"{os.path.basename(path)} is empty"
    return True, 'present'


# --- static guards ---------------------------------------------------------
# A pipeline whose exit code is a verdict must be guarded, or the verdict is
# the last stage's. Checked statically so the fix cannot silently regress.
_PIPE_RE = re.compile(r'^\t.*\|\s*(tail|head|grep|tee)\b')

WATCHED = [
    'Makefile',
]


def unguarded_pipelines():
    out = []
    for rel in WATCHED:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        for n, line in enumerate(open(path), 1):
            if _PIPE_RE.match(line) and 'pipefail' not in line:
                out.append(f"{rel}:{n}")
    # Python runners that shell out with a pipeline need pipefail too.
    for rel in ('compiler/test/IR/run_ir_tests.py',):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            src = open(path).read()
            if 'shell=True' in src and 'pipefail' not in src:
                out.append(f"{rel}: shell=True pipeline without pipefail")
    return out


def spec_problems(spec=None):
    """R1-01: the frozen inventory must itself be well formed."""
    spec = spec or load_spec()
    problems = []
    seen = set()
    for c in spec['cases']:
        for field in ('id', 'group', 'entry', 'profile', 'expect',
                      'source', 'work_package', 'baseline_status'):
            if not c.get(field):
                problems.append(f"{c.get('id', '?')}: empty field {field!r}")
        if c['id'] in seen:
            problems.append(f"duplicate id in the spec: {c['id']}")
        seen.add(c['id'])
        blob = json.dumps(c)
        for banned in ('TBD', 'TODO', 'N/A', 'not applicable'):
            if banned in blob:
                problems.append(f"{c['id']}: contains {banned!r}")
    if not spec['cases']:
        problems.append('the required-case list is empty')
    return problems


if __name__ == '__main__':
    import sys
    sp = load_spec()
    probs = spec_problems(sp)
    print(f"spec: {len(sp['cases'])} required cases, {len(probs)} problem(s)")
    for p in probs:
        print('  ' + p)
    pipes = unguarded_pipelines()
    if pipes:
        print("unguarded pipelines:")
        for p in pipes:
            print('  ' + p)
    sys.exit(1 if (probs or pipes) else 0)
