#!/usr/bin/env python3
"""
The R3 v2 gate -- doc/phase3_tasks_and_acceptance.md section 6.

Runs every required check once, into one run directory, and writes the
evidence the verifier reads back. A sub-target passing is not the stage
passing: this is the only thing that produces a verdict, and it returns
non-zero for a checker that does not exist yet.

It does NOT decide the verdict from its own opinion of how things went.
It records what each checker returned, and tools/r3v2/verify_evidence.py
recomputes the verdict from the frozen inventory and the raw results --
which is the point of R3V2-EV-01: a PASS field written into a file is not
evidence that anything passed.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTRACT_ID = 'r3-v2-core-vrf'
SIGNOFF = 'R3_V2_CORE_VRF_ACCEPTED'

# Every checker, the gate it belongs to, and where its raw result lands.
CHECKS = [
    ('G0', 'plan',   ['tools/r3v2/plan.py']),
    ('G1', 'rf',     ['tools/r3v2/rf_tests.py']),
    ('G2', 'core',   ['tools/r3v2/core_tests.py']),
    ('G3', 'flow',   ['tools/r3v2/flow_tests.py']),
    ('G4', 'e2e',    ['tools/r3v2/e2e_tests.py']),
    ('G5', 'count',  ['tools/r3v2/count_tests.py']),
    ('G5', 'model',  ['tools/r3v2/model_tests.py']),
    ('G6', 'syn',    ['tools/r3v2/syn_tests.py']),
    ('G7', 'mut',    ['tools/r3v2/mutation_tests.py']),
    ('G7', 'rel',    ['tools/r3v2/rel_tests.py']),
]


# Section 6 names the areas the evidence must contain. Each one holds the
# checker output that actually measures it -- the same bytes as in runs/,
# filed where a reader looks for them. An area with nothing real to put in
# it stays empty rather than getting a placeholder that reads like
# evidence.
AREA_OF = {
    'plan':  ('inventory',  'plan.json'),
    'model': ('model',      'model.json'),
    'syn':   ('synthesis',  'synthesis.json'),
    'mut':   ('mutations',  'mutations.json'),
    'rel':   ('runtime',    'second-environment.json'),
    'e2e':   ('runtime',    'end-to-end.json'),
}
# A case that belongs in an area of its own, pulled out of the checker
# that measures it.
CASE_AREA = {
    'R3V2-REG-01': ('inherited', 'inherited-regressions.json'),
}


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def tool_versions():
    out = {}
    for name, argv in (('iverilog', ['iverilog', '-V']),
                       ('yosys', ['yosys', '-V']),
                       ('python', [sys.executable, '--version'])):
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=60)
            out[name] = (r.stdout or r.stderr).splitlines()[0].strip()
        except Exception as e:
            out[name] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--python', default=sys.executable,
                    help='the interpreter the checkers run under')
    args = ap.parse_args()

    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain')
    run_id = f"{int(time.time())}-{commit[:12] or 'nogit'}"

    # Section 6: "成功重跑必须产生新的 run ID，保留失败历史". Each run gets
    # its own directory and nothing overwrites an earlier one, so a run
    # that failed stays on disk next to the one that passed. `latest`
    # points at the most recent; it is a pointer, not the record.
    root_ev = os.path.abspath(args.evidence)
    ev = os.path.join(root_ev, run_id)
    for sub in ('contracts', 'inventory', 'build', 'runs', 'runtime',
                'model', 'synthesis', 'mutations', 'inherited'):
        os.makedirs(os.path.join(ev, sub), exist_ok=True)

    print(f"R3 v2 gate: {CONTRACT_ID}")
    print(f"  run {run_id}")
    print(f"  commit {commit[:12] or 'unknown'}"
          f"{' (WORKING TREE DIRTY)' if dirty else ' (clean tree)'}")

    # The frozen contracts and inventory, copied in with their hashes, so
    # the verifier checks the results against the same ones the run used.
    import shutil
    frozen = {}
    for name in ('r3v2_contract.json', 'r3v2_inventory.json'):
        src = os.path.join(ROOT, 'spec', name)
        shutil.copy2(src, os.path.join(ev, 'contracts', name))
        frozen[name] = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    shutil.copy2(os.path.join(ROOT, 'doc',
                              'phase3_tasks_and_acceptance.md'),
                 os.path.join(ev, 'contracts', 'contract.md'))
    # The profiles are part of the contract: they are what "the candidate"
    # names, and a result is only attributable if they are recorded with it.
    shutil.copy2(os.path.join(ROOT, 'spec', 'r1_profiles.json'),
                 os.path.join(ev, 'contracts', 'r1_profiles.json'))

    # build/: what was built, from what, with which tools.
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
    import build_tests as BT
    sources = BT.tree_hashes(ROOT)
    with open(os.path.join(ev, 'build', 'build.json'), 'w') as fh:
        json.dump({'commit': commit, 'clean_tree': not dirty,
                   'run_id': run_id,
                   'platform': platform.platform(),
                   'machine': platform.machine(),
                   'tools': tool_versions(),
                   'source_files': len(sources),
                   'source_hashes': sources}, fh, indent=2, sort_keys=True)
        fh.write('\n')

    # inventory/: the frozen sub-lists, separated so each can be read on
    # its own -- which is what "按子清单独立核验" asks for.
    inv = json.load(open(os.path.join(ROOT, 'spec',
                                      'r3v2_inventory.json')))
    for key in ('core', 'runtime', 'components', 'budget', 'size_caps',
                'inherited', 'second_environment'):
        if key in inv:
            with open(os.path.join(ev, 'inventory', f'{key}.json'),
                      'w') as fh:
                json.dump(inv[key], fh, indent=2, sort_keys=True)
                fh.write('\n')

    results, gates = [], {}
    for gate, name, argv in CHECKS:
        out = os.path.join(ev, 'runs', f'{name}.json')
        started = time.time()
        r = subprocess.run([args.python] + argv + ['--json', out],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=14400)
        elapsed = time.time() - started
        with open(os.path.join(ev, 'runs', f'{name}.log'), 'w') as fh:
            fh.write(r.stdout + r.stderr)
        cases = json.load(open(out)) if os.path.exists(out) else None
        if cases is None:
            print(f"  {gate} {name:<8s} NO RESULT -- the checker produced "
                  f"no json; it may not exist yet")
            gates.setdefault(gate, []).append(False)
            results.append({'gate': gate, 'checker': name, 'exit': r.returncode,
                            'elapsed_s': round(elapsed, 2), 'cases': None})
            continue
        bad = [c['id'] for c in cases if not c['passed']]
        gates.setdefault(gate, []).append(r.returncode == 0)
        print(f"  {gate} {name:<8s} exit={r.returncode} "
              f"{len(cases) - len(bad)}/{len(cases)} case(s) "
              f"in {elapsed:>6.1f}s"
              + (f"   open: {', '.join(bad)}" if bad else ''))
        results.append({'gate': gate, 'checker': name,
                        'exit': r.returncode,
                        'elapsed_s': round(elapsed, 2),
                        'cases': cases})

        # File the same result under the section 6 area it measures.
        if name in AREA_OF:
            area, fname = AREA_OF[name]
            shutil.copy2(out, os.path.join(ev, area, fname))
            shutil.copy2(os.path.join(ev, 'runs', f'{name}.log'),
                         os.path.join(ev, area,
                                      fname.replace('.json', '.log')))
        for c in cases:
            if c['id'] in CASE_AREA:
                area, fname = CASE_AREA[c['id']]
                with open(os.path.join(ev, area, fname), 'w') as fh:
                    json.dump(c, fh, indent=2, sort_keys=True)
                    fh.write('\n')

    every = all(all(v) for v in gates.values())
    verdict = SIGNOFF if (every and not dirty) else 'NOT ACCEPTED'
    reasons = []
    if dirty:
        reasons.append('the working tree is not clean')
    for gate, oks in sorted(gates.items()):
        if not all(oks):
            reasons.append(f"{gate} has an open checker")

    report = {
        'contract_id': CONTRACT_ID,
        'run_id': run_id,
        'commit': commit,
        'clean_tree': not dirty,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'tools': tool_versions(),
        'frozen_hashes': frozen,
        'checks': results,
        'gates': {g: all(v) for g, v in sorted(gates.items())},
        'verdict': verdict,
        'reasons': reasons,
        'note': ('This file records what each checker returned. It is not '
                 'the verdict: tools/r3v2/verify_evidence.py recomputes '
                 'that from the frozen inventory and the raw results, and '
                 'does not trust the field above.'),
    }
    with open(os.path.join(ev, 'acceptance.json'), 'w') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write('\n')

    latest = os.path.join(root_ev, 'latest')
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(run_id, latest)
    except OSError as e:
        print(f"  note: could not update {latest} ({e})")

    print()
    for r in reasons:
        print(f"  open: {r}")
    print(f"  verdict: {verdict}")
    print(f"  evidence: {ev}")
    print(f"  latest:   {latest} -> {run_id}")
    return 0 if verdict == SIGNOFF else 1


if __name__ == '__main__':
    sys.exit(main())
