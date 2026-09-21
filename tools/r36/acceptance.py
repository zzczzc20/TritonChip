#!/usr/bin/env python3
"""
The R3.6 gate -- phase3_6_tasks_and_acceptance.md sections 6 and 7.

Runs every checker, collects the evidence, and computes the verdict from
what the checkers returned rather than from a summary anybody wrote.

The sign-off this gate can produce is R36_ENGINEERING_READY and nothing
more. Section 1 and R36-REL-01 both say it: FPGA and ASIC physical
verification are an external follow-on, `platform-verified` is a
reserved state, and there is no combination of results here that turns
into a claim about a board.

Three rules carried from the earlier gates, each for a reason:

  A checker that does not exist is NOT a pass. It is reported as NOT
  IMPLEMENTED and its package fails.

  A dirty working tree cannot be signed off. Evidence that describes a
  tree nobody can check out describes nothing.

  Every artefact is bound to one commit. R36-EV-02 asks for the packages
  to be traceable to a single integration candidate, and a run whose
  pieces came from different trees is the thing that prevents.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTRACT_ID = 'r36-engineering'
SIGNOFF = 'R36_ENGINEERING_READY'

# package -> (name, argv). Ten packages, thirty standards.
CHECKS = [
    ('R36-01', 'plan',       ['tools/r36/plan.py']),
    ('R36-02', 'backend',    ['tools/r36/backend_tests.py']),
    ('R36-03', 'config',     ['tools/r36/config_tests.py']),
    ('R36-04', 'rtl',        ['tools/r36/rtl_tests.py']),
    ('R36-05', 'memory',     ['tools/r36/mem_tests.py']),
    ('R36-06', 'io',         ['tools/r36/io_tests.py']),
    ('R36-07', 'build',      ['tools/r36/build_tests.py']),
    ('R36-08', 'conformance', ['tools/r36/conformance.py']),
    ('R36-08', 'mutation',   ['tools/r36/mutation_tests.py']),
    ('R36-08', 'regression', ['tools/r36/regression.py']),
    ('R36-09', 'handoff',    ['tools/r36/handoff_tests.py']),
]

AREA_OF = {
    'plan':        ('inventory', 'plan.json'),
    'backend':     ('interfaces', 'backend.json'),
    'config':      ('interfaces', 'config.json'),
    'rtl':         ('hardware', 'rtl.json'),
    'memory':      ('hardware', 'memory.json'),
    'io':          ('interfaces', 'io.json'),
    'build':       ('packages', 'build.json'),
    'conformance': ('conformance', 'conformance.json'),
    'mutation':    ('mutations', 'mutation.json'),
    'regression':  ('inherited', 'regression.json'),
    'handoff':     ('handoff', 'handoff.json'),
}

FROZEN = ('r36_capabilities.json', 'r36_required_cases.json',
          'r36_memories.json', 'r36_provenance.json',
          'r36_regression.json', 'r36_baseline_delta.json')


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def tool_versions(python):
    out = {'python': sys.version.split()[0]}
    for name, argv in (('iverilog', ['iverilog', '-V']),
                       ('yosys', ['yosys', '-V'])):
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=60)
            out[name] = (r.stdout or r.stderr).splitlines()[0].strip()
        except Exception as e:                             # noqa: BLE001
            out[name] = f"unavailable: {e}"
    try:
        r = subprocess.run([python, '-c',
                            'import triton;'
                            'print(triton.__version__, triton.__file__)'],
                           capture_output=True, text=True, timeout=120)
        out['triton'] = r.stdout.strip() or r.stderr.strip()[-120:]
    except Exception as e:                                 # noqa: BLE001
        out['triton'] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--python', default=os.environ.get('TRITON_PYTHON',
                                                       sys.executable))
    ap.add_argument('--only', nargs='*',
                    help='run only these checkers; the verdict is then '
                         'PARTIAL and cannot be a sign-off')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain')
    run_id = f"{int(time.time())}-{commit[:12] or 'nogit'}"
    ev = os.path.join(os.path.abspath(args.evidence), run_id)
    for sub in ('contracts', 'inventory', 'interfaces', 'hardware',
                'packages', 'conformance', 'mutations', 'inherited',
                'handoff', 'runs'):
        os.makedirs(os.path.join(ev, sub), exist_ok=True)

    print(f"Phase 3.6 gate: {CONTRACT_ID}")
    print(f"  run {run_id}")
    print(f"  commit {commit[:12] or 'unknown'}"
          f"{' (WORKING TREE DIRTY)' if dirty else ' (clean tree)'}")
    print(f"  the only sign-off this gate can produce is {SIGNOFF}; "
          f"platform verification is an external follow-on")

    frozen = {}
    for name in FROZEN:
        src = os.path.join(ROOT, 'spec', name)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(ev, 'contracts', name))
        frozen[name] = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    for src, dst in (
            (os.path.join(ROOT, 'doc',
                          'phase3_6_tasks_and_acceptance.md'),
             'contract.md'),
            (os.path.join(ROOT, 'doc', 'r36_handoff.md'),
             'handoff.md')):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ev, 'contracts', dst))

    env = dict(os.environ)
    env['TRITON_PYTHON'] = args.python

    # The deliverables, built once and bound to this commit. R36-EV-02:
    # every package traceable to one integration candidate.
    print("  building the deliverables...", flush=True)
    started = time.time()
    b = subprocess.run([sys.executable, 'tools/r36/build.py', '--what',
                        'all', '--out', os.path.join(ev, 'packages',
                                                     'tnca-export'),
                        '--json', os.path.join(ev, 'packages',
                                               'build-report.json')],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=7200)
    with open(os.path.join(ev, 'packages', 'build.log'), 'w') as fh:
        fh.write(b.stdout + b.stderr)
    print(f"  {'built' if b.returncode == 0 else 'BUILD FAILED'} in "
          f"{time.time() - started:.0f}s", flush=True)
    for ln in b.stdout.strip().splitlines():
        print(f"    {ln.strip()[:92]}", flush=True)

    results, packages = [], {}
    for pkg, name, argv in CHECKS:
        if args.only and name not in args.only:
            continue
        out = os.path.join(ev, 'runs', f'{name}.json')
        started = time.time()
        if not os.path.exists(os.path.join(ROOT, argv[0])):
            print(f"  {pkg} {name:<12s} NOT IMPLEMENTED -- {argv[0]} "
                  f"does not exist", flush=True)
            packages.setdefault(pkg, []).append(False)
            results.append({'package': pkg, 'checker': name, 'exit': None,
                            'elapsed_s': 0.0, 'cases': None,
                            'why': 'the checker does not exist yet'})
            continue
        r = subprocess.run([args.python] + argv + ['--json', out],
                           cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=21600)
        elapsed = time.time() - started
        with open(os.path.join(ev, 'runs', f'{name}.log'), 'w') as fh:
            fh.write(r.stdout + r.stderr)
        cases = json.load(open(out)) if os.path.exists(out) else None
        packages.setdefault(pkg, []).append(r.returncode == 0)
        if isinstance(cases, list):
            bad = [c['id'] for c in cases if not c.get('passed')]
            shape = (f"{len(cases) - len(bad)}/{len(cases)} case(s)"
                     + (f"   open: {', '.join(bad)}" if bad else ''))
        elif cases is None:
            shape = 'NO RESULT -- it produced no json'
        else:
            shape = 'a report, not a case list'
        print(f"  {pkg} {name:<12s} exit={r.returncode} {shape} in "
              f"{elapsed:>6.1f}s", flush=True)
        results.append({'package': pkg, 'checker': name,
                        'exit': r.returncode,
                        'elapsed_s': round(elapsed, 2), 'cases': cases})
        if name in AREA_OF and os.path.exists(out):
            area, fname = AREA_OF[name]
            shutil.copy2(out, os.path.join(ev, area, fname))
            shutil.copy2(os.path.join(ev, 'runs', f'{name}.log'),
                         os.path.join(ev, area,
                                      fname.replace('.json', '.log')))

    every = (all(all(v) for v in packages.values())
             and b.returncode == 0
             and len(packages) == len({p for p, _n, _a in CHECKS}))
    partial = bool(args.only)
    verdict = (SIGNOFF if (every and not dirty and not partial)
               else ('PARTIAL' if partial else 'NOT ACCEPTED'))

    reasons = []
    if partial:
        reasons.append('--only was given: this run did not attempt every '
                       'package and cannot be a sign-off')
    if b.returncode != 0:
        reasons.append('the deliverables did not build')
    if dirty:
        reasons.append('the working tree is not clean')
    for pkg, oks in sorted(packages.items()):
        if not all(oks):
            reasons.append(f"{pkg} has an open checker")
    missing = sorted({p for p, _n, _a in CHECKS} - set(packages))
    if missing and not partial:
        reasons.append(f"no result for {missing}")

    report = {
        'contract_id': CONTRACT_ID, 'run_id': run_id, 'commit': commit,
        'clean_tree': not dirty, 'platform': platform.platform(),
        'machine': platform.machine(),
        'tools': tool_versions(args.python), 'frozen_hashes': frozen,
        'checks': results,
        'packages': {p: all(v) for p, v in sorted(packages.items())},
        'verdict': verdict, 'reasons': reasons,
        'claim_level': 'engineering-ready',
        'not_claimed': ('platform-verified. No device has run this. '
                        'Section 1 of the contract puts FPGA and ASIC '
                        'physical verification in a separate branch.'),
        'note': ('This file records what each checker returned. It is '
                 'not a judgement written by hand: '
                 'tools/r36/verify_evidence.py recomputes the verdict '
                 'from the raw exit codes and outputs in runs/.'),
    }
    with open(os.path.join(ev, 'acceptance.json'), 'w') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write('\n')

    print()
    for r in reasons:
        print(f"  open: {r}")
    print(f"  verdict: {verdict}")
    print(f"  evidence: {ev}")
    return 0 if verdict == SIGNOFF else 1


if __name__ == '__main__':
    sys.exit(main())
