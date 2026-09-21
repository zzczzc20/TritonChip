#!/usr/bin/env python3
"""
The Phase 3.5 gate -- phase3_5_tasks_and_acceptance.md section 8.

Runs every checker once into one run directory and writes the evidence
the verifier reads back. A sub-target passing is not the stage passing:
this is the only thing that produces a verdict, and a checker that does
not exist yet is a non-zero exit rather than a silence.

It does not decide the verdict from its own opinion of how things went.
It records what each checker returned; tools/r35/verify_evidence.py
recomputes the verdict from the frozen inventory and the raw results and
does not read the fields below. Section 8: "研发重跑保留旧失败并产生新
run ID，禁止覆盖记录".
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

CONTRACT_ID = 'r35-compiler'
SIGNOFF = 'R35_COMPILER_ACCEPTED'
SHORTFALL = 'OPTIMIZATION_DELIVERED / PERFORMANCE_NOT_MET'

# Every checker, the package it answers for, and where its result lands.
CHECKS = [
    ('R35-01', 'plan',     ['tools/r35/plan.py']),
    ('R35-02', 'upstream', ['tools/r35/upstream_tests.py']),
    ('R35-03', 'layout',   ['tools/r35/layout_tests.py']),
    ('R35-04', 'opt',      ['tools/r35/opt_tests.py']),
    ('R35-05', 'alloc',    ['tools/r35/alloc_tests.py']),
    ('R35-06', 'safety',   ['tools/r35/safety_tests.py']),
    ('R35-07', 'perf',     ['tools/r35/perf.py',
                            '--modes', 'B0', 'O2', 'O2-no-LL']),
    ('R35-08', 'package',  ['tools/r35/package_tests.py']),
    ('R35-09', 'release',  ['tools/r35/release_tests.py']),
]

# Section 8's evidence layout, and which checker fills each area.
AREA_OF = {
    'plan':     ('inventory', 'plan.json'),
    'layout':   ('ir',        'layout.json'),
    'opt':      ('ir',        'optimisation.json'),
    'upstream': ('build',     'upstream.json'),
    'perf':     ('performance', 'perf.json'),
    'safety':   ('mutations', 'safety.json'),
    'release':  ('inherited', 'release.json'),
    'package':  ('code',      'package.json'),
    'alloc':    ('code',      'allocation.json'),
}


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
                            'import triton,sys;'
                            'print(triton.__version__, triton.__file__)'],
                           capture_output=True, text=True, timeout=120)
        out['triton'] = r.stdout.strip() or r.stderr.strip()[-120:]
    except Exception as e:                                 # noqa: BLE001
        out['triton'] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain')
    run_id = f"{int(time.time())}-{commit[:12] or 'nogit'}"
    root_ev = os.path.abspath(args.evidence)
    ev = os.path.join(root_ev, run_id)
    for sub in ('contracts', 'inventory', 'build', 'ir', 'code', 'runs',
                'performance', 'mutations', 'inherited'):
        os.makedirs(os.path.join(ev, sub), exist_ok=True)

    print(f"Phase 3.5 gate: {CONTRACT_ID}")
    print(f"  run {run_id}")
    print(f"  commit {commit[:12] or 'unknown'}"
          f"{' (WORKING TREE DIRTY)' if dirty else ' (clean tree)'}")

    frozen = {}
    for name in ('r35_workloads.json', 'r35_required_cases.json',
                 'r35_toolchain.json'):
        src = os.path.join(ROOT, 'spec', name)
        shutil.copy2(src, os.path.join(ev, 'contracts', name))
        frozen[name] = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    shutil.copy2(os.path.join(ROOT, 'doc',
                              'phase3_5_tasks_and_acceptance.md'),
                 os.path.join(ev, 'contracts', 'contract.md'))
    # The R3 v2 sign-off this stage stands on. Section 5: the gains are
    # measured on hardware that was accepted, and the record of that
    # acceptance belongs with them.
    r3 = os.path.join(ROOT, 'doc', 'evidence', 'r3v2',
                      'acceptance-linux.json')
    if os.path.exists(r3):
        shutil.copy2(r3, os.path.join(ev, 'inherited', 'r3v2-acceptance.json'))

    env = dict(os.environ)
    env['TRITON_PYTHON'] = args.python
    with open(os.path.join(ev, 'build', 'build.json'), 'w') as fh:
        json.dump({'commit': commit, 'clean_tree': not dirty,
                   'run_id': run_id, 'platform': platform.platform(),
                   'machine': platform.machine(),
                   'tools': tool_versions(args.python)},
                  fh, indent=2, sort_keys=True)
        fh.write('\n')

    inv = json.load(open(os.path.join(ROOT, 'spec', 'r35_workloads.json')))
    for key in ('workloads', 'matrix', 'budget', 'gain_thresholds',
                'hardware_precondition'):
        with open(os.path.join(ev, 'inventory', f'{key}.json'), 'w') as fh:
            json.dump(inv[key], fh, indent=2, sort_keys=True)
            fh.write('\n')

    results, packages = [], {}
    for pkg, name, argv in CHECKS:
        out = os.path.join(ev, 'runs', f'{name}.json')
        started = time.time()
        if not os.path.exists(os.path.join(ROOT, argv[0])):
            print(f"  {pkg} {name:<9s} NOT IMPLEMENTED -- {argv[0]} does "
                  f"not exist", flush=True)
            packages.setdefault(pkg, []).append(False)
            results.append({'package': pkg, 'checker': name, 'exit': None,
                            'elapsed_s': 0.0, 'cases': None,
                            'why': 'the checker does not exist yet'})
            continue
        r = subprocess.run([args.python] + argv + ['--json', out],
                           cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=14400)
        elapsed = time.time() - started
        with open(os.path.join(ev, 'runs', f'{name}.log'), 'w') as fh:
            fh.write(r.stdout + r.stderr)
        cases = json.load(open(out)) if os.path.exists(out) else None
        if cases is None:
            print(f"  {pkg} {name:<9s} NO RESULT -- it produced no json",
                  flush=True)
            packages.setdefault(pkg, []).append(False)
            results.append({'package': pkg, 'checker': name,
                            'exit': r.returncode,
                            'elapsed_s': round(elapsed, 2), 'cases': None})
            continue
        bad = [c['id'] for c in cases if not c['passed']]
        packages.setdefault(pkg, []).append(r.returncode == 0)
        print(f"  {pkg} {name:<9s} exit={r.returncode} "
              f"{len(cases) - len(bad)}/{len(cases)} case(s) in "
              f"{elapsed:>6.1f}s"
              + (f"   open: {', '.join(bad)}" if bad else ''), flush=True)
        results.append({'package': pkg, 'checker': name,
                        'exit': r.returncode,
                        'elapsed_s': round(elapsed, 2), 'cases': cases})
        if name in AREA_OF:
            area, fname = AREA_OF[name]
            shutil.copy2(out, os.path.join(ev, area, fname))
            shutil.copy2(os.path.join(ev, 'runs', f'{name}.log'),
                         os.path.join(ev, area,
                                      fname.replace('.json', '.log')))

    every = all(all(v) for v in packages.values())
    # Section 6.3: a run that measured honestly and fell short is not the
    # same as a run that did not happen. The gain gate has its own name.
    perf = next((e for e in results if e['checker'] == 'perf'), None)
    perf_ran = bool(perf and perf.get('cases'))
    perf_met = bool(perf and perf['exit'] == 0)
    verdict = SIGNOFF if (every and not dirty) else (
        SHORTFALL if (perf_ran and not perf_met) else 'NOT ACCEPTED')

    reasons = []
    if dirty:
        reasons.append('the working tree is not clean')
    for pkg, oks in sorted(packages.items()):
        if not all(oks):
            reasons.append(f"{pkg} has an open checker")

    report = {
        'contract_id': CONTRACT_ID, 'run_id': run_id, 'commit': commit,
        'clean_tree': not dirty, 'platform': platform.platform(),
        'machine': platform.machine(), 'tools': tool_versions(args.python),
        'frozen_hashes': frozen, 'checks': results,
        'packages': {p: all(v) for p, v in sorted(packages.items())},
        'verdict': verdict, 'reasons': reasons,
        'note': ('This file records what each checker returned. It is not '
                 'the verdict: tools/r35/verify_evidence.py recomputes '
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
    return 0 if verdict == SIGNOFF else 1


if __name__ == '__main__':
    sys.exit(main())
