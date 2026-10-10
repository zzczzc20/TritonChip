#!/usr/bin/env python3
"""R40 acceptance: every gate of the plan, with a machine-readable status.

    python3 tools/r40/acceptance.py [--plan spec/r40/plan.json] [--wp R40-00 ...]
        [--run] [--json sim/build/r40/acceptance.json]

doc/r40_tasks_and_acceptance.md R40-00 acceptance 3: each gate is one of

  capability-pass       the gate's command ran and its evidence says the
                        capability is there and correct
  perf-pass             a performance gate whose comparison tools/r40/ablate.py
                        accepted, with the frozen threshold met and the 95 %
                        interval above 1
  no-significant-gain   accepted, measured, and not enough
  fail                  the command failed, or the evidence says no, or the
                        comparison was refused
  not-measured          the evidence is not there yet

and nothing missing can pass: a gate whose command does not exist, whose
evidence file is absent, or whose evidence does not name the gate is not
PASS. A work package passes when every one of its required gates does; R40
when every package does.

Without --run the evidence already on disk is judged; with it, each gate's
command runs first (in the order of the plan) and its exit status is part of
the judgement.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUSES = ('capability-pass', 'perf-pass', 'no-significant-gain', 'fail', 'not-measured')
PASSING = ('capability-pass', 'perf-pass')


def make_targets():
    """The targets the Makefile defines, so a gate naming one that does not
    exist is caught rather than run."""
    out = set()
    for line in (ROOT / 'Makefile').read_text().splitlines():
        if line and not line[0].isspace() and ':' in line and not line.startswith(('#', '.')):
            head = line.split(':', 1)[0]
            if '=' in head or '$' in head:
                continue
            out.update(head.split())
    return out


def judge(gate, *, ran=None):
    """One gate's status and why."""
    cmd = gate.get('command')
    if not cmd:
        return 'fail', 'the gate names no command'
    if cmd.startswith('make '):
        target = cmd.split()[1]
        if target not in make_targets():
            return 'fail', f'make target {target} does not exist'
    if ran is not None and ran != 0:
        return 'fail', f'{cmd} exited {ran}'
    ev = gate.get('evidence')
    if not ev:
        return 'fail', 'the gate names no evidence'
    path = ROOT / ev
    if not path.exists():
        return 'not-measured', f'no evidence at {ev}'
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return 'fail', f'{ev}: {e}'
    entry = doc.get('gates', {}).get(gate['id']) if isinstance(doc, dict) else None
    if entry is None:
        return 'fail', f"{ev} does not name gate {gate['id']}"
    status = entry.get('status')
    if status not in STATUSES:
        return 'fail', f'{ev}: status {status!r} is not one of {STATUSES}'
    if gate['kind'] == 'capability' and status == 'perf-pass':
        return 'fail', 'a capability gate reported perf-pass'
    if gate['kind'] == 'perf' and status == 'capability-pass':
        return 'fail', 'a performance gate cannot pass as a capability'
    if status in PASSING and not entry.get('checks'):
        return 'fail', f'{ev}: {status} without the checks that support it'
    if status in PASSING and any(not c.get('ok') for c in entry['checks']):
        bad = [c.get('name') for c in entry['checks'] if not c.get('ok')]
        return 'fail', f'{ev}: {status} with failing checks {bad}'
    return status, entry.get('detail', '')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', type=Path, default=ROOT / 'spec' / 'r40' / 'plan.json')
    ap.add_argument('--wp', nargs='*')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    report = dict(schema='tnca-r40-acceptance/1', plan=str(args.plan.relative_to(ROOT)
                  if args.plan.is_absolute() and ROOT in args.plan.parents else args.plan),
                  work_packages={})
    all_ok = True
    for wp in plan['work_packages']:
        if args.wp and wp['id'] not in args.wp:
            continue
        rows = []
        for gate in wp['gates']:
            ran = None
            if args.run and gate.get('command'):
                ran = subprocess.run(gate['command'], shell=True, cwd=ROOT).returncode
            status, why = judge(gate, ran=ran)
            rows.append(dict(id=gate['id'], kind=gate['kind'], required=gate.get('required', True),
                             status=status, why=why))
        ok = all(r['status'] in PASSING for r in rows if r['required'])
        all_ok &= ok
        report['work_packages'][wp['id']] = dict(passed=ok, gates=rows)
        print(f"{wp['id']:<8s} {'PASS' if ok else 'not passed'}")
        for r in rows:
            print(f"    {r['id']:<34s} {r['kind']:<10s} {r['status']:<20s} {r['why']}")
    report['passed'] = all_ok
    evaluated = sorted(report['work_packages'])
    whole = len(evaluated) == len(plan['work_packages'])
    report['scope'] = 'all work packages' if whole else evaluated
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + '\n')
    if whole:
        print(f"\n{plan.get('signoff', 'R40')}: {'PASS' if all_ok else 'NOT PASSED'}")
    else:
        print(f"\n{', '.join(evaluated)}: {'PASS' if all_ok else 'NOT PASSED'} "
              f"(the sign-off {plan.get('signoff', 'R40')} is judged on every work package)")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
