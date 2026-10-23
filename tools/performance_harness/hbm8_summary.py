#!/usr/bin/env python3
"""Summarize hbm8_case.py reports into the experiment's conclusion record.

    hbm8_summary.py --group NAME CASE=REPORT ... --out summary.json

The record carries experiment.py's conclusion fields (result, residual,
conclusion, next_step), each case's status and checks, and the addresses
and counters for the cores and outlets that carried traffic. Full addresses
and PC coverage remain in the individual case reports. It
states a result only from the reports; it never upgrades a missing report.
"""
import argparse
import json
from pathlib import Path
from hbm8_manifest import GROUPS


def load(path):
    p = Path(path)
    if not p.is_file():
        return dict(status='missing', path=str(p))
    return json.loads(p.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', required=True, choices=GROUPS)
    ap.add_argument('reports', nargs='+', help='CASE=report.json')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    cases = {}
    for item in a.reports:
        name, _, path = item.partition('=')
        if name in cases:
            ap.error(f'duplicate case: {name}')
        r = load(path)
        checks = r.get('checks') or []
        launches = r.get('launches') or []
        cases[name] = dict(
            status=r.get('status'), mode=r.get('mode'), backend=r.get('backend'),
            software_tree=r.get('software_tree'), window=r.get('window'),
            checks_passed=sum(1 for c in checks if c.get('ok')), checks_failed=[c['name'] for c in checks if not c.get('ok')],
            launches=len(launches),
            cores_all_three=all(all(c['issued'] > 0 for c in row['last_launch_cores']) for row in launches) if launches else None,
            outlet_lines=[sum(row['outlets'][k]['read_lines'] + row['outlets'][k]['write_lines'] for row in launches)
                          for k in range(3)] if launches else None,
            axi_status_at_end=r.get('axi_status_at_end'),
            error=(r.get('error') or '').strip().splitlines()[-1:] or None)
    ok = [n for n, c in cases.items() if c['status'] == 'functional-pass' and
          not c['checks_failed'] and c['axi_status_at_end'] == dict(error=False, stray=False)]
    bad = [n for n in cases if n not in ok]
    trees = {c['software_tree'] for c in cases.values() if c.get('software_tree')}
    modes = {c['mode'] for c in cases.values()}
    problems = []
    if bad:
        problems.append(f'not passed: {bad}')
    if set(cases) != set(GROUPS[a.group]):
        problems.append('case set differs from registered manifest')
    if len(trees) != 1 or any(not c['software_tree'] for c in cases.values()):
        problems.append('missing or mixed software trees')
    if len(modes) != 1 or not modes <= {'sim', 'board'}:
        problems.append('missing or mixed execution modes')
    qualified = not problems
    mode = next(iter(modes)) if len(modes) == 1 else 'mixed'
    record = dict(schema='tnca-r41-hbm8-summary/1', group=a.group, cases=cases,
                  passed=ok, not_passed=bad, software_trees=sorted(trees),
                  mode=mode, qualified=qualified, qualification_problems=problems,
                  result=(f'{len(ok)}/{len(cases)} cases functional-pass' + (f'; not passed: {bad}' if bad else '')),
                  residual=('none' if qualified else '; '.join(problems)),
                  conclusion=(f'{a.group}: every registered case passed in {mode} on one software tree'
                              if qualified else f'{a.group}: not qualified; ' + '; '.join(problems)),
                  next_step=(('run the waiting board experiment; RTL simulation does not qualify physical PC routing'
                              if mode == 'sim' else
                              'record the board result in evidence/r41/hbm8/README.md')
                             if qualified else 'diagnose qualification problems before any rerun; do not relabel'),
                  performance_gate='not-measured')
    a.out.write_text(json.dumps(record, indent=1) + '\n')
    print(record['result'])
    return 0 if qualified else 1


if __name__ == '__main__':
    raise SystemExit(main())
