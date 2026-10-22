#!/usr/bin/env python3
"""Summarize hbm8_case.py reports into the experiment's conclusion record.

    hbm8_summary.py --group NAME CASE=REPORT ... --out summary.json

The record carries experiment.py's conclusion fields (result, residual,
conclusion, next_step), each case's status and checks, and the addresses
and counters a reader needs to judge the window: highest device and physical
address reached, PCs covered, cores and outlets that carried traffic. It
states a result only from the reports; it never upgrades a missing report.
"""
import argparse
import json
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.is_file():
        return dict(status='missing', path=str(p))
    return json.loads(p.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', required=True)
    ap.add_argument('reports', nargs='+', help='CASE=report.json')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    cases = {}
    for item in a.reports:
        name, _, path = item.partition('=')
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
    ok = [n for n, c in cases.items() if c['status'] == 'functional-pass']
    bad = [n for n in cases if n not in ok]
    trees = {c['software_tree'] for c in cases.values() if c.get('software_tree')}
    record = dict(schema='tnca-r41-hbm8-summary/1', group=a.group, cases=cases,
                  passed=ok, not_passed=bad, software_trees=sorted(trees),
                  result=(f'{len(ok)}/{len(cases)} cases functional-pass' + (f'; not passed: {bad}' if bad else '')),
                  residual=('none' if not bad else 'see each not-passed case report (error, failed checks, preserved state)'),
                  conclusion=(f'{a.group}: every registered case passed on one software tree'
                              if not bad and len(trees) == 1 else
                              f'{a.group}: not qualified; first diagnose {bad[0] if bad else "mixed software trees"}'),
                  next_step=('record the board result in evidence/r41/hbm8/README.md and plan R41 FP8 models on the 8 GiB window'
                             if not bad else 'diagnose the failed case from its report before any rerun; do not relabel'),
                  performance_gate='not-measured')
    a.out.write_text(json.dumps(record, indent=1) + '\n')
    print(record['result'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
