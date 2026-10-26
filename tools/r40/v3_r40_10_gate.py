#!/usr/bin/env python3
"""R40 v3 record of R40-10.acceptance: the sign-off gate, from the other gates' evidence.

    R40_CONTRACT=v3 python3 tools/r40/v3_r40_10_gate.py [--dry-run] [--pr-url URL]

doc/r40_tasks_and_acceptance_v3.md (R40-10.acceptance) requires:
- every required v3 gate passes;
- doc/r40_report.md has per-gate evidence, the performance-target table,
  the moved-out items with their reasons, resources/timing/CDC and the 90 MHz
  exception;
- the final PR is non-Draft and targets main;
- v1 R40-10 items 4 (GPU comparison) and 5 (independent reproduction from a
  clean environment) are kept.

Each item is judged from preserved evidence: acceptance.judge on every other
gate, the report's sections, the GPU comparison report and the reproduction
run, all bound to the delivery identity. The gate passes only if every check
does. It never passes on a deferred or not-measured required gate.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'tools/r40')]
import acceptance
import contract_v3
import evidence

PLAN_PATH = ROOT / 'doc/r40_v3/plan.json'
EVIDENCE = ROOT / 'evidence/r40/v3'
REPORT = ROOT / 'doc/r40_report.md'
SECTIONS = ('Gate results', 'Performance against the v2 targets', 'Bottlenecks', 'Frequency exception',
            'Resources', 'Contract changes', 'GPU comparison', 'Reproduction', 'Carried into R41')


def check(name, ok, **extra):
    return dict(name=name, ok=bool(ok), **extra)


def gates(plan):
    rows = {}
    for wp in plan['work_packages']:
        for gate in wp['gates']:
            if gate['id'] == 'R40-10.acceptance':
                continue
            status, detail = acceptance.judge(gate, plan=plan)
            rows[gate['id']] = dict(status=status, kind=gate['kind'], required=gate.get('required', True),
                                    detail=detail[:300])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--pr-url', help='the final PR (checked with gh: non-Draft, base main)')
    a = ap.parse_args()
    os.environ['R40_CONTRACT'] = 'v3'
    contract_v3.verify()
    plan = json.loads(PLAN_PATH.read_text())
    ident = contract_v3.delivery(plan)
    checks = []

    rows = gates(plan)
    passing = {'capability-pass', 'perf-pass'}
    open_rows = {g: r['status'] for g, r in rows.items() if r['required'] and r['status'] not in passing}
    moved = {g: r['status'] for g, r in rows.items() if not r['required']}
    checks.append(check('every required v3 gate passes', not open_rows, not_passing=open_rows,
                        not_required=moved, gates={g: r['status'] for g, r in rows.items()}))

    text = REPORT.read_text() if REPORT.exists() else ''
    heads = re.findall(r'^#{2,3} .*$', text, re.M)
    missing = [s for s in SECTIONS if not any(s.lower() in h.lower() for h in heads)]
    named = [g for g in rows if g not in text]
    checks.append(check('doc/r40_report.md: per-gate table, target table, bottlenecks, moved-out items, '
                        '90 MHz exception, resources/timing/CDC, GPU comparison, reproduction',
                        text and not missing and not named and ident['software_tree'] in text,
                        missing_sections=missing, gates_not_named=named))

    gpu = EVIDENCE / 'gpu_comparison/report.json'
    g = json.loads(gpu.read_text()) if gpu.exists() else {}
    trees = set(json.dumps(g.get('identities', {})).split('"')) & {ident['software_tree']}
    checks.append(check('GPU comparison (v1 R40-10 item 4) with TNCA records on the delivery tree',
                        g.get('schema') == 'r40-v3-gpu-comparison/1' and trees, report=str(gpu.relative_to(ROOT))))

    rep = sorted((EVIDENCE / 'reproduction').glob('run_*/compare.log'))
    chk = sorted((EVIDENCE / 'reproduction').glob('run_*/checks.log'))
    ok = bool(rep and chk) and rep[-1].read_text().strip().endswith('REPRODUCTION PASS') and \
        f"software_tree {ident['software_tree']}" in chk[-1].read_text() and \
        'R40 v3 immutable contract: PASS' in chk[-1].read_text()
    checks.append(check('independent reproduction from a clean archive (v1 R40-10 item 5) on the delivery tree',
                        ok, run=str(rep[-1].parent.relative_to(ROOT)) if rep else None))

    pr = None
    if a.pr_url:
        out = subprocess.run(['gh', 'pr', 'view', a.pr_url, '--json', 'isDraft,baseRefName,url,state'],
                             capture_output=True, text=True)
        pr = json.loads(out.stdout) if out.returncode == 0 else dict(error=out.stderr.strip())
    checks.append(check('final PR non-Draft against main', pr is not None and pr.get('isDraft') is False
                        and pr.get('baseRefName') == 'main', pr=pr))

    status = 'capability-pass' if all(c['ok'] for c in checks) else 'fail'
    detail = ('every item holds' if status == 'capability-pass' else
              'not passed: ' + '; '.join(c['name'] for c in checks if not c['ok']))
    if a.dry_run:
        print(json.dumps(dict(status=status, checks=checks), indent=1, default=str))
    else:
        evidence.record('R40-10.acceptance', status, checks, detail=detail, identity=ident)
    print('R40-10.acceptance:', status, '-', detail)
    return 0


if __name__ == '__main__':
    sys.exit(main())
