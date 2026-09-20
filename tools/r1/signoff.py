#!/usr/bin/env python3
"""
Check the sign-off table -- doc/phase1_tasks_and_acceptance.md R1-12.

    "签收表 12 项全为 PASS，所有证据由同一候选 commit 的机器验证器核验通过"

A sign-off table is a set of claims, and the one failure mode that matters is
a row that says PASS while the run says otherwise. So this checks two things
and refuses to do the third:

  structure   all twelve work packages present, every column filled, no TBD,
              no TODO, no "n/a", every row naming commits, cases and evidence
  agreement   given an evidence directory, each row's status against what
              that run actually recorded for the cases belonging to it

It does NOT decide the verdict. `acceptance.json` does, and
`verify_evidence.py` re-derives that from the archive. This is the check that
the prose agrees with them.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import inventory as INV  # noqa: E402

SIGNOFF = os.path.join(ROOT, 'doc', 'r1_signoff.md')
PACKAGES = [f"R1-{i:02d}" for i in range(1, 13)]
ALLOWED = ('PASS', 'PARTIAL', 'NOT RUN')
FORBIDDEN = re.compile(r'\b(TBD|TODO|FIXME|n/?a|not applicable|待定)\b', re.I)

ROW = re.compile(r'^\|\s*(R1-\d\d)\s+([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|'
                 r'([^|]*)\|\s*([^|]*?)\s*\|\s*$')


def parse(path):
    rows = {}
    for line in open(path):
        m = ROW.match(line)
        if m:
            wp = m.group(1)
            rows[wp] = dict(
                title=m.group(2).strip(), owner=m.group(3).strip(),
                commits=m.group(4).strip(), cases=m.group(5).strip(),
                evidence=m.group(6).strip(), status=m.group(7).strip())
    return rows


def structure(rows):
    problems = []
    for wp in PACKAGES:
        if wp not in rows:
            problems.append(f"{wp} has no row in the sign-off table")
            continue
        r = rows[wp]
        for field in ('title', 'owner', 'commits', 'cases', 'evidence',
                      'status'):
            if not r[field]:
                problems.append(f"{wp}: {field} is empty")
            elif FORBIDDEN.search(r[field]):
                problems.append(
                    f"{wp}: {field} says {r[field]!r}; the contract forbids "
                    "TBD, TODO and 'not applicable' in this table")
        if r['status'] not in ALLOWED:
            problems.append(
                f"{wp}: status {r['status']!r} is not one of "
                f"{', '.join(ALLOWED)}")
    extra = sorted(set(rows) - set(PACKAGES))
    if extra:
        problems.append(f"unknown work packages in the table: {extra}")
    return problems


def agreement(rows, run_dir):
    """Each row's status against what the run recorded for its cases."""
    problems = []
    notes = []
    res = os.path.join(run_dir, 'results.json')
    if not os.path.exists(res):
        return [f"no results.json in {run_dir}"], notes
    with open(res) as fh:
        statuses = json.load(fh).get('statuses', {})

    spec = INV.load_spec()
    by_wp = {}
    for c in spec['cases']:
        by_wp.setdefault(c['work_package'], []).append(c['id'])

    for wp in PACKAGES:
        ids = by_wp.get(wp, [])
        if not ids:
            notes.append(f"{wp}: no cases of its own; its evidence is the "
                         f"artefact named in the table")
            continue
        bad = sorted(i for i in ids
                     if statuses.get(i, 'NOT RUN') in INV.NOT_A_PASS)
        claimed = rows.get(wp, {}).get('status')
        if bad and claimed == 'PASS':
            problems.append(
                f"{wp} is signed off PASS, and this run has "
                f"{len(bad)} of its {len(ids)} case(s) not passing: "
                + ', '.join(bad[:4]) + (' ...' if len(bad) > 4 else ''))
        elif not bad and claimed != 'PASS':
            notes.append(f"{wp}: every case passed in this run and the table "
                         f"says {claimed} -- allowed, since a package can be "
                         f"partial for something no case covers")
        else:
            notes.append(f"{wp}: {len(ids) - len(bad)}/{len(ids)} case(s) "
                         f"pass; table says {claimed}")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', help='cross-check against this run')
    args = ap.parse_args()

    if not os.path.exists(SIGNOFF):
        print(f"  r1-signoff             FAIL   {SIGNOFF} does not exist")
        return 1
    rows = parse(SIGNOFF)
    problems = structure(rows)
    notes = []
    if args.evidence:
        p2, notes = agreement(rows, args.evidence)
        problems += p2

    for wp in PACKAGES:
        r = rows.get(wp)
        if r:
            print(f"    {wp}  {r['status']:<8s} {r['title'][:52]}")
    for n in notes:
        print('    ' + n)

    if problems:
        print(f"  r1-signoff             FAIL   {len(problems)} problem(s)")
        for p in problems[:12]:
            print('    ' + p)
        return 1
    npass = sum(1 for wp in PACKAGES if rows[wp]['status'] == 'PASS')
    print(f"  r1-signoff             PASS   table is complete and consistent "
          f"({npass}/12 PASS)")
    if npass < len(PACKAGES):
        print("    NOT a sign-off: the contract requires all twelve PASS "
              "before R1 is accepted.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
