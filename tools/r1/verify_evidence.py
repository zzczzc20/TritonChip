#!/usr/bin/env python3
"""
Re-verify an evidence directory -- doc/phase1_tasks_and_acceptance.md R1-10.

The gate writes `acceptance.json` saying what it concluded. This does not
believe it. It recomputes every hash, re-derives the required/discovered/
executed relation from the frozen inventory, and checks that everything in the
directory belongs to the same commit and the same run.

That distinction is the point of the work package: a CI job that reports
success because a file said PASS has verified nothing. The contract lists six
specific tamperings this must catch, and tools/r1/selftest_evidence.py injects
each one.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inventory as INV  # noqa: E402

REQUIRED_FILES = ('run.json', 'inventory.json', 'results.json',
                  'acceptance.json', 'checksums.sha256')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(run_dir):
    problems = []

    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(run_dir, f)):
            problems.append(f"missing evidence file: {f}")
    if problems:
        return problems

    def load(n):
        with open(os.path.join(run_dir, n)) as fh:
            return json.load(fh)

    run, rec, res, acc = (load('run.json'), load('inventory.json'),
                          load('results.json'), load('acceptance.json'))

    # --- integrity: recompute every hash, and notice extra files too -------
    listed = {}
    for line in open(os.path.join(run_dir, 'checksums.sha256')):
        line = line.strip()
        if line:
            digest, rel = line.split('  ', 1)
            listed[rel] = digest
    on_disk = set()
    for base, _d, files in os.walk(run_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(base, f), run_dir)
            if rel == 'checksums.sha256':
                continue
            on_disk.add(rel)
    for rel in sorted(on_disk - set(listed)):
        problems.append(f"file not in the manifest: {rel}")
    for rel, digest in sorted(listed.items()):
        p = os.path.join(run_dir, rel)
        if not os.path.exists(p):
            problems.append(f"manifest lists a missing file: {rel}")
        elif sha256(p) != digest:
            problems.append(f"content changed since the run: {rel}")

    # --- provenance: one commit, one run ----------------------------------
    git = run.get('git') or {}
    if not git.get('commit'):
        problems.append('run.json records no commit')
    if not git.get('clean'):
        problems.append(f"source tree was dirty: {git.get('dirty_paths')}")
    if 'started' not in run:
        problems.append('run.json records no start time')
    if 'seconds' not in acc:
        problems.append('acceptance.json records no end state')

    # --- the verdict, re-derived rather than trusted ----------------------
    required = INV.required_ids()
    statuses = res.get('statuses', {})
    redone = INV.reconcile(required, rec.get('discovered', []), statuses)
    if set(redone['required']) != set(rec.get('required', [])):
        problems.append('the run used a different required set than the spec')
    if redone['problems'] and acc.get('verdict') == 'ACCEPTED':
        problems.append('acceptance.json says ACCEPTED but the sets disagree')
    if not redone['problems'] and acc.get('verdict') != 'ACCEPTED':
        # Not an error -- the gate may have failed for a reason outside the
        # sets -- but the stated problems must exist.
        if not acc.get('problems'):
            problems.append('NOT ACCEPTED with no recorded reason')

    counted = sum(1 for v in statuses.values() if v == 'PASS')
    if acc.get('passed') != counted:
        problems.append(f"acceptance.json claims {acc.get('passed')} passes, "
                        f"results.json has {counted}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evidence')
    args = ap.parse_args()
    if not os.path.isdir(args.evidence):
        print(f"  r1-verify-evidence     FAIL   no such directory: {args.evidence}")
        return 2
    problems = verify(args.evidence)
    if problems:
        print(f"  r1-verify-evidence     FAIL   {len(problems)} problem(s)")
        for p in problems[:20]:
            print('    ' + p)
        return 1
    print("  r1-verify-evidence     PASS   evidence is internally consistent")
    print("    (consistent does not mean ACCEPTED; read acceptance.json)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
