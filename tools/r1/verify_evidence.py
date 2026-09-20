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

    # --- the archived configuration has to describe the archived run ------
    #
    # `profiles/` exists so that "this run was on THAT machine" survives the
    # run. Copying one file out of another candidate's evidence -- a manifest
    # from a different profile or a different commit -- passes every hash
    # check, because the hashes were recomputed along with it. So the
    # manifests are RE-DERIVED from the profile spec archived beside them,
    # and a manifest that does not follow from that spec is a mixture.
    pdir = os.path.join(run_dir, 'profiles')
    spec_copy = os.path.join(pdir, 'r1_profiles.json')
    man_copy = os.path.join(pdir, 'manifests.json')
    if os.path.isdir(pdir) and os.path.exists(spec_copy):
        if not os.path.exists(man_copy):
            problems.append('profiles/ has a spec but no manifests')
        else:
            try:
                sys.path.insert(0, HERE)
                import profiles as P
                archived = json.load(open(man_copy))
                spec_data = json.load(open(spec_copy))
                for name, man in sorted(archived.items()):
                    if name not in spec_data.get('profiles', {}):
                        problems.append(
                            f"profiles/manifests.json describes {name!r}, "
                            "which the archived spec does not define")
                        continue
                    want = P.config_hash(name, spec=spec_data,
                                        targets_dir=pdir)
                    got = man.get('config_hash')
                    if want != got:
                        problems.append(
                            f"profile {name}: archived manifest hash {got} "
                            f"does not follow from the archived spec ({want})")
                for name in sorted(spec_data.get('profiles', {})):
                    if name not in archived:
                        problems.append(
                            f"profile {name} is in the archived spec and has "
                            "no manifest")
            except Exception as e:
                problems.append(f"profiles/ could not be re-derived: "
                                f"{type(e).__name__}: {e}")

    # --- the verdict, re-derived rather than trusted ----------------------
    required = INV.required_ids()
    statuses = res.get('statuses', {})

    # The discovered set is a CLAIM. Recompute it from the statuses that were
    # actually recorded: padding `discovered` with IDs nothing ran is how a
    # run comes to report full coverage of a suite it skipped.
    claimed = set(rec.get('discovered', []))
    recorded = set(statuses)
    if claimed - recorded:
        problems.append(
            f"inventory.json claims {len(claimed - recorded)} discovered ID(s) "
            f"with no recorded status, e.g. "
            f"{sorted(claimed - recorded)[:3]}")
    if recorded - claimed:
        problems.append(
            f"results.json records {len(recorded - claimed)} status(es) that "
            f"inventory.json does not list as discovered, e.g. "
            f"{sorted(recorded - claimed)[:3]}")

    # Every required case must have a recorded status. Deleting one from
    # results.json AND from inventory.json leaves both files internally
    # consistent and the checksums correct; only the frozen inventory says
    # the case should be there at all.
    absent = set(required) - recorded
    if absent:
        problems.append(
            f"{len(absent)} required case(s) have no recorded status: "
            + ', '.join(sorted(absent)[:4])
            + (' ...' if len(absent) > 4 else ''))

    redone = INV.reconcile(required, sorted(recorded), statuses)
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
