#!/usr/bin/env python3
"""
Machine-code and cycle fidelity against the frozen baseline --
phase3_6_tasks_and_acceptance.md section 5.2, checked by R36-TEST-02.

Section 5.2, in its own words: on the same hardware implementation, a
purely engineering refactor must not change the machine code (ignoring
metadata declared non-semantic in advance) or the device cycles of
R3.5's frozen workloads; a necessary semantic or implementation change
must be identified separately, with its own regression and performance
evidence, and must not be folded in with directory and interface tidying
as one "no behaviour change".

So this does not assert that nothing changed. It MEASURES both trees and
requires every difference to be accounted for by name.

HOW THE BASELINE IS OBTAINED

Not from a recorded number. A git worktree is created at the frozen
commit and the workloads are compiled and run there, in that tree, with
that tree's compiler and that tree's device. A baseline read out of a
JSON file written by the tree being tested is the tree agreeing with
itself.

WHAT COUNTS AS A DIFFERENCE

Two things per workload per mode: a digest of the machine code, and the
device cycles. Both come from the same run. Anything that differs has to
appear in spec/r36_baseline_delta.json with a class:

    engineering        the change was supposed to be invisible. A
                       difference here FAILS -- that is the whole point
                       of the standard.
    semantic           a deliberate change in what the machine does or
                       what the compiler emits. Allowed, and required to
                       name its evidence.

An unattributed difference fails whichever it is.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

DELTA = json.load(open(os.path.join(ROOT, 'spec',
                                    'r36_baseline_delta.json')))
BASELINE_COMMIT = DELTA['baseline']['commit']
PROFILE = json.load(open(os.path.join(
    ROOT, 'spec', 'r35_workloads.json')))['hardware_precondition']['profile']

# The measurement, run inside whichever tree it is pointed at. Written
# once and used for both, so the baseline and the candidate are measured
# by the same code -- a difference in the measurement is then not one of
# the things that can differ.
PROBE = r"""
# Measure ONE workload in ONE mode, in its own process, in whichever
# tree it was pointed at.
#
# One process per measurement rather than one for all of them: R3.5
# found that a shared interpreter and a shared temporary directory moved
# L06's baseline by twenty cycles, which is larger than most of the
# differences this file exists to notice.
import hashlib, json, os, sys, tempfile

ROOT, OUT, WID, MODE, REPEATS = sys.argv[1:6]
REPEATS = int(REPEATS)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r35'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))

out = {}
try:
    import perf as P
    from tnca import aot

    # The machine code, caught on its way past. perf.py already wraps
    # compile_aot to count words; wrapping it first makes this the inner
    # one, so it sees every compile the measurement performs.
    seen = []
    real = aot.compile_aot

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen.append(r.asm['hex'])
        return r
    aot.compile_aot = spy
    runs = []
    try:
        for _ in range(REPEATS):
            runs.append(P.measure_one(WID, MODE, tempfile.mkdtemp()))
    finally:
        aot.compile_aot = real
    cyc = sorted(r.get('cycles') for r in runs
                 if r.get('cycles') is not None)
    out = {'cycles': cyc[len(cyc) // 2] if cyc else None,
           'samples': cyc,
           'spread': (cyc[-1] - cyc[0]) if cyc else None,
           'words': runs[0].get('words'),
           'correct': all(r.get('correct') for r in runs),
           'compiles': len(seen),
           'code': hashlib.sha256(
               '\n'.join(seen).encode()).hexdigest()[:16] if seen else None}
except Exception as e:
    import traceback
    out = {'error': f"{type(e).__name__}: {e}",
           'trace': traceback.format_exc()[-600:]}
json.dump(out, open(OUT, 'w'), indent=1, sort_keys=True)
"""


def _run(cmd, cwd=ROOT, timeout=7200, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def workloads():
    with open(os.path.join(ROOT, 'spec', 'r35_workloads.json')) as fh:
        return [w['id'] for w in json.load(fh)['workloads']]


# R3.5 measured these and recorded why: the slow-reorder memory does not
# return the same number twice, and a single reading is a draw from a
# distribution rather than a measurement of it. A fidelity check that
# compared one draw against one draw would report a difference whenever
# the dice disagreed.
STOCHASTIC = {'L06'}
REPEATS = 5


def measure(tree, modes, python, label, log):
    """Compile and run every frozen workload in `tree`."""
    d = tempfile.mkdtemp(prefix='tnca-probe-')
    probe = os.path.join(d, 'probe.py')
    with open(probe, 'w') as fh:
        fh.write(PROBE)
    out = {}
    for wid in workloads():
        out[wid] = {}
        for mode in modes:
            res_path = os.path.join(d, f"{wid}-{mode}.json")
            reps = REPEATS if wid in STOCHASTIC else 1
            r = _run([python, probe, tree, res_path, wid, mode,
                      str(reps)], cwd=tree)
            if not os.path.exists(res_path):
                log.append(f"    {label} {wid}/{mode}: the probe produced "
                           f"nothing: {(r.stderr or r.stdout).strip()[-200:]}")
                out[wid][mode] = {'error': 'no result'}
                continue
            with open(res_path) as fh:
                out[wid][mode] = json.load(fh)
            e = out[wid][mode].get('error')
            if e:
                log.append(f"    {label} {wid}/{mode}: {e[:100]}")
    done = sum(1 for w in out for m in out[w] if 'error' not in out[w][m])
    log.append(f"    {label}: {done}/{len(out) * len(modes)} measurement(s)")
    shutil.rmtree(d, ignore_errors=True)
    return out


def worktree(commit, log):
    """A worktree at the frozen commit, and its own staged data.

    The runtime reads python/tnca/data in preference to the working
    tree, and that directory is not tracked, so a fresh worktree has
    none -- which is right: the baseline must stage its OWN sources, or
    it would measure the candidate's RTL with the baseline's compiler.
    """
    d = tempfile.mkdtemp(prefix='tnca-baseline-')
    tree = os.path.join(d, 'tree')
    r = _run(['git', 'worktree', 'add', '--detach', tree, commit])
    if r.returncode != 0:
        log.append(f"    could not create a worktree at {commit}: "
                   f"{r.stderr.strip()[:140]}")
        return None, None
    log.append(f"    a worktree at {commit} in {tree}")
    return tree, d


def release(tree, holder):
    if tree:
        _run(['git', 'worktree', 'remove', '--force', tree])
    if holder:
        shutil.rmtree(holder, ignore_errors=True)


def compare(base, cand, log):
    """Every difference, attributed or not.

    A stochastic workload's cycle difference is only a difference if it
    clears the spread of the two samples that produced it. R3.5 recorded
    the same rule for its gains, and it cuts the same way here: a
    difference smaller than the noise has not been shown either.
    """
    rows = []
    for wid in sorted(set(base) | set(cand)):
        for mode in sorted(set(base.get(wid, {})) | set(cand.get(wid, {}))):
            b = base.get(wid, {}).get(mode, {})
            c = cand.get(wid, {}).get(mode, {})
            for field in ('code', 'cycles'):
                if b.get(field) == c.get(field):
                    continue
                row = {'workload': wid, 'mode': mode, 'field': field,
                       'baseline': b.get(field),
                       'candidate': c.get(field)}
                if field == 'cycles' and wid in STOCHASTIC:
                    noise = max(b.get('spread') or 0, c.get('spread') or 0)
                    row['noise'] = noise
                    row['samples'] = {'baseline': b.get('samples'),
                                      'candidate': c.get('samples')}
                    if abs((c.get('cycles') or 0)
                           - (b.get('cycles') or 0)) <= noise:
                        row['within_noise'] = True
                        log.append(
                            f"    {wid}/{mode}: {b.get('cycles')} -> "
                            f"{c.get('cycles')}, inside the measured "
                            f"spread of {noise}; NOT SHOWN")
                        continue
                rows.append(row)
    return rows


def attribute(rows, log):
    """Match each difference to a declared change."""
    changes = DELTA['changes']
    unattributed, by_change = [], {}
    for row in rows:
        hit = None
        for ch in changes:
            scope = ch['covers']
            if row['field'] not in scope.get('fields', []):
                continue
            wls = scope.get('workloads')
            if wls not in (None, 'all') and row['workload'] not in wls:
                continue
            md = scope.get('modes')
            if md not in (None, 'all') and row['mode'] not in md:
                continue
            hit = ch
            break
        if hit is None:
            unattributed.append(row)
        else:
            by_change.setdefault(hit['id'], []).append(row)
    return unattributed, by_change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='*', default=['B0', 'O2'])
    ap.add_argument('--python', default=os.environ.get(
        'TRITON_PYTHON', sys.executable))
    ap.add_argument('--json')
    args = ap.parse_args()

    log = []
    print("Section 5.2: the machine code and the cycles, against the "
          "frozen baseline")
    print(f"  baseline {BASELINE_COMMIT} -- "
          f"{DELTA['baseline']['what']}")
    print(f"  profile  {PROFILE}, modes {args.modes}")

    tree, holder = worktree(BASELINE_COMMIT, log)
    for line in log:
        print(line, flush=True)
    if tree is None:
        return 2
    log = []
    try:
        base = measure(tree, args.modes, args.python, 'the baseline', log)
        for line in log:
            print(line, flush=True)
        if base is None:
            return 2
        log = []
        cand = measure(ROOT, args.modes, args.python, 'the candidate', log)
        for line in log:
            print(line, flush=True)
        if cand is None:
            return 2
    finally:
        release(tree, holder)

    rows = compare(base, cand, log)
    print(f"  {len(rows)} difference(s) between the two trees")
    unattributed, by_change = attribute(rows, log)
    for cid, hits in sorted(by_change.items()):
        ch = next(c for c in DELTA['changes'] if c['id'] == cid)
        print(f"    {cid} [{ch['class']}] {ch['what'][:62]}")
        print(f"      {len(hits)} difference(s): "
              f"{sorted({(h['workload'], h['field']) for h in hits})}")
        print(f"      evidence: {ch['evidence'][:70]}")
        if ch['class'] == 'engineering':
            print("      A DIFFERENCE UNDER AN ENGINEERING CHANGE IS A "
                  "FAILURE (section 5.2)")
    for row in unattributed:
        print(f"    UNATTRIBUTED {row['workload']} {row['mode']} "
              f"{row['field']}: {row['baseline']} -> {row['candidate']}")
    # A declared change that moved nothing is not a failure, and it is
    # not nothing either: it means these workloads do not exercise it,
    # and the reader should know that rather than infer from the
    # declaration that it was measured.
    idle = [c['id'] for c in DELTA['changes'] if c['id'] not in by_change]
    if idle:
        print(f"    declared and not exercised by these workloads: "
              f"{idle}")
        for cid in idle:
            ch = next(c for c in DELTA['changes'] if c['id'] == cid)
            print(f"      {cid}: {ch['what'][:74]}")

    bad_class = [cid for cid in by_change
                 if next(c for c in DELTA['changes']
                         if c['id'] == cid)['class'] == 'engineering']
    ok = not unattributed and not bad_class
    result = {'baseline_commit': BASELINE_COMMIT, 'profile': PROFILE,
              'modes': args.modes, 'baseline': base, 'candidate': cand,
              'differences': rows, 'unattributed': unattributed,
              'attributed': {k: len(v) for k, v in by_change.items()},
              'ok': ok}
    if args.json:
        R.write_json(args.json, result)
    print(f"  fidelity  {'PASS' if ok else 'FAIL'}: "
          f"{len(rows)} difference(s), {len(unattributed)} unattributed")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
