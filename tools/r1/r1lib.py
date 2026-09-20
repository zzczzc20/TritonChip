#!/usr/bin/env python3
"""
r1lib -- the execution core the R1 acceptance gate is built on.

doc/phase1_tasks_and_acceptance.md R1-02. The one thing this module exists to
guarantee is that **a failure cannot be mistaken for a success**, in any of the
ways the audit found or predicted:

  - a non-zero exit swallowed by a shell pipeline or by `tail`
  - a simulator that prints PASS and then exits non-zero
  - a simulator that exits 0 and prints nothing, or prints both PASS and FAIL
  - a hung subprocess, or a hung *grandchild* the parent left behind
  - a result file left over from the previous run being read as this one's
  - a suite that discovered fewer cases than it was required to run
  - a skip, an xfail or an xpass counted as a pass

Every one of those is a separate check below, and every one has a matching
injection in tools/r1/selftest.py that fails the gate if the check is removed.

WHY NOT pytest / unittest
-------------------------
Both report skips and xfails as success, both leave process-group cleanup to
the OS, and neither has a notion of "the set of cases that were *required* to
run". The contract forbids all three. This is deliberately small and does one
thing.
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))

# --------------------------------------------------------------------------
# Fixed timeout caps (contract R1-02). These are NOT configurable from the
# environment: a release run that could be told to wait longer is a release
# run whose timeouts mean nothing. `selftest_scale` exists only so the
# failure-injection suite can use a one-second fake process instead of
# actually waiting ten minutes, and it can only make them SMALLER.
# --------------------------------------------------------------------------
CAPS = {
    'meta':  120,      # metadata / unit / IR case
    'unit':  120,
    'ir':    120,
    'rtl':   600,      # one RTL simulation
    'build': 3600,     # one native build step
}
ACCEPTANCE_CAP = 7200
BOOTSTRAP_CAP  = 14400

PASS, FAIL, TIMEOUT, ERROR = 'PASS', 'FAIL', 'TIMEOUT', 'ERROR'
TERMINAL_BAD = (FAIL, TIMEOUT, ERROR)

# Outcomes that some frameworks treat as success and this one does not.
FORBIDDEN_IN_STRICT = ('SKIP', 'XFAIL', 'XPASS')


class StepResult(dict):
    """One subprocess invocation, as a JSON-serialisable record."""

    @property
    def ok(self):
        return self['status'] == PASS


def _kill_group(pid, log, budget=4.0):
    """Kill the whole process group and keep going until ps says it is gone.

    "The leader has exited" is not "the group is empty": the injection this
    guards against is a parent that forks a child and then hangs, and killing
    only the leader leaves the child holding whatever it held. So the liveness
    test is ps over the process group, not a signal probe on the leader, and
    SIGKILL is re-sent until the group is empty or the budget runs out.

    The contract gives five seconds; this uses four so the caller still has a
    second to report the failure.
    """
    end = time.time() + budget
    first = True
    while time.time() < end:
        if not survivors(pid):
            return True
        sig = signal.SIGTERM if first else signal.SIGKILL
        first = False
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # A group id can be reused once its leader is gone, so a probe can
            # land on a group we do not own. ps settles it either way.
            log.append(f"killpg({pid}, {sig}): not permitted (group reused?)")
        stop = min(end, time.time() + 0.25)
        while time.time() < stop and survivors(pid):
            time.sleep(0.02)
    return not survivors(pid)


def survivors(pid):
    """PIDs still alive in the process group, for the 5-second check."""
    out = subprocess.run(['ps', '-eo', 'pid,pgid'],
                         capture_output=True, text=True).stdout
    live = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) == pid:
            live.append(int(parts[0]))
    return live


def run_step(argv, *, kind, case_dir, name, env=None, cwd=None,
             cap_override=None):
    """Run one subprocess under a hard deadline, capturing everything.

    `cap_override` may only LOWER the cap; the selftest uses it so a hang
    injection costs a second instead of ten minutes.
    """
    cap = CAPS[kind]
    if cap_override is not None:
        cap = min(cap, cap_override)

    os.makedirs(case_dir, exist_ok=True)
    out_path = os.path.join(case_dir, f"{name}.stdout")
    err_path = os.path.join(case_dir, f"{name}.stderr")
    notes = []

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    started = time.time()
    with open(out_path, 'wb') as fo, open(err_path, 'wb') as fe:
        # start_new_session makes the child a process-group leader, so a
        # timeout can take its descendants with it. Without it, killing the
        # child leaves anything it spawned running.
        proc = subprocess.Popen(argv, stdout=fo, stderr=fe, cwd=cwd or ROOT,
                                env=full_env, start_new_session=True)
        try:
            rc = proc.wait(timeout=cap)
            status = PASS if rc == 0 else FAIL
        except subprocess.TimeoutExpired:
            notes.append(f"deadline {cap}s exceeded")
            killed = _kill_group(proc.pid, notes)
            if not killed:
                notes.append("process group survived SIGKILL")
            rc = None
            status = TIMEOUT
            try:
                # Reap the leader so it does not linger as a zombie and get
                # counted as a survivor of its own process group.
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                notes.append('leader did not reap within 1s')
    elapsed = time.time() - started

    left = survivors(proc.pid)
    if left:
        notes.append(f"orphan processes still in group: {left}")
        status = ERROR if status == PASS else status

    return StepResult(
        name=name, argv=list(argv), status=status, returncode=rc,
        seconds=round(elapsed, 3), cap=cap, notes=notes,
        stdout=os.path.relpath(out_path, ROOT),
        stderr=os.path.relpath(err_path, ROOT),
        orphans=left,
    )


def read_text(path):
    try:
        with open(os.path.join(ROOT, path), 'rb') as fh:
            return fh.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def check_sim_markers(step, *, expect=PASS):
    """A simulator's verdict, checked the way the audit says it must be.

    `'PASS' in stdout` is not a verdict. Three separate things have to hold:
    the process exited 0, exactly one verdict marker is present, and it is the
    one expected. A run that printed both, or neither, is a failure -- those
    are how a corrupted or truncated log looks, and reading either as a pass
    is how a broken machine gets a green tick.
    """
    text = read_text(step['stdout']) + read_text(step['stderr'])
    has_pass = 'PASS' in text
    has_fail = 'FAIL' in text
    problems = []
    if step['status'] == TIMEOUT:
        problems.append('simulation timed out')
    if step['returncode'] not in (0, None):
        problems.append(f"simulator exit={step['returncode']}")
    if has_pass and has_fail:
        problems.append('output contains both PASS and FAIL')
    elif not has_pass and not has_fail:
        problems.append('output contains no verdict marker')
    elif expect == PASS and not has_pass:
        problems.append('expected PASS, got FAIL')
    elif expect == FAIL and not has_fail:
        problems.append('expected FAIL, got PASS')
    return problems


def fresh_run_dir(base=None, tag='run'):
    """A directory nothing else writes to.

    Per-run and per-case isolation is a correctness requirement, not tidiness:
    the audit's RUN-08 is a run that deleted its own output and silently
    consumed the previous run's.
    """
    base = base or os.path.join(ROOT, 'sim', 'r1')
    d = os.path.join(base, f"{tag}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=False)
    return d


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write('\n')


def git_state():
    """What commit this run is evidence for, and whether the tree was clean.

    An acceptance environment is a `git archive` of the candidate, so it has no
    .git at all. That is deliberate -- an environment that could commit could
    also drift -- so the commit id travels with the archive in `.r1-commit`,
    and a tree with neither is recorded as unknown rather than guessed.
    """
    def g(*a):
        r = subprocess.run(['git'] + list(a), cwd=ROOT, capture_output=True,
                           text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    if os.path.isdir(os.path.join(ROOT, '.git')):
        status = g('status', '--porcelain')
        return {
            'source': 'git',
            'commit': g('rev-parse', 'HEAD'),
            'tree': g('rev-parse', 'HEAD^{tree}'),
            'clean': status == '',
            'dirty_paths': [l[3:] for l in (status or '').splitlines()],
        }

    stamp = os.path.join(ROOT, '.r1-commit')
    if os.path.exists(stamp):
        with open(stamp) as fh:
            return {'source': 'archive', 'commit': fh.read().strip(),
                    'tree': None, 'clean': True, 'dirty_paths': []}
    return {'source': 'unknown', 'commit': None, 'tree': None,
            'clean': False, 'dirty_paths': ['no .git and no .r1-commit']}
