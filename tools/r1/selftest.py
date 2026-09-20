#!/usr/bin/env python3
"""
Prove the gate can fail -- doc/phase1_tasks_and_acceptance.md R1-02, RUN-01..10.

A test suite that has never been seen to fail is not evidence of anything. The
audit found the project's end-to-end entry points returning success with a
deliberately broken interpreter, so before any R1 result is believed, the thing
that produces R1 results has to be shown detecting each way a run can go wrong.

Each injection below breaks one thing on purpose and asserts the checker
noticed. This program exits 0 only when **every** injection was detected; an
injection that slips through is this program's failure, not its success.

Nothing here modifies the repository. Fake interpreters, fake simulators and
fake hangs are written into a scratch directory and deleted with it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1lib as R  # noqa: E402

ROOT = R.ROOT


class Check:
    def __init__(self):
        self.results = []

    def record(self, cid, detected, detail):
        self.results.append(dict(id=cid, detected=bool(detected), detail=detail))
        mark = 'detected' if detected else '** MISSED **'
        print(f"  {cid:<8s} {mark:<12s} {detail}")
        return detected

    @property
    def ok(self):
        return all(r['detected'] for r in self.results)


def script(tmp, name, body, mode=0o755):
    # lstrip: a dedented triple-quoted string starts with a newline, which
    # would put the shebang on line 2 and make exec() fail with ENOEXEC.
    p = os.path.join(tmp, name)
    with open(p, 'w') as fh:
        fh.write(textwrap.dedent(body).lstrip('\n'))
    os.chmod(p, mode)
    return p


def make_exit(target, extra=()):
    """Run one make target and return its exit code."""
    r = subprocess.run(['make', target] + list(extra), cwd=ROOT,
                       capture_output=True, text=True)
    return r.returncode


# ---------------------------------------------------------------------------
def run01(c, tmp):
    """Both E2E entry points, with an interpreter that cannot succeed."""
    a = make_exit('triton-e2e', ['TRITON_PYTHON=/usr/bin/false'])
    b = make_exit('e2e', ['TRITON_PYTHON=/usr/bin/false'])
    return c.record('RUN-01', a != 0 and b != 0,
                    f"make triton-e2e exit={a}, make e2e exit={b}")


def run02(c, tmp):
    """A missing tool is a failure in strict mode, not a skip."""
    miss = os.path.join(tmp, 'no-such-python')
    strict = make_exit('triton-e2e', [f'TRITON_PYTHON={miss}', 'R1_STRICT=1'])
    dev = make_exit('triton-e2e', [f'TRITON_PYTHON={miss}'])

    # The IR runner, whose tools may legitimately be absent on a dev box.
    env = dict(os.environ, R1_STRICT='1', TNCA_LLVM=os.path.join(tmp, 'nollvm'))
    ir = subprocess.run([sys.executable, 'compiler/test/IR/run_ir_tests.py'],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    ir_strict_fails = ir.returncode != 0 and 'SKIP' not in ir.stdout

    # An interpreter that imports but then raises.
    boom = script(tmp, 'boom.py', """
        import sys
        raise ImportError("triton not importable (injected)")
    """)
    st = R.run_step([sys.executable, boom], kind='meta',
                    case_dir=os.path.join(tmp, 'run02'), name='import')
    return c.record('RUN-02',
                    strict != 0 and dev == 0 and ir_strict_fails
                    and st['status'] == R.FAIL,
                    f"strict={strict} dev={dev} ir_strict={ir.returncode} "
                    f"import_error={st['status']}")


def run03(c, tmp):
    """A Python assertion and a non-zero assembler keep their diagnostics."""
    bad = script(tmp, 'assert.py', """
        assert False, "injected assertion"
    """)
    st = R.run_step([sys.executable, bad], kind='meta',
                    case_dir=os.path.join(tmp, 'run03'), name='assert')
    kept = 'injected assertion' in R.read_text(st['stderr'])

    asm = subprocess.run([sys.executable, 'tools/asm/tnca_asm.py',
                          os.path.join(tmp, 'nonexistent.s'), '-o',
                          os.path.join(tmp, 'x.hex')],
                         cwd=ROOT, capture_output=True, text=True)
    return c.record('RUN-03', st['status'] == R.FAIL and kept
                    and asm.returncode != 0,
                    f"assert={st['status']} traceback_kept={kept} "
                    f"assembler_exit={asm.returncode}")


def run04(c, tmp):
    """A simulator that prints PASS and then exits non-zero."""
    sim = script(tmp, 'sim_pass_then_7.sh', """
        #!/bin/sh
        echo "tb_fake: 1 vector, 0 failures"
        echo PASS
        exit 7
    """)
    st = R.run_step([sim], kind='rtl', case_dir=os.path.join(tmp, 'run04'),
                    name='sim')
    problems = R.check_sim_markers(st)
    return c.record('RUN-04',
                    st['status'] == R.FAIL and st['returncode'] == 7
                    and any('exit=7' in p for p in problems),
                    f"status={st['status']} rc={st['returncode']} {problems}")


def run05(c, tmp):
    """Exit 0, but the output says nothing, or says two things, or is junk."""
    cases = {
        'silent':   '#!/bin/sh\nexit 0\n',
        'both':     '#!/bin/sh\necho PASS\necho FAIL\nexit 0\n',
        'corrupt':  '#!/bin/sh\nprintf "tb_fake: \\xff\\xfe garbage"\nexit 0\n',
    }
    verdicts = {}
    for name, body in cases.items():
        s = script(tmp, f'sim_{name}.sh', body)
        st = R.run_step([s], kind='rtl',
                        case_dir=os.path.join(tmp, 'run05', name), name='sim')
        verdicts[name] = R.check_sim_markers(st)
    return c.record('RUN-05', all(verdicts.values()),
                    '; '.join(f"{k}={len(v)} problem(s)" for k, v in verdicts.items()))


def run06(c, tmp):
    """A hang that leaves a hanging grandchild behind."""
    child = script(tmp, 'hang_child.sh', """
        #!/bin/sh
        sleep 600
    """)
    parent = script(tmp, 'hang_parent.sh', f"""
        #!/bin/sh
        {child} &
        sleep 600
    """)
    st = R.run_step([parent], kind='rtl',
                    case_dir=os.path.join(tmp, 'run06'), name='hang',
                    cap_override=1)
    # run_step already re-checked the group after killing it; the contract
    # gives five seconds and this is well inside that.
    return c.record('RUN-06',
                    st['status'] == R.TIMEOUT and not st['orphans'],
                    f"status={st['status']} orphans={st['orphans']} "
                    f"elapsed={st['seconds']}s (cap 1s)")


def run07(c, tmp):
    """Inventory failures: empty, missing, duplicated, or skipped."""
    import inventory as INV
    required = {'A', 'B', 'C'}
    allpass = {'A': 'PASS', 'B': 'PASS', 'C': 'PASS'}

    def st(**over):
        d = dict(allpass)
        d.update(over)
        return d

    bad = {
        'empty':      INV.reconcile(required, [], {}),
        'missing':    INV.reconcile(required, ['A', 'B'], st()),
        'duplicate':  INV.reconcile(required, ['A', 'B', 'C', 'C'], st()),
        'skip':       INV.reconcile(required, ['A', 'B', 'C'], st(C='SKIP')),
        'xfail':      INV.reconcile(required, ['A', 'B', 'C'], st(C='XFAIL')),
        'xpass':      INV.reconcile(required, ['A', 'B', 'C'], st(C='XPASS')),
        'notrun':     INV.reconcile(required, ['A', 'B', 'C'], {'A': 'PASS'}),
    }
    good = INV.reconcile(required, ['A', 'B', 'C'], allpass)
    return c.record('RUN-07',
                    all(v['problems'] for v in bad.values()) and not good['problems'],
                    '; '.join(f"{k}={len(v['problems'])}" for k, v in bad.items())
                    + f"; clean={len(good['problems'])}")


def run08(c, tmp):
    """This run's dump is missing; last run's correct dump is still there."""
    import inventory as INV
    old = os.path.join(tmp, 'run08', 'old')
    new = R.fresh_run_dir(os.path.join(tmp, 'run08'), 'dump')
    os.makedirs(old, exist_ok=True)
    with open(os.path.join(old, 'result.dump'), 'w') as fh:
        fh.write('0000000000000001\n')
    # The new run produces nothing.
    ok, why = INV.require_fresh_artifact(os.path.join(new, 'result.dump'))
    stale_ok, _ = INV.require_fresh_artifact(os.path.join(old, 'result.dump'))
    return c.record('RUN-08', (not ok) and stale_ok,
                    f"missing_this_run={why!r}; previous run's file unaffected")


def run09(c, tmp):
    """A pipeline whose first stage fails and whose last stage succeeds."""
    naive = subprocess.run(['/bin/sh', '-c', 'false | true'],
                           capture_output=True).returncode
    guarded = subprocess.run(['/bin/bash', '-o', 'pipefail', '-c', 'false | true'],
                             capture_output=True).returncode
    # And the repository must not contain the naive form in a place whose exit
    # code is a verdict.
    import inventory as INV
    offenders = INV.unguarded_pipelines()
    return c.record('RUN-09',
                    naive == 0 and guarded != 0 and not offenders,
                    f"naive={naive} guarded={guarded} offenders={offenders}")


def run10(c, tmp):
    """Two concurrent runs of the same case, with different inputs."""
    tool = script(tmp, 'echo_input.sh', """
        #!/bin/sh
        printf '%s' "$1"
        echo
        echo PASS
    """)
    out = {}

    def one(tag, payload):
        d = R.fresh_run_dir(os.path.join(tmp, 'run10'), tag)
        st = R.run_step([tool, payload], kind='meta', case_dir=d, name='case')
        out[tag] = (R.read_text(st['stdout']).splitlines()[0], st, d)

    ts = [threading.Thread(target=one, args=(t, p))
          for t, p in (('a', 'INPUT-AAA'), ('b', 'INPUT-BBB'))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    correct = (out['a'][0] == 'INPUT-AAA' and out['b'][0] == 'INPUT-BBB')
    disjoint = out['a'][2] != out['b'][2]
    return c.record('RUN-10', correct and disjoint,
                    f"a={out['a'][0]} b={out['b'][0]} separate_dirs={disjoint}")


INJECTIONS = [run01, run02, run03, run04, run05,
              run06, run07, run08, run09, run10]


def main():
    print("R1-02 gate self-test: ten injected failures, each must be detected")
    c = Check()
    tmp = tempfile.mkdtemp(prefix='r1-selftest-')
    try:
        for fn in INJECTIONS:
            try:
                fn(c, tmp)
            except Exception as e:  # an injection that crashes is a miss
                c.record(fn.__name__.upper().replace('RUN', 'RUN-'), False,
                         f"self-test raised {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    missed = [r['id'] for r in c.results if not r['detected']]
    print()
    if c.ok:
        print(f"  r1-gate-selftest       PASS   {len(c.results)}/10 injections detected")
        return 0
    print(f"  r1-gate-selftest       FAIL   missed: {', '.join(missed)}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
