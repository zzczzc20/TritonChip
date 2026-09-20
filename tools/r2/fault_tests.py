#!/usr/bin/env python3
"""
ERR-01 .. ERR-07 -- doc/phase2_tasks_and_acceptance.md R2-11.

What happens when a kernel does not finish the way it was supposed to.

The rule underneath all of these is one sentence from the contract: a test
driver passing is not the same as the faulting kernel succeeding. So every
case asserts the EVENT's state, not the test's own exit code, and a task that
faulted must be FAILED -- never SUCCEEDED, and never left without a terminal
state at all.

The other rule is about what survives. A drainable fault leaves the session
usable, and the proof is that an independent, legitimate task submitted
afterwards succeeds. A session that reports a fault and then quietly stops
working has not drained; it has died politely.
"""
import argparse
import os
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R                                        # noqa: E402

from tnca import aot                                     # noqa: E402
from tnca.runtime import Session                         # noqa: E402
from tnca.errors import (BufferError, DeviceLost, SessionPoisoned,  # noqa
                         TncaError)

N = 16
SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'BLOCK': 'constexpr'}


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def session(**kw):
    d = tempfile.mkdtemp(prefix='r2-err-')
    return Session(workdir=os.path.join(d, 'dev'), deadline_cycles=200000,
                   host_timeout=120, **kw), d


def good_artifact(d):
    import kernels_triton as K
    return aot.compile_aot(K.add_kernel, signature=SIG,
                           constexprs={'BLOCK': N}, profile='core-sim',
                           output=os.path.join(d, 'add.tnca')).path


def trap_artifact(d, ident=7):
    """A TNCA IR program that traps, through the product compiler.

    `.assert_eq` on a value that is not what it claims raises TRAP_USER with
    the identifier in the fault detail. It is the simplest fault a legitimate
    compile can produce, and the point here is the runtime's handling rather
    than the trap's provenance.
    """
    def program(b, target):
        s = b.const(1)
        b.assert_eq(s, 0xDEAD, ident)      # 1 != 0xDEAD, so it traps
        b.exit()

    return aot.compile_aot(program, kernel_name='trapper', signature={},
                           profile='core-sim', source_kind='tnca-ir',
                           output=os.path.join(d, f'trap{ident}.tnca')).path


def err01(log):
    """A faulting kernel is FAILED, attributed, and never SUCCEEDED."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        mod = s.load(trap_artifact(d))
        ev = s.submit(mod, {})
        s.wait(ev)
        log.append(f"    a trapping kernel: {ev.state}")
        log.append(f"    the states it passed through: "
                   f"{' -> '.join(h[0] for h in ev.history)}")
        log.append(f"    the error: [{ev.error.code}] {ev.error.message[:56]}")
        log.append(f"    attribution: {ev.error.detail}")
        ok &= ev.state == 'FAILED'
        ok &= ev.error is not None and ev.error.code == 'KERNEL_FAULT'
        # The states matter: a fault goes through FAILING and DRAINING, so a
        # reader can tell "it stopped" from "it was stopped and drained".
        states = [h[0] for h in ev.history]
        ok &= 'FAILING' in states and 'DRAINING' in states
        log.append(f"    it passed through FAILING and DRAINING: "
                   f"{'FAILING' in states and 'DRAINING' in states}")
    finally:
        s.poisoned = True
        s.close()
    return ok, 'a fault is FAILED, with a trap code and a PC'


def err02(log):
    """After a drainable fault the session still works."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        bad = s.load(trap_artifact(d))
        good = s.load(good_artifact(d))

        ev_bad = s.submit(bad, {})
        s.wait(ev_bad)
        log.append(f"    the faulting task: {ev_bad.state}")
        ok &= ev_bad.state == 'FAILED'

        # The session is NOT poisoned by a drainable fault.
        log.append(f"    the session after it: poisoned={s.poisoned}")
        ok &= not s.poisoned

        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        xs = [float(i) for i in range(N)]
        s.upload(x, b''.join(struct.pack('<f', v) for v in xs))
        s.upload(y, b''.join(struct.pack('<f', 1.0) for _ in range(N)))
        ev = s.submit(good, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N})
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        want = [v + 1.0 for v in xs]
        log.append(f"    an independent task afterwards: {ev.state}, "
                   f"correct={got == want}")
        ok &= ev.ok and got == want
    finally:
        s.poisoned = True
        s.close()
    return ok, 'a drained fault leaves the session usable'


def err04(log):
    """recover() starts a new epoch and invalidates every old handle."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        mod = s.load(good_artifact(d))
        old_x = s.allocate(N * 4, name='old')
        old_epoch = s.epoch
        s.upload(old_x, b'\x00' * (N * 4))

        new_epoch = s.recover()
        log.append(f"    recover(): epoch {old_epoch} -> {new_epoch}")
        ok &= new_epoch == old_epoch + 1

        for what, call in (
                ('a buffer from the old epoch',
                 lambda: s.upload(old_x, b'\x00' * 8)),
                ('a module from the old epoch',
                 lambda: s.submit(mod, {}))):
            try:
                call()
                log.append(f"    {what}: ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                log.append(f"    {what:<32s} -> [{e.code}]")

        # And the session works again after re-uploading and re-loading.
        mod2 = s.load(good_artifact(d))
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        xs = [float(i) for i in range(N)]
        s.upload(x, b''.join(struct.pack('<f', v) for v in xs))
        s.upload(y, b''.join(struct.pack('<f', 2.0) for _ in range(N)))
        ev = s.submit(mod2, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N})
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        log.append(f"    after re-uploading and re-loading: {ev.state}, "
                   f"correct={got == [v + 2.0 for v in xs]}")
        ok &= ev.ok and got == [v + 2.0 for v in xs]
    finally:
        s.poisoned = True
        s.close()
    return ok, 'a new epoch, every old handle refused, and work again'


def err06(log):
    """A device that dies is DEVICE_LOST, with a log and no orphan."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        mod = s.load(good_artifact(d))
        pid = s.dev.proc.pid
        log.append(f"    the device is pid {pid}")

        # Kill it the way a crash would.
        s.dev.proc.kill()
        s.dev.proc.wait(timeout=10)

        try:
            s.dev.ping()
            log.append("    a request to a dead device: ACCEPTED (bad)")
            ok = False
        except DeviceLost as e:
            log.append(f"    a request to a dead device -> [{e.code}] "
                       f"{e.message[:46]}")
            log.append(f"    it kept the log: "
                       f"{os.path.basename(e.detail.get('log', '')) or 'none'}")
            ok &= bool(e.detail.get('log'))

        # No orphan: the process is reaped, not left behind.
        alive = s.dev.alive()
        log.append(f"    the device process is still running: {alive}")
        ok &= not alive

        # And recover brings the session back.
        s.recover()
        mod2 = s.load(good_artifact(d))
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        s.upload(x, b''.join(struct.pack('<f', 3.0) for _ in range(N)))
        s.upload(y, b''.join(struct.pack('<f', 4.0) for _ in range(N)))
        ev = s.submit(mod2, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N})
        s.wait(ev)
        got = list(struct.unpack(f'<{N}f', s.download(o, N * 4)))
        log.append(f"    after recover(): {ev.state}, correct="
                   f"{got == [7.0] * N}")
        ok &= ev.ok and got == [7.0] * N
    finally:
        s.poisoned = True
        s.close()
    return ok, 'a dead device is DEVICE_LOST, reaped, and recoverable'


def err07(log):
    """Sixteen alternating faulting and legitimate tasks."""
    if not have_triton():
        return False, 'needs the locked Triton'
    s, d = session()
    ok = True
    try:
        bad = s.load(trap_artifact(d))
        good = s.load(good_artifact(d))
        x, y, o = (s.allocate(N * 4, name=c) for c in 'xyo')
        xs = [float(i) for i in range(N)]
        s.upload(x, b''.join(struct.pack('<f', v) for v in xs))
        s.upload(y, b''.join(struct.pack('<f', 1.0) for _ in range(N)))
        args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N}

        states = []
        for i in range(16):
            if i % 2 == 0:
                ev = s.submit(bad, {})
            else:
                ev = s.submit(good, args)
            s.wait(ev)
            states.append((i % 2, ev.state))

        faults = [st for kind, st in states if kind == 0]
        goods = [st for kind, st in states if kind == 1]
        log.append(f"    8 faulting tasks: {sorted(set(faults))}")
        log.append(f"    8 legitimate tasks: {sorted(set(goods))}")
        ok &= set(faults) == {'FAILED'} and set(goods) == {'SUCCEEDED'}

        # No leaks: every event terminal, every lease released.
        unfinished = [e.job_id for e in s._events.values() if not e.terminal]
        leases = {b.name: b.leases for b in s._buffers.values() if b.leases}
        log.append(f"    events without a terminal state: "
                   f"{unfinished or 'none'}")
        log.append(f"    buffers still leased: {leases or 'none'}")
        ok &= not unfinished and not leases

        # A task that never ran is not reported as having run.
        ran = [e.job_id for e in s._events.values()
               if any(h[0] == 'RUNNING' for h in e.history)]
        log.append(f"    tasks that reached RUNNING: {len(ran)} of "
                   f"{len(s._events)}")
        ok &= len(ran) == len(s._events)
    finally:
        s.poisoned = True
        s.close()
    return ok, '16 alternating tasks, no leak, no task reported wrongly'


TESTS = [('ERR-01', err01), ('ERR-02', err02), ('ERR-04', err04),
         ('ERR-06', err06), ('ERR-07', err07)]

NOT_YET = {
    'ERR-03': 'a device deadline with a non-returning transaction needs a '
              'fault-injection hook in the memory model (R2-11 batch 4)',
    'ERR-05': 'late and duplicate responses from an old epoch need the '
              'transport epoch tag to be exercised by injection',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-11 faults, drain and recovery")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-800:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    for cid, why in sorted(NOT_YET.items()):
        if args.only and cid not in args.only:
            continue
        results.append(dict(id=cid, passed=False, detail=f"NOT RUN: {why}"))
        print(f"  {cid}  NOT RUN  {why}", flush=True)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-fault-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-fault-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
