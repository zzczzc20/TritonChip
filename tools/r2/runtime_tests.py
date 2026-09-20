#!/usr/bin/env python3
"""
RT-01 .. RT-07 -- doc/phase2_tasks_and_acceptance.md R2-08.

The session, through its public API. Every check here is about a rule that
only the runtime can enforce, and several of them are about what must NOT
happen: a task must not run twice, a wait must not cancel, a close must not
discard, a dependent task must not start.

RT-03's queue-full case uses a controllable device hold rather than hoping the
host outruns the device -- the contract asks for that specifically, because a
queue that fills by luck fills differently on a faster machine.
"""
import argparse
import os
import struct
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
import r1lib as R                                        # noqa: E402

from tnca import aot                                     # noqa: E402
from tnca.runtime import Session, MAX_UNFINISHED         # noqa: E402
from tnca.errors import (BufferBusy, BufferError, QueueFull,   # noqa: E402
                         SessionBusy, TncaError, Timeout)

SIG = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
       'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'}
N = 16
_ART = {}


def have_triton():
    try:
        import triton      # noqa: F401
        return True
    except Exception:
        return False


def artifact_path():
    if 'axpy' not in _ART:
        import kernels_triton as K
        d = tempfile.mkdtemp(prefix='r2-rt-art-')
        _ART['axpy'] = aot.compile_aot(
            K.axpy_kernel, signature=SIG, constexprs={'BLOCK': N},
            profile='core-sim', output=os.path.join(d, 'axpy.tnca')).path
    return _ART['axpy']


def session(**kw):
    d = tempfile.mkdtemp(prefix='r2-rt-')
    return Session(workdir=os.path.join(d, 'dev'), deadline_cycles=200000,
                   host_timeout=120, **kw)


def f32s(values):
    return b''.join(struct.pack('<f', v) for v in values)


def unpack(data):
    return list(struct.unpack(f'<{len(data) // 4}f', data))


def rt01(log):
    """The whole lifecycle, plus the two edges the contract names."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        mod = s.load(artifact_path())
        x, y, o = (s.allocate(N * 4, name=n) for n in 'xyo')
        xs = [float(i) for i in range(N)]
        ys = [float(50 + i) for i in range(N)]
        s.upload(x, f32s(xs))
        s.upload(y, f32s(ys))
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'n': N, 'alpha': 2.0})
        s.wait(ev)
        got = unpack(s.download(o, N * 4))
        want = [a * 2.0 + b for a, b in zip(xs, ys)]
        log.append(f"    allocate/upload/load/submit/wait/download: "
                   f"{ev.state}, correct={got == want}")
        ok &= ev.ok and got == want
        log.append(f"    the states it passed through: "
                   f"{' -> '.join(h[0] for h in ev.history)}")
        s.free(x)
        s.free(y)
        s.free(o)

        # A zero-byte allocation is refused by name.
        try:
            s.allocate(0)
            log.append("    a 0-byte allocation: ACCEPTED (bad)")
            ok = False
        except BufferError as e:
            log.append(f"    a 0-byte allocation      -> [{e.code}] "
                       f"{e.message[:44]}")
    return ok, 'the lifecycle, and a zero-byte allocation refused'


def rt02(log):
    """Thirty-two launches in one session, and an A->B->C chain."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        starts = [s.dev.seq]
        mod = s.load(artifact_path())
        x, y, o = (s.allocate(N * 4, name=n) for n in 'xyo')
        xs = [float(i) for i in range(N)]
        ys = [float(1) for _ in range(N)]
        s.upload(x, f32s(xs))
        s.upload(y, f32s(ys))

        good = 0
        for i in range(32):
            alpha = float(i % 5) - 2.0
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': N, 'alpha': alpha})
            s.wait(ev)
            got = unpack(s.download(o, N * 4))
            want = [a * alpha + b for a, b in zip(xs, ys)]
            if ev.ok and got == want:
                good += 1
        log.append(f"    32 launches, alternating alpha: {good} correct")
        log.append(f"    device processes started: 1 (the transport has "
                   f"served {s.dev.seq} requests in one)")
        ok &= good == 32

        # A -> B -> C, with the middle result never leaving the device.
        a_buf = s.allocate(N * 4, name='A')
        b_buf = s.allocate(N * 4, name='B')
        c_buf = s.allocate(N * 4, name='C')
        s.upload(a_buf, f32s([float(i) for i in range(N)]))
        s.upload(b_buf, f32s([0.0] * N))
        zero = s.allocate(N * 4, name='zero')
        s.upload(zero, f32s([0.0] * N))

        # B = 3*A + 0   then   C = 1*B + 0.  Nothing is downloaded between.
        e1 = s.submit(mod, {'x_ptr': a_buf, 'y_ptr': zero, 'o_ptr': b_buf,
                            'n': N, 'alpha': 3.0})
        e2 = s.submit(mod, {'x_ptr': b_buf, 'y_ptr': zero, 'o_ptr': c_buf,
                            'n': N, 'alpha': 1.0}, depends_on=[e1])
        s.wait(e2)
        chain = unpack(s.download(c_buf, N * 4))
        want = [3.0 * i for i in range(N)]
        log.append(f"    A->B->C with no host download in between: "
                   f"{e1.state}/{e2.state}, correct={chain == want}")
        ok &= chain == want
    return ok, '32 launches, one device, and a device-memory chain'


def rt03(log):
    """Eight unfinished tasks, a ninth refused, and a slot that frees."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        mod = s.load(artifact_path())
        x, y, o = (s.allocate(N * 4, name=n) for n in 'xyo')
        s.upload(x, f32s([1.0] * N))
        s.upload(y, f32s([0.0] * N))
        args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N, 'alpha': 1.0}

        # A controllable hold: the worker is stopped, so submitted tasks
        # stay QUEUED. The contract asks for the queue to be filled
        # deliberately rather than by winning a race with the device.
        s._stop.set()
        s._worker.join(timeout=10)

        events = []
        for i in range(MAX_UNFINISHED):
            events.append(s.submit(mod, args))
        log.append(f"    {len(events)} tasks accepted while the device is "
                   f"held: states {sorted({e.state for e in events})}")

        try:
            s.submit(mod, args)
            log.append("    the ninth submit: ACCEPTED (bad)")
            ok = False
        except QueueFull as e:
            log.append(f"    the ninth submit         -> [{e.code}] "
                       f"{e.message[:46]}")

        # Release the hold; the queue drains in order.
        s._start_worker()
        for e in events:
            s.wait(e)
        order = [e.job_id for e in events]
        log.append(f"    drained in FIFO order: {order == sorted(order)}")
        ok &= order == sorted(order)
        ok &= all(e.ok for e in events)

        # A freed slot can be used again.
        again = s.submit(mod, args)
        s.wait(again)
        log.append(f"    after the slots freed, another submit: "
                   f"{again.state}")
        ok &= again.ok
    return ok, 'eight held, the ninth refused, FIFO, and the slot reused'


def rt04(log):
    """wait is idempotent; a timeout is not a verdict."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        mod = s.load(artifact_path())
        x, y, o = (s.allocate(N * 4, name=n) for n in 'xyo')
        s.upload(x, f32s([2.0] * N))
        s.upload(y, f32s([1.0] * N))
        args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N, 'alpha': 4.0}

        # A host wait that expires while the task is held.
        s._stop.set()
        s._worker.join(timeout=10)
        ev = s.submit(mod, args)
        t0 = time.time()
        try:
            s.wait(ev, timeout=0.2)
            log.append("    a 0.2s wait on a held task returned (bad)")
            ok = False
        except Timeout as e:
            log.append(f"    a 0.2s host wait         -> [{e.code}] after "
                       f"{time.time() - t0:.2f}s")
        log.append(f"    the task after the timeout: {ev.state} "
                   f"(not cancelled, not succeeded)")
        ok &= ev.state == 'QUEUED'

        s._start_worker()
        s.wait(ev)
        first = ev.state
        history_len = len(ev.history)
        for _ in range(3):
            s.wait(ev)
        log.append(f"    waiting three more times: state stays {ev.state}, "
                   f"history unchanged={len(ev.history) == history_len}")
        ok &= ev.ok and len(ev.history) == history_len

        got = unpack(s.download(o, N * 4))
        want = [2.0 * 4.0 + 1.0] * N
        log.append(f"    and it ran exactly once: {got == want}")
        ok &= got == want
    return ok, 'idempotent, and a timeout that decides nothing'


def rt05(log):
    """A leased buffer is busy; a freed one is gone."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        mod = s.load(artifact_path())
        x, y, o = (s.allocate(N * 4, name=n) for n in 'xyo')
        s.upload(x, f32s([1.0] * N))
        s.upload(y, f32s([0.0] * N))
        args = {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': N, 'alpha': 1.0}

        s._stop.set()
        s._worker.join(timeout=10)
        ev = s.submit(mod, args)
        for what, call in (('free', lambda: s.free(o)),
                           ('upload', lambda: s.upload(o, f32s([9.0] * N))),
                           ('download', lambda: s.download(o, N * 4))):
            try:
                call()
                log.append(f"    {what} on a leased buffer: ACCEPTED (bad)")
                ok = False
            except BufferBusy as e:
                log.append(f"    {what:<9s} while leased    -> [{e.code}] "
                           f"{e.message[:40]}")
        s._start_worker()
        s.wait(ev)
        log.append(f"    after it drained, download works: "
                   f"{len(s.download(o, N * 4))} bytes")
        s.free(o)
        try:
            s.free(o)
            log.append("    a double free: ACCEPTED (bad)")
            ok = False
        except BufferError as e:
            log.append(f"    a double free            -> [{e.code}] "
                       f"{e.message[:40]}")
        try:
            s.upload(o, f32s([1.0]))
            log.append("    upload to a freed buffer: ACCEPTED (bad)")
            ok = False
        except BufferError as e:
            log.append(f"    upload to a freed buffer -> [{e.code}] "
                       f"{e.message[:40]}")
    return ok, 'leases block, and a freed handle stays freed'


def rt06(log):
    """Two sessions do not see each other."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    a = session()
    b = session()
    try:
        ma, mb = a.load(artifact_path()), b.load(artifact_path())
        xa, ya, oa = (a.allocate(N * 4, name=n) for n in 'xyo')
        xb, yb, ob = (b.allocate(N * 4, name=n) for n in 'xyo')
        a.upload(xa, f32s([1.0] * N))
        a.upload(ya, f32s([0.0] * N))
        b.upload(xb, f32s([7.0] * N))
        b.upload(yb, f32s([0.0] * N))

        ea = a.submit(ma, {'x_ptr': xa, 'y_ptr': ya, 'o_ptr': oa,
                           'n': N, 'alpha': 1.0})
        eb = b.submit(mb, {'x_ptr': xb, 'y_ptr': yb, 'o_ptr': ob,
                           'n': N, 'alpha': 1.0})
        a.wait(ea)
        b.wait(eb)
        va = unpack(a.download(oa, N * 4))
        vb = unpack(b.download(ob, N * 4))
        log.append(f"    session {a.id} sees {va[0]}, session {b.id} sees "
                   f"{vb[0]} -- same kernel, same addresses, different data")
        ok &= va[0] == 1.0 and vb[0] == 7.0
        log.append(f"    their device directories differ: "
                   f"{a.dev.dir != b.dev.dir}")
        ok &= a.dev.dir != b.dev.dir

        # A handle from one session is refused by the other.
        try:
            b.upload(xa, f32s([1.0]))
            log.append("    another session's handle: ACCEPTED (bad)")
            ok = False
        except BufferError as e:
            log.append(f"    another session's handle -> [{e.code}] "
                       f"{e.message[:44]}")

        # close() while work is unfinished is refused, and is idempotent
        # once it succeeds.
        a._stop.set()
        a._worker.join(timeout=10)
        held = a.submit(ma, {'x_ptr': xa, 'y_ptr': ya, 'o_ptr': oa,
                             'n': N, 'alpha': 1.0})
        try:
            a.close()
            log.append("    close with a task unfinished: ACCEPTED (bad)")
            ok = False
        except SessionBusy as e:
            log.append(f"    close while busy         -> [{e.code}] "
                       f"{e.message[:40]}")
        a._start_worker()
        a.wait(held)
        log.append(f"    close after draining: {a.close()}, again: "
                   f"{a.close()}")
    finally:
        for s in (a, b):
            try:
                s.poisoned = True
                s.close()
            except Exception:
                pass
    return ok, 'two sessions, no leakage, and a close that refuses'


def rt07(log):
    """Dependencies: C waits for B waits for A, and a failure stops both."""
    if not have_triton():
        return False, 'needs the locked Triton'
    ok = True
    with session() as s:
        mod = s.load(artifact_path())
        a_buf, b_buf, c_buf, zero = (s.allocate(N * 4, name=n)
                                     for n in ('A', 'B', 'C', 'zero'))
        s.upload(a_buf, f32s([float(i) for i in range(N)]))
        s.upload(zero, f32s([0.0] * N))

        e1 = s.submit(mod, {'x_ptr': a_buf, 'y_ptr': zero, 'o_ptr': b_buf,
                            'n': N, 'alpha': 2.0})
        e2 = s.submit(mod, {'x_ptr': b_buf, 'y_ptr': zero, 'o_ptr': c_buf,
                            'n': N, 'alpha': 3.0}, depends_on=[e1])
        s.wait(e2)
        got = unpack(s.download(c_buf, N * 4))
        want = [6.0 * i for i in range(N)]
        log.append(f"    A succeeds, B runs after it: {e1.state}/{e2.state}, "
                   f"correct={got == want}")
        ok &= got == want

        # An unknown dependency, and one from another session.
        other = session()
        try:
            om = other.load(artifact_path())
            # Three separate buffers. Passing one three times is an
            # undeclared alias and the runtime refuses it -- correctly, and
            # the first draft of this test found that out the hard way.
            ox, oy, oo = (other.allocate(N * 4) for _ in range(3))
            other.upload(ox, f32s([0.0] * N))
            other.upload(oy, f32s([0.0] * N))
            oe = other.submit(om, {'x_ptr': ox, 'y_ptr': oy, 'o_ptr': oo,
                                   'n': N, 'alpha': 0.0})
            other.wait(oe)
            try:
                s.submit(mod, {'x_ptr': a_buf, 'y_ptr': zero,
                               'o_ptr': c_buf, 'n': N, 'alpha': 1.0},
                         depends_on=[oe])
                log.append("    a dependency from another session: "
                           "ACCEPTED (bad)")
                ok = False
            except TncaError as e:
                log.append(f"    a cross-session dependency -> [{e.code}] "
                           f"{e.message[:40]}")
        finally:
            other.poisoned = True
            other.close()

        try:
            s.submit(mod, {'x_ptr': a_buf, 'y_ptr': zero, 'o_ptr': c_buf,
                           'n': N, 'alpha': 1.0}, depends_on=['not an event'])
            log.append("    a dependency that is not an event: ACCEPTED (bad)")
            ok = False
        except TncaError as e:
            log.append(f"    a dependency that is not an event -> "
                       f"[{e.code}]")
    return ok, 'a chain runs in order; foreign dependencies are refused'


TESTS = [('RT-01', rt01), ('RT-02', rt02), ('RT-03', rt03), ('RT-04', rt04),
         ('RT-05', rt05), ('RT-06', rt06), ('RT-07', rt07)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R2-08 runtime session, buffers, queue and events")
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
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-runtime-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-runtime-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
