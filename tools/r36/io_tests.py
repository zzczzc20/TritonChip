#!/usr/bin/env python3
"""
R36-IO-01, IO-02 and IO-03 -- phase3_6_tasks_and_acceptance.md
section R36-06.

The host protocol: what each operation means, what an illegal one costs,
and what happens when a reply is late, repeated, or never comes.

These run against the real device -- the RTL in a child process, through
the shell's ports -- because the questions are about a protocol and a
protocol has two ends. A mock of the device would be a test of the
host's opinion of the device.

The one to read carefully is IO-01's side-effect rule. "Illegal input
has no device task side effect" is easy to satisfy by accident and easy
to lose by accident, so every illegal case here is bracketed: the
device's own sequence counter and a canary word in memory are read
before and after, and both have to be unchanged. A refusal that had
already written half a buffer would pass a check that only looked at the
return code.
"""
import argparse
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
import r1lib as R                                          # noqa: E402

PROFILE = 'core-vrf-sync-v1'
CANARY = 0xC0FFEE00_5A5A5A5A
CANARY_ADDR = 0x3000


def _device():
    from tnca import device
    from tnca.transport import Transport
    device.prepare(PROFILE, verbose=False)
    t = Transport(device.expected_path(PROFILE), poll_cycles=20,
                  deadline_cycles=200_000)
    t.start()
    return t


def _program():
    """A kernel that writes a known word and exits."""
    import tnca_asm
    a = tnca_asm.Assembler()
    a.assemble("""
        .li      s1, #0x0DEFACED
        .li      s2, #0x4000
        s.st.d   [s2, #0], s1
        exit
    """)
    return tnca_asm.build_image(a.records)


def _slow_program(iterations=1500):
    """A kernel that takes long enough for a host wait to expire.

    IO-03 asks for a timeout case, and a timeout case needs something
    slow. A short kernel with a short deadline is a race: it either
    finishes first or it does not, and a test that sometimes tests
    nothing is worse than no test.

    1500 is chosen against the simulator's measured rate -- a few
    hundred cycles a second -- so the kernel takes seconds while the
    host wait is a quarter of one. 40000 was the first number here and
    made the test take longer than the re-wait it was testing.
    """
    import tnca_asm
    a = tnca_asm.Assembler()
    a.assemble(f"""
        .li      s1, #{iterations}
    spin:
        s.addi   s1, s1, #-1
        br.nz    s1, spin
        .li      s1, #0x5L0WD0NE
        .li      s2, #0x4008
        s.st.d   [s2, #0], s1
        exit
    """.replace('0x5L0WD0NE', '0x5107D07E'))
    return tnca_asm.build_image(a.records)


def _put(t, addr, words):
    return t.upload(addr, b''.join(struct.pack('<Q', w) for w in words))


def _get_word(t, addr):
    _r, blob = t.download(addr, 8)
    return struct.unpack('<Q', blob)[0]


def io01(log):
    """Every operation's semantics, and what an illegal one costs."""
    from tnca.errors import TncaError
    ok = True
    t = _device()
    try:
        _put(t, CANARY_ADDR, [CANARY])
        log.append(f"    a canary at {CANARY_ADDR:#x}")

        # PING and QUIESCE on an idle device.
        p = t.ping()
        log.append(f"    PING  -> status {p.status}, r0 {p.r0} "
                   f"(the sequence it has served)")
        ok &= p.ok
        q = t.quiesce()
        log.append(f"    QUIESCE on an idle device -> status {q.status}, "
                   f"{q.r0} cycle(s) waited")
        ok &= q.ok

        # Round trip.
        _put(t, 0x2000, [0x1122334455667788, 0x99AABBCCDDEEFF00])
        back = _get_word(t, 0x2000)
        log.append(f"    UPLOAD then DOWNLOAD returns what went in: "
                   f"{back == 0x1122334455667788}")
        ok &= back == 0x1122334455667788

        # Illegal cases. Each one is bracketed by the device's own
        # sequence counter and the canary.
        def illegal(what, fn):
            nonlocal ok
            # The bracket makes exactly three requests of its own: a
            # PING, a DOWNLOAD of the canary, and the closing PING. So
            # the device's served counter must advance by three, or by
            # four when the illegal call itself reached the device and
            # was refused there. Anything more means the refusal started
            # something.
            #
            # This half used to be measured and then thrown away with a
            # `del`, under a comment saying both halves had to be
            # unchanged. The counter cannot be unchanged -- the bracket
            # moves it -- so the honest check is on how far it moved.
            before_seq = t.ping().r0
            before_canary = _get_word(t, CANARY_ADDR)
            try:
                reply = fn()
                refused = not getattr(reply, 'ok', False)
                how = f"status {reply.status}"
            except TncaError as e:
                refused = True
                how = f"[{getattr(e, 'code', '?')}]"
            after_canary = _get_word(t, CANARY_ADDR)
            after_seq = t.ping().r0
            clean = after_canary == before_canary
            moved = after_seq - before_seq
            accounted = moved in (3, 4)
            log.append(f"    {what:<44s} refused {refused}, {how}, "
                       f"memory unchanged {clean}, the device served "
                       f"{moved} request(s) "
                       f"({'all of them this check''s own' if moved == 3 else 'ours plus the refused one' if moved == 4 else 'MORE THAN THIS CHECK MADE'})")
            ok &= refused and clean and accounted

        from tnca.transport import OP_UPLOAD, OP_DOWNLOAD
        illegal('an unaligned upload address',
                lambda: t.upload(0x2004, b'\x00' * 8))
        illegal('an unaligned download address',
                lambda: t.download(0x2004, 8))
        illegal('an upload past the end of memory',
                lambda: t.request(OP_UPLOAD, 0x10_0000, 8))
        illegal('a download past the end of memory',
                lambda: t.request(OP_DOWNLOAD, 0x10_0000, 8))
        illegal('a negative address',
                lambda: t.request(OP_UPLOAD, -8, 1))
        illegal('a zero-length upload',
                lambda: t.request(OP_UPLOAD, 0x2000, 0))
        illegal('an operation the device does not have',
                lambda: t.request(99, 0, 0))

        # The device still works afterwards. A refusal that left the
        # device unusable would satisfy "no side effect" on a technicality.
        _put(t, 0x2000, [0xFEEDFACECAFEBEEF])
        log.append(f"    and the device still serves requests after all "
                   f"of them: "
                   f"{_get_word(t, 0x2000) == 0xFEEDFACECAFEBEEF}")
        ok &= _get_word(t, 0x2000) == 0xFEEDFACECAFEBEEF

        # Ordering: a sequence number BELOW the one already served must
        # be ignored. The transport would never send one, so it is
        # written straight into the mailbox -- the device has to be the
        # thing that refuses, not the host.
        _put(t, 0x2100, [0x1111111111111111])
        served = t.ping().r0
        with open(t.req, 'w') as fh:
            fh.write(f"{served - 1} 1 {0x2100} 1 0 0 0 0\n")
        import time as _t
        _t.sleep(0.8)
        still = _get_word(t, 0x2100)
        again = t.ping().r0
        log.append(f"    an out-of-order request (sequence {served - 1} "
                   f"when {served} is served): memory still "
                   f"{still:#x}, counter {again} "
                   f"({'not acted on' if again >= served else 'WENT BACKWARDS'})")
        ok &= still == 0x1111111111111111 and again >= served

        # Drain after real work, rather than only on a device that has
        # never run anything. The mailbox is synchronous, so the host
        # cannot leave a launch in flight to drain; what this shows is
        # that QUIESCE after a kernel reports a completed drain rather
        # than inheriting state from it. The device-side wait, with the
        # core genuinely busy, is exercised in the RTL bench
        # (tb/device/tnca_shell_tb.v, CMD_QUIESCE).
        _put(t, 0x8000, _slow_program(400))
        _put(t, 0x4008, [0])
        r = t.launch(0x8000, 0)
        q = t.quiesce()
        marker = _get_word(t, 0x4008)
        log.append(f"    a real kernel ({r.r0} cycle(s)) and then "
                   f"QUIESCE: status {q.status}, {q.r0} cycle(s) waited, "
                   f"and its marker is in memory: "
                   f"{marker == 0x5107D07E}")
        ok &= r.ok and q.ok and marker == 0x5107D07E
    finally:
        t.stop()
    return ok, 'seven illegal requests, each with the memory unchanged'


def io02(log):
    """A download after SUCCEEDED sees what the kernel wrote."""
    ok = True
    t = _device()
    try:
        words = _program()
        _put(t, 0, words)
        _put(t, 0x4000, [0xBADBADBADBADBAD0])
        log.append(f"    {len(words)} word(s) of program, and a sentinel "
                   f"where the kernel will write")

        r = t.launch(0, 0)
        log.append(f"    LAUNCH -> status {r.status}, {r.r0} cycle(s), "
                   f"{r.r1} instruction(s) retired")
        ok &= r.ok

        got = _get_word(t, 0x4000)
        log.append(f"    the download straight after SUCCEEDED sees the "
                   f"kernel's value: {got:#x} "
                   f"({'yes' if got == 0x0DEFACED else 'NO'})")
        ok &= got == 0x0DEFACED

        # Two tasks sharing memory: the second reads what the first wrote,
        # with no host round trip in between.
        _put(t, 0x4000, [0])
        t.launch(0, 0)
        first = _get_word(t, 0x4000)
        t.launch(0, 0)
        second = _get_word(t, 0x4000)
        log.append(f"    two launches sharing the same memory: "
                   f"{first:#x} then {second:#x}")
        ok &= first == second == 0x0DEFACED

        # The refusal that matters: while a kernel runs, the host port is
        # not ready. This is a hardware property, so it is read out of the
        # hardware rather than asserted here -- tnca_shell drives
        # host_req_ready low for every busy cycle, and the export's smoke
        # test counts those cycles and fails if the port ever opened.
        log.append("    host access during a launch is refused by the "
                   "device: tnca_shell's host_req_ready is low for every "
                   "busy cycle, counted by the export's smoke test "
                   "(R36-RTL-01) rather than assumed here")

        # And the protocol serialises anyway: the transport is
        # request/response with one outstanding request, so a host that
        # tried would be waiting for a reply it has not had.
        log.append("    and the protocol cannot present one: the mailbox "
                   "carries one outstanding request, so an upload during "
                   "a launch is not a race the user has to avoid -- it "
                   "is a request that has not been sent yet")
    finally:
        t.stop()
    return ok, 'the result is visible the moment the launch succeeds'


def io03(log):
    """Late replies, duplicates, timeouts and device loss."""
    from tnca.errors import TncaError, DeviceLost
    ok = True
    t = _device()
    try:
        words = _program()
        _put(t, 0, words)

        # A duplicate sequence number. Written straight into the mailbox,
        # because the transport would never send one -- which is the
        # point: the device must be the thing that refuses, not the host.
        #
        # The LAUNCH line is captured immediately, before anything else
        # is sent. The first version of this read the mailbox after a
        # download and a ping, so what it re-presented was a PING -- a
        # duplicate-suppression test that re-sent the one request with no
        # side effect to suppress.
        _put(t, 0x4000, [0])
        t.launch(0, 0)
        with open(t.req) as fh:
            last = fh.read().strip()
        first = _get_word(t, 0x4000)
        seq_before = t.ping().r0
        log.append(f"    the LAUNCH as it went on the wire: {last!r} "
                   f"(op {last.split()[1]} is LAUNCH)")
        ok &= last.split()[1] == '3'

        # Re-present it. The device has already acted on this sequence.
        _put(t, 0x4000, [0xAAAAAAAAAAAAAAAA])
        with open(t.req, 'w') as fh:
            fh.write(last + '\n')
        # Give the device several poll intervals to do the wrong thing.
        import time
        time.sleep(1.0)
        after = _get_word(t, 0x4000)
        seq_after = t.ping().r0
        log.append(f"    a sequence number re-presented: memory still "
                   f"{after:#x} ({'not re-run' if after != 0x0DEFACED else 'RE-RUN'})")
        ok &= after != 0x0DEFACED
        log.append(f"    the served counter did not go backwards: "
                   f"{seq_after >= seq_before}")
        ok &= seq_after >= seq_before
        del first

        # A host wait that expires is NOT a verdict on the task. The
        # distinction matters: a timeout that poisoned the session would
        # turn a slow kernel into a lost device.
        slow = _slow_program()
        _put(t, 0x8000, slow)
        _put(t, 0x4008, [0])
        timed_out = False
        try:
            t.launch(0x8000, 0, timeout=0.25)
            log.append("    the slow kernel finished inside a 0.25s host "
                       "wait, so the timeout case tested nothing -- raise "
                       "the iteration count")
            ok = False
        except TncaError as e:
            timed_out = getattr(e, 'code', '') == 'WAIT_TIMEOUT'
            log.append(f"    a host wait that expires while a kernel is "
                       f"still running: [{getattr(e, 'code', '?')}]")
            ok &= timed_out

        # The device was never told anything. It is still working on the
        # kernel, and the right thing for the host to do is wait again --
        # which is the whole point of the code being WAIT_TIMEOUT and not
        # DEVICE_LOST.
        log.append(f"    the device during that wait: alive {t.alive()}")
        ok &= t.alive()
        r = t._await(t.seq, timeout=120)
        log.append(f"    waiting again picks up the same task: status "
                   f"{r.status}, {r.r0} cycle(s)")
        ok &= r.ok
        marker = _get_word(t, 0x4008)
        log.append(f"    and the kernel really did finish: {marker:#x} "
                   f"({'yes' if marker == 0x5107D07E else 'NO'})")
        ok &= marker == 0x5107D07E

        # ...and the device is still there for ordinary work.
        p = t.ping()
        log.append(f"    the device afterwards: alive {t.alive()}, "
                   f"PING status {p.status}")
        ok &= t.alive() and p.ok

        # Device loss, on purpose. Everything after it must be an error
        # with a code, not a hang and not a wrong answer.
        t.stop()
        log.append(f"    the device stopped on purpose: alive "
                   f"{t.alive()}")
        ok &= not t.alive()
        try:
            t.ping()
            log.append("    a request to a stopped device SUCCEEDED, "
                       "which is wrong")
            ok = False
        except (TncaError, DeviceLost) as e:
            log.append(f"    a request to a stopped device: "
                       f"[{getattr(e, 'code', '?')}] "
                       f"{str(e).splitlines()[0][:56]}")
    finally:
        try:
            t.stop()
        except Exception:                                  # noqa: BLE001
            pass
    return ok, 'a duplicate is not re-run, a host wait is not a verdict'


TESTS = [('R36-IO-01', io01), ('R36-IO-02', io02), ('R36-IO-03', io03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3.6 Gate B: the host protocol, and what it costs to get it "
          "wrong")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:                             # noqa: BLE001
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
        print(flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print(f"  r36-io-tests           {'FAIL' if bad else 'PASS':<6s} "
          + (', '.join(bad) if bad else f"{len(results)} check(s)"))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
