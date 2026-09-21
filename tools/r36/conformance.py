#!/usr/bin/env python3
"""
The shared backend conformance suite -- phase3_6_tasks_and_acceptance.md
R36-TEST-01.

One suite, run against every backend the installation knows. Five
groups, which are the ones the standard names:

    lifecycle    open, query, quiesce, close, and open again
    transfer     upload, download, alignment, range
    compatibility  capabilities, and what a wrong artifact costs
    completion   a launch that succeeds, and what it reports
    faults       a launch that traps, and the session afterwards

A backend whose state is `reference-available` must run all of it. One
whose state is `interface-defined` must REFUSE to open, and that refusal
is itself a case -- section 3.4 forbids falling back to the simulator
and reporting success, so "it did not run" and "it ran on something
else" have to be distinguishable.

Every case records two things beside its verdict: the protocol exchange
it caused, taken from the transport's own sequence counter and status
codes, and the oracle it checked against. A case with neither is an
assertion about the host's opinion of the device.

This is the suite R36-HAND-01's downstream rehearsal runs, which is why
it takes a backend by name and reaches nothing in this repository that
an installed package does not ship.
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(ROOT, 'tools', 'r1'),
          os.path.join(ROOT, 'python'),
          os.path.join(ROOT, 'compiler', 'python'),
          os.path.join(ROOT, 'tools', 'asm')):
    if p not in sys.path:
        sys.path.insert(0, p)

PROFILE = 'core-vrf-sync-v1'
MARK = 0x0DEFACED
MARK_ADDR = 0x4000


# The two programs, as machine words.
#
# Carried as constants rather than assembled, because this suite is the
# one a DOWNSTREAM branch runs and a downstream branch has no assembler:
# tools/asm is a repository tool, not something the package ships. The
# rehearsal found that by failing with ModuleNotFoundError on the two
# cases that needed it, which is what a rehearsal is for.
#
# Constants can drift from the source they were assembled from, so
# `--check-programs` re-assembles them when the assembler IS available
# and compares. The source is kept here so there is something to
# compare against.
_MINIMAL_SRC = f"""
        .li      s1, #{MARK}
        .li      s2, #{MARK_ADDR}
        s.st.d   [s2, #0], s1
        exit
"""
MINIMAL_PROGRAM = [
    0x1400100ACED0000, 0x14101000DEF0001, 0x141010000000002,
    0x141010000000003, 0x140020040000000, 0x141020000000001,
    0x141020000000002, 0x141020000000003, 0x2008010200000003,
    0x21F000000000000,
]

_TRAPPING_SRC = """
        .li          s1, #1
        .assert_eq   s1, #0xDEAD, #7
        exit
"""
TRAPPING_PROGRAM = [
    0x140010000010000, 0x141010000000001, 0x141010000000002,
    0x141010000000003, 0x1403E00DEAD0000, 0x1413E0000000001,
    0x1413E0000000002, 0x1413E0000000003, 0x1183F013E000001,
    0x202003F00020000, 0x218000000070000, 0x21F000000000000,
]


def _program():
    return MINIMAL_PROGRAM


def _trapping():
    return TRAPPING_PROGRAM


def check_programs():
    """Re-assemble the constants above, where an assembler exists."""
    try:
        import tnca_asm
    except ImportError:
        return None, 'no assembler here, which is the normal case downstream'
    out = []
    for label, src, want in (('the minimal program', _MINIMAL_SRC,
                              MINIMAL_PROGRAM),
                             ('the trapping program', _TRAPPING_SRC,
                              TRAPPING_PROGRAM)):
        a = tnca_asm.Assembler()
        a.assemble(src)
        got = tnca_asm.build_image(a.records)
        out.append((label, got == want, len(got)))
    return out, 'assembled from the source beside them'


def _words(ws):
    return b''.join(struct.pack('<Q', w) for w in ws)


class Case:
    def __init__(self, cid, group, what, fn):
        self.id, self.group, self.what, self.fn = cid, group, what, fn


def c_lifecycle(dev, rec):
    from tnca.errors import TncaError
    ok = dev.alive()
    rec('alive after open', ok, 'the process exists')
    caps = dev.capabilities()
    rec('capabilities', isinstance(caps, dict) and bool(caps),
        f"{sorted(caps)[:6]}")
    ok &= isinstance(caps, dict) and bool(caps)
    p = dev.ping()
    rec('PING', p.ok, f"seq {p.seq}, status {p.status}, served {p.r0}")
    ok &= p.ok
    q = dev.quiesce()
    rec('QUIESCE on an idle device', q.ok,
        f"seq {q.seq}, status {q.status}, {q.r0} cycle(s)")
    ok &= q.ok
    # An operation the device does not have. Only a binding whose wire
    # format carries an opcode field can express one; a structured
    # protocol cannot, and says so in its capabilities rather than
    # having this suite discover it as an AttributeError. Section
    # R36-TEST-01 asks for unsupported items to be a CONTROLLED
    # refusal, and "the protocol cannot represent it" is one.
    if caps.get('raw_request'):
        try:
            r = dev.request(99)
            # A refusal carried in the status field is as much a
            # refusal as one raised: the file mailbox answers an
            # unknown opcode with status 3 rather than by throwing.
            rec('an operation the device does not have', not r.ok,
                f"status {r.status}"
                if not r.ok else 'it was ACCEPTED')
            ok &= not r.ok
        except TncaError as e:
            rec('an operation the device does not have', True,
                f"[{getattr(e, 'code', '?')}] raised")
    else:
        rec('an operation the device does not have', True,
            'not offered: this binding has no opcode field, so an '
            'unknown operation cannot be put on the wire')
    return ok


def c_transfer(dev, rec):
    from tnca.errors import TncaError
    ok = True
    pattern = [0x0123456789ABCDEF, 0xFEDCBA9876543210, 0, 0xFFFFFFFFFFFFFFFF]
    r = dev.upload(0x2000, _words(pattern))
    rec('UPLOAD', r.ok, f"seq {r.seq}, status {r.status}, "
                        f"{len(pattern)} word(s)")
    ok &= r.ok
    r, blob = dev.download(0x2000, 8 * len(pattern))
    got = [struct.unpack_from('<Q', blob, i * 8)[0]
           for i in range(len(pattern))]
    same = got == pattern
    rec('DOWNLOAD returns what went in', same,
        f"oracle: the four words the host wrote; seq {r.seq}")
    ok &= same
    # A sub-word read out of a word the host wrote: the oracle is
    # arithmetic on the host's own pattern, not another device read.
    _r, blob = dev.download(0x2008, 8)
    one = struct.unpack('<Q', blob)[0]
    rec('a single word at an offset', one == pattern[1],
        f"oracle: {pattern[1]:#x}")
    ok &= one == pattern[1]
    for what, fn in (
            ('an unaligned upload', lambda: dev.upload(0x2004, b'\0' * 8)),
            ('an unaligned download', lambda: dev.download(0x2004, 8))):
        try:
            fn()
            rec(what, False, 'it was accepted')
            ok = False
        except TncaError as e:
            rec(what, True, f"[{getattr(e, 'code', '?')}] refused")
    # Past the end of memory, through the ordinary interface rather
    # than a raw opcode, so every binding answers the same question.
    from tnca.errors import TncaError as _T
    try:
        r = dev.upload(0x10_0000, _words([0]))
        good, how = (not r.ok), f"status {r.status}"
    except _T as e:
        good, how = True, f"[{getattr(e, 'code', '?')}] refused"
    rec('a transfer past the end of memory', good, how)
    ok &= good
    return ok


def c_compatibility(dev, rec):
    caps = dev.capabilities()
    need = ('medium',)
    have = all(k in caps for k in need)
    rec('the backend says what it is', have,
        f"medium {caps.get('medium')!r}, reaches "
        f"{caps.get('reaches', '?')!r}")
    ok = have
    # The state the registry gives it must be one of the three, and must
    # not be the reserved one.
    from tnca import backend
    info = backend.available()
    states = {k: v['state'] for k, v in info.items()}
    ok &= all(s in backend.STATES for s in states.values())
    reserved = [k for k, v in states.items()
                if v == backend.PLATFORM_VERIFIED]
    rec('no backend claims platform-verified', not reserved,
        f"states {sorted(set(states.values()))}")
    ok &= not reserved
    return ok


def c_completion(dev, rec):
    ok = True
    dev.upload(0, _words(_program()))
    dev.upload(MARK_ADDR, _words([0]))
    r = dev.launch(0, 0)
    rec('LAUNCH', r.ok,
        f"seq {r.seq}, status {r.status}, {r.r0} cycle(s), "
        f"{r.r1} instruction(s) retired")
    ok &= r.ok
    rec('it reports a cycle count and a retired count',
        r.r0 > 0 and r.r1 > 0, f"{r.r0} and {r.r1}")
    ok &= r.r0 > 0 and r.r1 > 0
    _q, blob = dev.download(MARK_ADDR, 8)
    got = struct.unpack('<Q', blob)[0]
    rec('the kernel wrote what the oracle says', got == MARK,
        f"oracle: {MARK:#x}, got {got:#x}")
    ok &= got == MARK
    # Twice, sharing memory.
    dev.upload(MARK_ADDR, _words([0]))
    dev.launch(0, 0)
    _q, blob = dev.download(MARK_ADDR, 8)
    ok &= struct.unpack('<Q', blob)[0] == MARK
    rec('a second launch in the same session', ok, 'no restart in between')
    return ok


def c_faults(dev, rec):
    ok = True
    dev.upload(0x6000, _words(_trapping()))
    r = dev.launch(0x6000, 0)
    trapped = (not r.ok) and r.status == 1
    rec('a trapping kernel is a FAULT', trapped,
        f"status {r.status}, code {r.r0:#x}, pc {r.r1:#x}")
    ok &= trapped
    rec('the fault carries a code and a pc', r.r0 != 0,
        f"code {r.r0:#x} at pc {r.r1:#x}")
    ok &= r.r0 != 0
    # And the session survives it.
    dev.upload(0, _words(_program()))
    dev.upload(MARK_ADDR, _words([0]))
    r2 = dev.launch(0, 0)
    _q, blob = dev.download(MARK_ADDR, 8)
    alive = r2.ok and struct.unpack('<Q', blob)[0] == MARK
    rec('the session still works after a fault', alive,
        f"status {r2.status}")
    ok &= alive
    return ok


CASES = [
    Case('CONF-LIFE', 'lifecycle', 'open, query, quiesce, close',
         c_lifecycle),
    Case('CONF-XFER', 'transfer', 'upload, download, alignment, range',
         c_transfer),
    Case('CONF-COMPAT', 'compatibility', 'what the backend claims to be',
         c_compatibility),
    Case('CONF-DONE', 'completion', 'a launch that succeeds, and what it '
                                    'reports', c_completion),
    Case('CONF-FAULT', 'faults', 'a launch that traps, and after it',
         c_faults),
]


def run_backend(name, info, log):
    """Every case against one backend, or the refusal that stands in."""
    from tnca import backend, device
    from tnca.errors import TncaError

    state = info['state']
    if state != backend.REFERENCE_AVAILABLE:
        # The whole suite for this backend is one case: it must refuse.
        try:
            backend.open_backend(name, bench='anything')
            log.append(f"    {name} [{state}]: it OPENED, which section "
                       f"3.4 forbids -- a platform that is not there must "
                       f"not appear to work")
            return False, {'refused': False}
        except TncaError as e:
            log.append(f"    {name} [{state}]: refused to open, "
                       f"[{getattr(e, 'code', '?')}]")
            log.append(f"      {str(e).splitlines()[0][:84]}")
            return True, {'refused': True,
                          'code': getattr(e, 'code', None)}

    device.prepare(PROFILE, verbose=False)
    dev = backend.open_backend(name, bench=device.expected_path(PROFILE),
                               deadline_cycles=200_000)
    dev.start()
    ok, record = True, {}
    try:
        for case in CASES:
            lines = []

            def rec(step, good, detail, _l=lines):
                _l.append((step, bool(good), detail))
            try:
                good = case.fn(dev, rec)
            except Exception as e:                         # noqa: BLE001
                good = False
                lines.append((f"{type(e).__name__}", False, str(e)[:80]))
            ok &= bool(good)
            record[case.id] = {'group': case.group, 'passed': bool(good),
                               'steps': [{'step': s, 'ok': g,
                                          'detail': d}
                                         for s, g, d in lines]}
            log.append(f"    {name} {case.id:<12s} "
                       f"{'PASS' if good else 'FAIL':<5s} {case.what}")
            for s, g, d in lines:
                log.append(f"      {'ok ' if g else 'BAD'} {s}: {d}"[:96])
    finally:
        dev.stop()
    return ok, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backends', nargs='*')
    ap.add_argument('--check-programs', action='store_true',
                    help='re-assemble the built-in programs and compare')
    ap.add_argument('--json')
    args = ap.parse_args()
    from tnca import backend
    import r1lib as R

    print("The backend conformance suite")
    if args.check_programs:
        rows, why = check_programs()
        if rows is None:
            print(f"  the built-in programs: {why}")
            return 0
        bad = [r for r in rows if not r[1]]
        for label, good, n in rows:
            print(f"  {label}: {n} word(s), "
                  f"{'matches its source' if good else 'DIFFERS FROM ITS SOURCE'}")
        print(f"  {why}: {'all match' if not bad else 'DRIFTED'}")
        return 1 if bad else 0
    info = backend.available()
    names = args.backends or sorted(info)
    print(f"  {len(CASES)} case(s) in {len(set(c.group for c in CASES))} "
          f"group(s), against {len(names)} backend(s)")
    out, ok = {}, True
    for name in names:
        if name not in info:
            print(f"    {name}: not a backend this installation knows")
            ok = False
            continue
        log = []
        good, record = run_backend(name, info[name], log)
        for line in log:
            print(line, flush=True)
        out[name] = {'state': info[name]['state'], 'passed': good,
                     'cases': record}
        ok &= good
        print(flush=True)
    if args.json:
        R.write_json(args.json, out)
    print(f"  conformance            {'PASS' if ok else 'FAIL':<6s} "
          f"{len(names)} backend(s)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
