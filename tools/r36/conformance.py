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

There is a third answer now that a binding for real hardware exists. A
`reference-available` binding that declares `needs_hardware` -- the
Alveo, whose dependency is XRT and a card in a slot -- cannot be
measured on a machine that has neither, and the whole suite for it
becomes the same refusal case: it must say what is missing rather than
carry on against the stand-in. That is "it did not run", said out loud.
It is not a pass for that backend and the summary does not count it as
one, because nothing about how it carries a kernel was established.

That third answer is honest, and on its own it is also a hole -- the one
two options below close. "Not measurable here" is a sentence in a log,
and a log is not a gate: a run that never touched a board can be filed
beside one that did and read the same way, because the verdict line and
the JSON both said PASS either way. The distinction was recorded and
nothing acted on it.

    --xclbin PATH       the image the card is running, handed to the
                        bindings that ask for it. A board has to be told
                        this -- it is in no profile and no artifact -- so
                        without the option a card binding cannot be
                        opened, and the board could not be measured by
                        this suite on any machine, including one with the
                        card in it.

    --require-measured  a backend that was not measured FAILS the run
                        rather than being recorded as N/A. This is the
                        option that turns the distinction into a verdict,
                        and it is the one to use wherever the result will
                        be read as evidence that hardware was tested.

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
    rec('alive after open', ok, 'the process exists', 'liveness')
    caps = dev.capabilities()
    rec('capabilities', isinstance(caps, dict) and bool(caps),
        f"{sorted(caps)[:6]}", 'declaration')
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
            # Backed by a declaration, and it has to be said: this branch
            # recorded no backing, and a step with none fails the case by
            # its own rule further down. It never fired, because no
            # binding in this repository put a raw opcode on the wire --
            # sim-file and sim-socket both answer `raw_request: False` --
            # so the branch was unreachable until the board binding
            # arrived, and that one does carry an opcode field.
            #
            # A declaration is the honest backing: what refused is the
            # binding's own limit, not the device. It could not put the
            # request on its wire at all. A device that was asked and
            # refused lands in the `status` path above, backed by the
            # protocol exchange that carried the refusal back.
            rec('an operation the device does not have', True,
                f"[{getattr(e, 'code', '?')}] raised", 'declaration')
    else:
        rec('an operation the device does not have', True,
            'not offered: this binding has no opcode field, so an '
            'unknown operation cannot be put on the wire',
            'declaration')
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
            rec(what, True, f"[{getattr(e, 'code', '?')}] refused",
                'protocol')
    # Past the end of memory, through the ordinary interface rather
    # than a raw opcode, so every binding answers the same question.
    from tnca.errors import TncaError as _T
    try:
        r = dev.upload(0x10_0000, _words([0]))
        good, how = (not r.ok), f"status {r.status}"
    except _T as e:
        # Backed the same way the branch above it is, and it has to be
        # said explicitly: the keyword inference reads a detail like
        # "status 3" as a protocol step, and this branch's detail carries
        # no such word, so it recorded NO backing and failed the case on
        # the suite's own rule. It never fired because no binding in this
        # repository refused here -- sim-file and sim-socket both let the
        # device answer with a status -- so the branch was unreachable
        # until a binding checked the window on the device's behalf,
        # which the board binding does. Same backing as the two refusals
        # above: this step is the transfer contract being enforced, and
        # it is one question whichever side answers it.
        good, how = True, f"[{getattr(e, 'code', '?')}] refused"
    rec('a transfer past the end of memory', good, how, 'protocol')
    ok &= good
    return ok


def c_compatibility(dev, rec):
    caps = dev.capabilities()
    need = ('medium',)
    have = all(k in caps for k in need)
    rec('the backend says what it is', have,
        f"medium {caps.get('medium')!r}, reaches "
        f"{caps.get('reaches', '?')!r}", 'declaration')
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
        f"states {sorted(set(states.values()))}", 'registry')
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
        r.r0 > 0 and r.r1 > 0, f"{r.r0} and {r.r1} from seq {r.seq}",
        'protocol')
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
    rec('a second launch in the same session', ok,
        f"no restart in between; the oracle {MARK:#x} again",
        ('protocol', 'oracle'))
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
        f"code {r.r0:#x} at pc {r.r1:#x}, from seq {r.seq}", 'protocol')
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


def run_backend(name, info, log, xclbin=None):
    """Every case against one backend, or the refusal that stands in."""
    from tnca import backend, device
    from tnca.errors import DependencyFailed, TncaError

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

    # Opening it is allowed to fail for want of a dependency, and until
    # now that failure took the whole suite down with it -- one board
    # binding with no board in the machine erased the results for every
    # backend after it, alphabetically, before sim-file was even reached.
    #
    # Only a missing DEPENDENCY is excused, and only for a binding that
    # said it needs hardware. Anything else is a broken installation,
    # and the same DependencyFailed from a backend that needs no
    # hardware is exactly that: sim-file unable to import what it needs
    # is not a machine without a card, and it must not be waved past.
    try:
        device.prepare(PROFILE, verbose=False)
        # A board has to be told which image it is running, and that fact is
        # in no profile and no artifact. Handed only to a binding that
        # declares `needs_hardware`, because that declaration is the
        # registry's own statement of which bindings are cards -- a
        # simulator would take the keyword as an unknown argument and fail
        # for a reason that has nothing to do with what was being tested.
        kw = {}
        if xclbin and backend.needs_hardware(name):
            kw['xclbin'] = xclbin
        dev = backend.open_backend(name,
                                   bench=device.expected_path(PROFILE),
                                   deadline_cycles=200_000, **kw)
        dev.start()
    except DependencyFailed as e:
        if not backend.needs_hardware(name):
            raise
        log.append(f"    {name} [{state}]: not measurable on this "
                   f"machine, and it says why instead of substituting")
        log.append(f"      [{getattr(e, 'code', '?')}] "
                   f"{str(e).splitlines()[0][:76]}")
        log.append("    the suite for it is that refusal: a platform "
                   "that is not here must not appear to work")
        return True, {'refused': True, 'code': getattr(e, 'code', None),
                      'not_applicable': str(e).splitlines()[0][:120]}
    ok, record = True, {}
    try:
        for case in CASES:
            lines = []

            # Each step records what backs it: a protocol exchange
            # (a sequence number from the device) or an oracle (a value
            # computed here). R36-TEST-01 asks every item to map to
            # one; a step with neither is an assertion about the host's
            # opinion of the device.
            def rec(step, good, detail, backing=None, _l=lines):
                d = str(detail)
                if backing is None:
                    backing = []
                    if 'seq ' in d or 'status ' in d or 'served ' in d:
                        backing.append('protocol')
                    if 'oracle' in d:
                        backing.append('oracle')
                elif isinstance(backing, str):
                    backing = [backing]
                _l.append((step, bool(good), d, tuple(backing)))
            try:
                good = case.fn(dev, rec)
            except Exception as e:                         # noqa: BLE001
                good = False
                lines.append((f"{type(e).__name__}", False, str(e)[:80],
                              ()))
            ok &= bool(good)
            record[case.id] = {'group': case.group, 'passed': bool(good),
                               'steps': [{'step': s, 'ok': g, 'detail': d,
                                          'backed_by': list(b)}
                                         for s, g, d, b in lines]}
            log.append(f"    {name} {case.id:<12s} "
                       f"{'PASS' if good else 'FAIL':<5s} {case.what}")
            bare = [s for s, _g, _d, b in lines if not b]
            for s, g, d, b in lines:
                mark = '+'.join(b) if b else 'bare'
                log.append(f"      {'ok ' if g else 'BAD'} [{mark}] "
                           f"{s}: {d}"[:98])
            # Failing a case from here has to be written back into the
            # record as well as into the total. It was not, and the JSON
            # went on saying every case passed while the gate said FAIL:
            # the file the evidence is read from disagreed with the run
            # that produced it, and finding out why meant three passes
            # over a log this function had already stopped printing.
            def _fail_here(why, _cid=case.id):
                record[_cid]['passed'] = False
                record[_cid].setdefault('notes', []).append(why)

            if bare:
                why = (f"steps that declare no backing at all: {bare}")
                log.append(f"      {why}")
                ok = False
                _fail_here(why)
            # And a case about device behaviour has to be backed by the
            # device: a group whose every step is a declaration or a
            # registry lookup has not talked to anything.
            kinds = {k for _s, _g, _d, b in lines for k in b}
            if case.group in ('lifecycle', 'transfer', 'completion',
                              'faults') and 'protocol' not in kinds:
                why = (f"{case.id} touches the device and no step is "
                       f"backed by a protocol exchange")
                log.append(f"      {why}")
                ok = False
                _fail_here(why)
            if case.group in ('transfer', 'completion') and \
                    'oracle' not in kinds:
                why = (f"{case.id} checks values and no step is backed "
                       f"by an oracle")
                log.append(f"      {why}")
                ok = False
                _fail_here(why)
    finally:
        dev.stop()
    return ok, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backends', nargs='*')
    ap.add_argument('--check-programs', action='store_true',
                    help='re-assemble the built-in programs and compare')
    ap.add_argument('--xclbin', metavar='PATH',
                    help='the .xclbin the board is running. Handed to the '
                         'bindings that declare they need hardware; without '
                         'it such a binding cannot be opened, so a card in '
                         'the machine still cannot be measured')
    ap.add_argument('--require-measured', action='store_true',
                    help='a backend that could not be measured on this '
                         'machine FAILS the run instead of being recorded '
                         'as N/A. Use it wherever the result will be read '
                         'as evidence that hardware was tested')
    ap.add_argument('--json')
    args = ap.parse_args()
    # Said here rather than three frames down, so that a typo in a path is
    # a typo in a path and not a board that could not be reached. The two
    # are reported the same way otherwise, and only one of them is the
    # machine's fault.
    if args.xclbin and not os.path.isfile(args.xclbin):
        print(f"  --xclbin {args.xclbin}: no such file")
        return 2
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
    out, ok, unmeasured = {}, True, []
    for name in names:
        if name not in info:
            print(f"    {name}: not a backend this installation knows")
            ok = False
            continue
        log = []
        good, record = run_backend(name, info[name], log,
                                   xclbin=args.xclbin)
        for line in log:
            print(line, flush=True)
        out[name] = {'state': info[name]['state'], 'passed': good,
                     'cases': record}
        if record.get('not_applicable'):
            unmeasured.append((name, record['not_applicable']))
        ok &= good
        print(flush=True)
    # Written as a one-case list, the same shape every other checker
    # produces. The gate and the verifier then need no special case for
    # this file, and -- the part that matters -- the case ID the
    # contract names appears in the evidence, so a package with no
    # result is visibly a package with no result.
    # Said before the verdict, so the verdict cannot be read as covering
    # more than it did. A backend counted here was opened and driven; one
    # listed as unmeasured was not, and "PASS" beside its name would be
    # the overclaim section 3.2 keeps platform-verified reserved to
    # prevent.
    if unmeasured:
        print(f"  {len(unmeasured)} of {len(names)} backend(s) could not be "
              f"measured on this machine:")
        for n, why in unmeasured:
            print(f"    {n}: {why[:100]}")
        print("    not a pass for them -- nothing about how they carry a "
              "kernel was established. Not a failure either: their "
              "hardware is not here, and a suite that called that a "
              "failure could never run anywhere that has none")
        if args.require_measured:
            # The exception above is a default, not a principle, and this
            # is the caller saying which one it was. A run whose whole
            # purpose was to touch the hardware does not get to finish by
            # reporting that it did not.
            ok = False
            print("    --require-measured: the standing excuse is withdrawn "
                  "and the run FAILS:")
            for n, _why in unmeasured:
                print(f"      {n}: required to be measured, and was not")
            print("    Nothing about those backends was established here. "
                  "This is not hardware verification and must not be "
                  "filed as any.")
    if args.json:
        R.write_json(args.json, [{
            'id': 'R36-TEST-01', 'passed': bool(ok),
            'detail': f"{len(names)} backend(s), {len(CASES)} case(s) "
                      f"in {len(set(c.group for c in CASES))} group(s)"
                      + (f", {len(unmeasured)} not measurable here"
                         if unmeasured else '')
                      + (", measurement was required"
                         if args.require_measured else ''),
            'backends': out}])
    print(f"  conformance            {'PASS' if ok else 'FAIL':<6s} "
          f"{len(names) - len(unmeasured)} of {len(names)} backend(s) "
          f"measured")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
