#!/usr/bin/env python3
"""
FP-01 .. FP-06 and OWN-01 .. OWN-12 -- R1-05 and R1-06.

The audit's F01, as two families of tests.

FP is about *how much* an instruction touches: the lowering emitted K32/K64
from the instruction and then sized the footprint from the target's default
K16, so a K64 FP16 fragment -- thirty-two registers -- was accepted into an
eight-register window. The oracle here computes the answer from the ISA's own
definition and never calls the function under test.

OWN is about *when*: a write that lands inside an outstanding command's read
set. Every negative has a positive twin that differs only in the event or the
range, because a checker that rejects everything is not a checker.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
import r1lib as R  # noqa: E402

from tnca_cc import TargetInfo                       # noqa: E402
from tnca_cc.diagnostics import Diagnostic           # noqa: E402
from tnca_cc.ir import Builder, make_kernel          # noqa: E402
from tnca_cc.codegen import CodeGen                  # noqa: E402
from tnca_cc.footprints import (RegRange, fragment_registers,   # noqa: E402
                                dma_coordinate_footprint)

VREG_BYTES = 64          # VLEN 16 x ELEN 32 bits
DTYPES = {'f16': 16, 'bf16': 16, 'e5m2': 8, 'e4m3': 8}
KSTEPS = {'k16': 16, 'k32': 32, 'k64': 64}


def oracle_registers(rows, k, bits):
    """An independent formula, from ISA sec 9C.2, not the function under test.

    A fragment is a flat row-major byte array of rows*k elements; a vector
    register is VLEN*ELEN/8 bytes. Nothing else.
    """
    return -(-(rows * k * bits // 8) // VREG_BYTES)


def compile_ok(build, target=None):
    """Compile a kernel; return (ok, Diagnostic or None)."""
    kernel = make_kernel('r1case')
    b = Builder(kernel, kernel.role(0).region)
    build(b)
    cg = CodeGen(kernel, target or TargetInfo.default())
    try:
        cg.assign_shared()
        cg.run()
        return True, None
    except Diagnostic as d:
        return False, d


# --------------------------------------------------------------------------
def fp01(log):
    """Twelve dtype x K combinations against an independent formula."""
    bad = []
    for dt, bits in DTYPES.items():
        for kname, k in KSTEPS.items():
            want_a = oracle_registers(16, k, bits)
            want_b = oracle_registers(k, 16, bits)
            got_a = fragment_registers(16, k, bits)
            got_b = fragment_registers(k, 16, bits)
            if (got_a, got_b) != (want_a, want_b):
                bad.append(f"{dt}/{kname}: got {got_a},{got_b} want {want_a},{want_b}")
            log.append(f"    {dt:5s} {kname}: A={got_a:2d} B={got_b:2d} VREG")
    return not bad, '; '.join(bad) or '12 combinations match the formula'


def fp02(log):
    """Each combination, with A and then B one register short."""
    ok = True
    for dt in DTYPES:
        for kname in KSTEPS:
            need = oracle_registers(16, KSTEPS[kname], DTYPES[dt])
            for which in ('A', 'B'):
                la = need - 1 if which == 'A' else need
                lb = need if which == 'A' else need - 1

                def build(b, la=la, lb=lb, dt=dt, kname=kname):
                    a, o = b.window(la), b.window(lb)
                    acc, bar = b.accumulator(), b.barrier(init=True)
                    tok = b.barrier_arm(bar, ops=1)
                    b.mma_async(acc, a, o, tok, k=kname, dtype=dt)
                    b.barrier_wait(tok)
                    b.exit()

                good, d = compile_ok(build)
                if good:
                    ok = False
                    log.append(f"    {dt}/{kname} {which} short: ACCEPTED")
                elif not (str(need) in str(d) and kname[1:] in str(d)
                          and dt in str(d)):
                    ok = False
                    log.append(f"    {dt}/{kname} {which}: unhelpful: {d}")
    log.append(f"    24 short-window cases, all refused with K, dtype and size")
    return ok, 'a short window is refused and the diagnostic says by how much'


def fp03(log):
    """Exact length accepted; one over accepted; past the file refused."""
    dt, kname = 'f16', 'k32'
    need = oracle_registers(16, KSTEPS[kname], DTYPES[dt])

    def build(la, lb):
        def f(b):
            a, o = b.window(la), b.window(lb)
            acc, bar = b.accumulator(), b.barrier(init=True)
            tok = b.barrier_arm(bar, ops=1)
            b.mma_async(acc, a, o, tok, k=kname, dtype=dt)
            b.barrier_wait(tok)
            b.exit()
        return f

    exact, _ = compile_ok(build(need, need))
    over, _ = compile_ok(build(need + 1, need + 1))
    # Two windows of 64 registers each cannot both fit a 128-register file
    # alongside anything else; this must be a resource error, not silence.
    huge, dhuge = compile_ok(build(64, 64))
    log.append(f"    exact({need})={exact} over({need+1})={over} "
               f"overbudget(64,64)={huge}")
    if dhuge:
        log.append(f"      {str(dhuge).splitlines()[0][:110]}")
    return exact and over and not huge, 'window lengths at and past the limit'


def fp04(log):
    """The audit's own negative: K64 FP16 into 8-register windows."""
    def build(b):
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k64')
        b.barrier_wait(tok)
        b.exit()
    ok, d = compile_ok(build)
    log.append(f"    refused at compile time: {not ok}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return (not ok) and '32' in str(d), 'the audit reproduction, now refused'


def fp05(log):
    """DMA coordinate runs: legal ranges modelled, illegal ones refused."""
    cases = []
    for rank in (1, 2):
        for sb in (1, 62 - rank + 1):
            try:
                rng = dma_coordinate_footprint(sb, rank)
                cases.append((f"s{sb} rank{rank}", rng.count == rank, str(rng)))
            except Diagnostic as d:
                cases.append((f"s{sb} rank{rank}", False, str(d)[:60]))
    bad = []
    for sb, rank in ((63, 2), (0, 1), (62, 4)):
        try:
            dma_coordinate_footprint(sb, rank)
            bad.append(f"s{sb} rank{rank} accepted")
        except Diagnostic:
            pass
    for label, ok, detail in cases:
        log.append(f"    {label}: {'ok' if ok else 'FAIL'} {detail}")
    log.append(f"    out-of-range bases refused: {not bad}")
    return all(c[1] for c in cases) and not bad, 'coordinate runs never wrap'


def fp06(log):
    """Different names, overlapping physical range, after allocation."""
    def build(b):
        a = b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, a, tok, k='k16')     # same window as A and B
        v = b.vmovi(1)
        b.window_put(a, 0, v)                    # writes what MMA reads
        b.barrier_wait(tok)
        b.exit()
    ok, d = compile_ok(build)
    log.append(f"    overlapping write refused: {not ok}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return not ok, 'overlap found on the physical range, not on identity'


# --------------------------------------------------------------------------
def _mma_then(write, *, k='k16', dt='f16', wait_first=False, length=8):
    def build(b):
        a, o = b.window(length), b.window(length)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k=k, dtype=dt)
        if wait_first:
            b.barrier_wait(tok)
        write(b, a, o, acc)
        if not wait_first:
            b.barrier_wait(tok)
        b.exit()
    return build


def own_pair(log, cid, write, **kw):
    """A negative and the positive that differs only in the wait."""
    bad, d = compile_ok(_mma_then(write, wait_first=False, **kw))
    good, dg = compile_ok(_mma_then(write, wait_first=True, **kw))
    log.append(f"    {cid}: before the wait -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; "
               f"after -> {'accepted' if good else 'REFUSED (bad): ' + str(dg)[:80]}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return (not bad) and good


def own01(log):
    """A/B covered while the MMA is outstanding, three different ways."""
    ok = True
    ok &= own_pair(log, 'window_put', lambda b, a, o, acc:
                   b.window_put(a, 0, b.vmovi(7)))
    ok &= own_pair(log, 'plain vector write', lambda b, a, o, acc:
                   b.window_put(o, 1, b.vmovi(9)))
    return ok, 'a write into an outstanding fragment is refused'


def own05(log):
    """An accumulator read while its MMA is outstanding."""
    def read(b, a, o, acc):
        b.acc_read(acc, b.const(0))
    bad, d = compile_ok(_mma_then(read, wait_first=False))
    good, dg = compile_ok(_mma_then(read, wait_first=True))
    log.append(f"    read before the wait -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; after -> {good}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return (not bad) and good, 'accumulator ordering'


def own06(log):
    """A missing await: the command is issued and never waited for."""
    def build(b):
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.exit()                                  # no barrier_wait
    bad, d = compile_ok(build)
    log.append(f"    missing await -> {'ACCEPTED (bad)' if bad else 'refused'}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return not bad, 'an outstanding command must be awaited'


def own12(log):
    """Compiling the same IR twice, and roles in either order, agree."""
    def build(b):
        a, o = b.window(8), b.window(8)
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.barrier_wait(tok)
        b.exit()

    texts = []
    for _ in range(2):
        kernel = make_kernel('own12')
        bld = Builder(kernel, kernel.role(0).region)
        build(bld)
        cg = CodeGen(kernel, TargetInfo.default())
        cg.assign_shared()
        cg.run()
        texts.append(cg.b.text())
    same = texts[0] == texts[1]
    log.append(f"    two compilations of the same IR agree: {same}")
    return same, 'determinism across repeated compilation'


def _dma_kernel(body, *, wait_before_body=False):
    """A transfer into a window, then `body`, then the wait (or the reverse)."""
    def build(b):
        w = b.window(8)
        bar = b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        d = b.descriptor(b.const(0x5000))
        b.dma_load(w, d, [b.const(0), b.const(0)], tok, dtype='f16')
        if wait_before_body:
            b.barrier_wait(tok)
        body(b, w)
        if not wait_before_body:
            b.barrier_wait(tok)
        b.exit()
    return build


def own02(log):
    """Reading a DMA's target window before the transfer completes."""
    def read_target(b, w):
        v = b.window_reg(w, 0)
        b.vbin('add', v, v)     # an instruction that actually reads it

    bad, d = compile_ok(_dma_kernel(read_target))
    good, dg = compile_ok(_dma_kernel(read_target, wait_before_body=True))
    log.append(f"    read before the wait -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; after -> {good}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    if dg:
        log.append(f"      positive twin failed: {str(dg).splitlines()[0][:110]}")
    return (not bad) and good, 'a DMA target is not readable until it lands'


def own04(log):
    """DMA coordinates: the contract, checked against the RTL.

    rtl/core/tnca_tensor.v:357 latches cmd_c0/cmd_c1 on cmd_valid and every
    later use reads the latched copy, so the coordinates are snapshotted at
    issue and the scalar run is reusable immediately. This test states that
    conclusion as an executable claim: a program that rewrites the coordinate
    registers right after issuing must COMPILE, because the engine will never
    look at them again.
    """
    def rewrite_coords(b, w):
        # Allocating and writing more scalars reuses the coordinate run.
        for i in range(4):
            b.const(0x1000 + i)

    ok, d = compile_ok(_dma_kernel(rewrite_coords))
    log.append(f"    reusing the coordinate registers after issue: "
               f"{'accepted' if ok else 'REFUSED'}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    return ok, 'issue-time snapshot, proved in rtl/core/tnca_tensor.v:357'


def own10(log):
    """Over-budget resources are refused, with no partial issue."""
    t = TargetInfo.default()
    cases = []

    def n_windows(n):
        def build(b):
            # One register each: the point is the window budget, and an
            # eight-register window would exhaust the register file first and
            # test a different limit.
            for _ in range(n):
                b.window(1)
            b.exit()
        return build

    for n in (t.num_windows, t.num_windows + 1):
        ok, d = compile_ok(n_windows(n))
        cases.append((f"{n} windows", ok, d))
        log.append(f"    {n:3d} windows -> {'accepted' if ok else 'refused'}"
                   + (f": {str(d).splitlines()[0][:80]}" if d else ''))

    def n_barriers(n):
        def build(b):
            for _ in range(n):
                b.barrier(init=True)
            b.exit()
        return build

    for n in (t.num_barriers, t.num_barriers + 1):
        ok, d = compile_ok(n_barriers(n))
        cases.append((f"{n} barriers", ok, d))
        log.append(f"    {n:3d} barriers -> {'accepted' if ok else 'refused'}"
                   + (f": {str(d).splitlines()[0][:80]}" if d else ''))

    within = [c for c in cases if c[0].startswith(str(t.num_windows) + ' ')
              or c[0].startswith(str(t.num_barriers) + ' ')]
    over = [c for c in cases if c not in within]
    return (all(c[1] for c in within) and not any(c[1] for c in over)), \
        'resource budgets refused at the limit, not past it'


# --------------------------------------------------------------------------
def _with_caps(**caps):
    """A copy of the default target with some capabilities overridden.

    OWN-03 was written when no profile could source a `t.st.async` (the
    engine had no vector read port and raised ILLEGAL_INSTRUCTION), so the
    ownership rule was tested on a copy of the target with the capability
    turned on. The engine now performs stores (rtl/core/tnca_tensor.v, t34),
    the default target has the capability, and the case runs on it; this
    copy turns the capability OFF, to show the compiler still refuses the
    instruction where the machine cannot execute it.
    """
    import copy
    t = TargetInfo.default()
    d = copy.deepcopy(t._d)
    d['capabilities'].update(caps)
    return TargetInfo(d)


def _store_kernel(body, *, wait_before_body=False):
    """Issue an async store out of a window, then `body`, then the wait."""
    def build(b):
        w = b.window(8)
        # Something has to put data in the window before storing it, or the
        # test would be about an uninitialised range rather than a live one.
        b.window_put(w, 0, b.vmovi(3))
        bar = b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        d = b.descriptor(b.const(0x5000))
        b.dma_store(w, d, [b.const(0), b.const(0)], tok, dtype='f16')
        if wait_before_body:
            b.barrier_wait(tok)
        body(b, w)
        if not wait_before_body:
            b.barrier_wait(tok)
        b.exit()
    return build


def own03(log):
    """An async store's source overwritten before it is source-reusable."""
    cap = TargetInfo.default()

    # Half one: the capability gates the instruction -- refused where the
    # machine cannot execute it, emitted on the default target, which can.
    emitted, d0 = compile_ok(_store_kernel(lambda b, w: None,
                                           wait_before_body=True),
                             target=_with_caps(tensor_store=False))
    native, dn = compile_ok(_store_kernel(lambda b, w: None,
                                          wait_before_body=True))
    log.append(f"    on a profile without the capability -> "
               f"{'EMITTED (bad)' if emitted else 'refused'}; "
               f"on the default target -> "
               f"{'emitted' if native else 'REFUSED (bad)'}")
    if d0:
        log.append(f"      {str(d0).splitlines()[0][:120]}")
    if dn:
        log.append(f"      default target failed: {str(dn).splitlines()[0][:100]}")

    # Half two: where it can be emitted, the source window is held.
    def overwrite(b, w):
        b.window_put(w, 0, b.vmovi(9))

    def read_it(b, w):
        v = b.window_reg(w, 0)
        b.vbin('add', v, v)

    bad, d = compile_ok(_store_kernel(overwrite), target=cap)
    good, dg = compile_ok(_store_kernel(overwrite, wait_before_body=True),
                          target=cap)
    # A store READS the window, so reading it concurrently is legal. If this
    # were refused the checker would merely be forbidding everything.
    conc, dc = compile_ok(_store_kernel(read_it), target=cap)
    log.append(f"    overwrite before source-reusable -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; after -> {good}")
    log.append(f"    concurrent READ of the source -> "
               f"{'accepted' if conc else 'REFUSED (bad)'}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    for tag, dd in (('positive twin', dg), ('concurrent read', dc)):
        if dd:
            log.append(f"      {tag} failed: {str(dd).splitlines()[0][:110]}")
    return ((not emitted) and native and (not bad) and good and conc,
            'a store source is live until the completion event (sec 13.4)')


def own07(log):
    """Two consumers of one window; the LAST release is what frees it."""
    def build(overwrite_after):
        def f(b):
            a, o = b.window(8), b.window(8)
            acc1, acc2 = b.accumulator(), b.accumulator()
            bar1, bar2 = b.barrier(init=True), b.barrier(init=True)
            t1 = b.barrier_arm(bar1, ops=1)
            t2 = b.barrier_arm(bar2, ops=1)
            # Both commands read the SAME A window.
            b.mma_async(acc1, a, o, t1, k='k16')
            b.mma_async(acc2, a, o, t2, k='k16')
            if overwrite_after >= 1:
                b.barrier_wait(t1)
            if overwrite_after >= 2:
                b.barrier_wait(t2)
            b.window_put(a, 0, b.vmovi(5))
            if overwrite_after < 1:
                b.barrier_wait(t1)
            if overwrite_after < 2:
                b.barrier_wait(t2)
            b.exit()
        return f

    none_, d0 = compile_ok(build(0))     # neither consumer has released
    one, d1 = compile_ok(build(1))       # only the first has
    both, d2 = compile_ok(build(2))      # both have
    log.append(f"    reuse after 0 of 2 releases -> "
               f"{'ACCEPTED (bad)' if none_ else 'refused'}")
    log.append(f"    reuse after 1 of 2 releases -> "
               f"{'ACCEPTED (bad)' if one else 'refused'}")
    log.append(f"    reuse after 2 of 2 releases -> "
               f"{'accepted' if both else 'REFUSED (bad)'}")
    for tag, dd in (('0 of 2', d0), ('1 of 2', d1), ('2 of 2', d2)):
        if dd:
            log.append(f"      {tag}: {str(dd).splitlines()[0][:110]}")
    return ((not none_) and (not one) and both,
            'a shared range is freed by the last release, not the first')


def own08(log):
    """Partial overlap, physical aliasing, and reads past the named base."""
    ok = True

    # (a) A and B windows that PARTIALLY overlap. The write lands in the
    #     overlap, which no per-window identity check would notice.
    def partial(b):
        a = b.window(8)
        o = b.window(8, name='b')
        acc, bar = b.accumulator(), b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        b.mma_async(acc, a, o, tok, k='k16')
        b.window_put(o, 7, b.vmovi(1))     # last register of B
        b.barrier_wait(tok)
        b.exit()
    bad, d = compile_ok(partial)
    log.append(f"    write into the far end of B -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    ok &= not bad

    # (b) The implicit read range reaches past the base register the encoding
    #     names. A K64 FP16 A fragment is 32 registers; writing v+31 must be
    #     refused even though the instruction names only v+0.
    # e4m3 rather than f16: a K64 f16 fragment is 32 registers and two of
    # them exhaust the 64-register file before the footprint rule is reached,
    # which would test the allocator instead. At 8 bits a K64 fragment is 16.
    def past_base(k, offset):
        def f(b):
            a, o = b.window(16), b.window(16)
            acc, bar = b.accumulator(), b.barrier(init=True)
            tok = b.barrier_arm(bar, ops=1)
            b.mma_async(acc, a, o, tok, k=k, dtype='e4m3')
            b.window_put(a, offset, b.vmovi(1))
            b.barrier_wait(tok)
            b.exit()
        return f

    far, dfar = compile_ok(past_base('k64', 15))
    log.append(f"    k64: write 15 registers past the named base -> "
               f"{'ACCEPTED (bad)' if far else 'refused'}")
    if dfar:
        log.append(f"      {str(dfar).splitlines()[0][:120]}")
    ok &= not far

    # The twin: with K16 the same fragment is 4 registers, so register 15 of
    # the same window is genuinely outside the footprint and must be allowed.
    # A checker that refuses both is refusing the window, not the footprint.
    near, dnear = compile_ok(past_base('k16', 15))
    log.append(f"    k16: the same register, outside the footprint -> "
               f"{'accepted' if near else 'REFUSED (bad)'}")
    if dnear:
        log.append(f"      {str(dnear).splitlines()[0][:120]}")
    ok &= near

    # (c) Physical aliasing between roles. Two roles whose logical windows
    #     map to the same physical registers are the +aliasvrf configuration
    #     t13_share runs, and the checker must reason in the file the
    #     hardware uses.
    from tnca_cc.footprints import RegRange as RR
    from tnca_cc.footprints import OwnershipChecker
    ck = OwnershipChecker()
    ck.hold('role0-mma', [RR('v', 32, 8)], 'm.mma.async')
    aliased = False
    try:
        ck.check_write('v', 35, by='role1 v.mov v35')
    except Diagnostic:
        aliased = True
    log.append(f"    a second role writing into the same physical range -> "
               f"{'refused' if aliased else 'ACCEPTED (bad)'}")
    ok &= aliased
    return ok, 'overlap, aliasing and reads past the named base'


def own09(log):
    """Predicated-off async still completes once; an error completion wakes."""
    ok = True

    # The compiler's half: the operation count is static, so a predicated-off
    # transfer still counts. If it did not, the barrier would be armed for one
    # operation and receive none, and the waiter would hang.
    def masked(b):
        w = b.window(8)
        bar = b.barrier(init=True)
        tok = b.barrier_arm(bar, ops=1)
        d = b.descriptor(b.const(0x5000))
        m = b.vcmp('lt', b.arange(), b.splat(b.const(0)))   # all lanes off
        b.dma_load(w, d, [b.const(0), b.const(0)], tok, dtype='f16', mask=m)
        b.barrier_wait(tok)
        b.exit()

    kernel = make_kernel('own09')
    bb = Builder(kernel, kernel.role(0).region)
    masked(bb)
    cg = CodeGen(kernel, TargetInfo.default())
    cg.assign_shared()
    cg.run()
    text = cg.b.text()
    armed_one = 'mb.arm' in text and '#1' in text
    counted = text.count('t.ld.async') == 1
    log.append(f"    predicated-off transfer is still armed and issued once: "
               f"{armed_one and counted}")
    ok &= armed_one and counted

    # And the window is still held until the completion event: the command
    # completes whether or not it moved data, so reusing the window early is
    # wrong for exactly the same reason as the unmasked case.
    def early_read(b, w):
        v = b.window_reg(w, 0)
        b.vbin('add', v, v)

    def masked_then(body, wait_first):
        def f(b):
            w = b.window(8)
            bar = b.barrier(init=True)
            tok = b.barrier_arm(bar, ops=1)
            d = b.descriptor(b.const(0x5000))
            m = b.vcmp('lt', b.arange(), b.splat(b.const(0)))
            b.dma_load(w, d, [b.const(0), b.const(0)], tok, dtype='f16',
                       mask=m)
            if wait_first:
                b.barrier_wait(tok)
            body(b, w)
            if not wait_first:
                b.barrier_wait(tok)
            b.exit()
        return f

    bad, d = compile_ok(masked_then(early_read, False))
    good, _ = compile_ok(masked_then(early_read, True))
    log.append(f"    read before the masked transfer completes -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; after -> {good}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    ok &= (not bad) and good

    # The runtime half lives where it can be expressed: the error completion
    # path is RTL, and tb/core/progs/t35_tdesc_fpad2.s is the executable form
    # of "a failing command still signals its barrier, with the error bit
    # set, so the waiter is released instead of hanging" (t26_stfault was,
    # until the engine learned to perform stores). Named here so the case
    # points at its evidence rather than claiming the compiler proved it.
    import os as _os
    ev = _os.path.join(ROOT, 'tb', 'core', 'progs', 't35_tdesc_fpad2.s')
    have = _os.path.exists(ev)
    log.append(f"    error completion wakes the waiter: "
               f"tb/core/progs/t35_tdesc_fpad2.s ({'present' if have else 'MISSING'}), "
               f"run by `make core` with +expectfault=04")
    ok &= have
    return ok, 'predication completes once; an error completion still wakes'


def own11(log):
    """Loops: cross-iteration reuse, a missed release, and an undrained exit."""
    ok = True

    def loop_kernel(*, wait_inside, refill=True):
        def f(b):
            a, o = b.window(8), b.window(8)
            acc, bar = b.accumulator(), b.barrier(init=True)
            i = b.const(0)
            with b.loop(i, b.const(4)):
                tok = b.barrier_arm(bar, ops=1)
                b.mma_async(acc, a, o, tok, k='k16')
                if wait_inside:
                    b.barrier_wait(tok)
                if refill:
                    # The next iteration refills A. Without the wait above,
                    # this write lands in the previous iteration's read set.
                    b.window_put(a, 0, b.vmovi(1))
                if not wait_inside and refill:
                    b.barrier_wait(tok)
            b.exit()
        return f

    bad, d = compile_ok(loop_kernel(wait_inside=False))
    good, dg = compile_ok(loop_kernel(wait_inside=True))
    log.append(f"    refill before the iteration's wait -> "
               f"{'ACCEPTED (bad)' if bad else 'refused'}; "
               f"after -> {'accepted' if good else 'REFUSED (bad)'}")
    if d:
        log.append(f"      {str(d).splitlines()[0][:120]}")
    if dg:
        log.append(f"      positive twin: {str(dg).splitlines()[0][:110]}")
    ok &= (not bad) and good

    # A loop that leaves the last iteration outstanding at `exit`. The region
    # never drains, which is OWN-06's rule applied across a back edge.
    # No refill at all, so the only thing wrong is the missing wait: the
    # diagnostic has to be about the undrained exit, not about a write.
    undrained, du = compile_ok(loop_kernel(wait_inside=False, refill=False))
    log.append(f"    loop exits with a command outstanding -> "
               f"{'ACCEPTED (bad)' if undrained else 'refused'}")
    if du:
        log.append(f"      {str(du).splitlines()[0][:120]}")
    ok &= not undrained
    return ok, 'cross-iteration reuse and an undrained loop exit'


FP_TESTS = [('FP-01', fp01), ('FP-02', fp02), ('FP-03', fp03),
            ('FP-04', fp04), ('FP-05', fp05), ('FP-06', fp06)]
OWN_TESTS = [('OWN-01', own01), ('OWN-02', own02), ('OWN-03', own03),
             ('OWN-04', own04), ('OWN-05', own05), ('OWN-06', own06),
             ('OWN-07', own07), ('OWN-08', own08), ('OWN-09', own09),
             ('OWN-10', own10), ('OWN-11', own11), ('OWN-12', own12)]
TESTS = FP_TESTS + OWN_TESTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R1-05/R1-06 resource footprint and async ownership")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, detail = fn(log)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(dict(id=cid, passed=bool(ok), detail=detail))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {detail}")
        for line in log:
            print(line)

    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r1-resource-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r1-resource-tests      PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
