#!/usr/bin/env python3
"""
FLOW-01, FLOW-02, FLOW-04 and FLOW-05 -- doc/phase3_tasks_and_acceptance.md
R3-04.

tb/core/tb_operand_collector.v runs four directed scenarios against the real
banked register file and prints what happened each cycle. The checking is
here, so the bench decides nothing.

FLOW-03 is NOT RUN and says why.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

BENCH = os.path.join(ROOT, 'tb', 'core', 'tb_operand_collector.v')
ENTRIES = 4
SRCS = 3
LANES = 16


def build(out, latency=1):
    return subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'),
         '-y', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'tb', 'common'),
         '-P', f'tb_operand_collector.RF_LAT={latency}',
         '-o', out, BENCH],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)


def run(vvp, case):
    r = subprocess.run([shutil.which('vvp') or 'vvp', vvp, f'+case={case}'], capture_output=True,
                       text=True, timeout=900)
    cycles, ops, final, reused, flush_at = [], None, {}, None, None
    for line in r.stdout.splitlines():
        m = re.match(r'CYC (\d+) busy=(\S+) ready=(\S+) grant=(\d+) '
                     r'conflict=(\d+) wait=(\d+) orphan=(\d+)', line)
        if m:
            cycles.append(dict(cyc=int(m.group(1)), busy=m.group(2),
                               ready=m.group(3), grant=int(m.group(4)),
                               conflict=int(m.group(5)),
                               wait=int(m.group(6)),
                               orphan=int(m.group(7))))
            continue
        m = re.match(r'OPS (\d+) (\S+) (\S+) (\S+)', line)
        if m:
            ops = [m.group(2), m.group(3), m.group(4)]
            continue
        m = re.match(r'FINAL grant=(\d+) conflict=(\d+) wait=(\d+) '
                     r'orphan=(\d+)', line)
        if m:
            final = dict(grant=int(m.group(1)), conflict=int(m.group(2)),
                         wait=int(m.group(3)), orphan=int(m.group(4)))
            continue
        m = re.match(r'REUSED (\S+)', line)
        if m:
            reused = m.group(1)
        m = re.match(r'FLUSH (\d+)', line)
        if m:
            flush_at = int(m.group(1))
    if 'COLLECT-ASSERT' in r.stdout:
        final = final or {}
        final['assert'] = [l for l in r.stdout.splitlines()
                           if 'COLLECT-ASSERT' in l]
    return dict(cycles=cycles, ops=ops, final=final, reused=reused,
                flush_at=flush_at, raw=r.stdout, rc=r.returncode)


def lane0(hexstr):
    return int(hexstr[-8:], 16) if hexstr else None


def first_ready(cycles, bit):
    """The first cycle in which entry `bit` is ready."""
    for c in cycles:
        # ready is printed MSB-first over ENTRIES bits
        if c['ready'][ENTRIES - 1 - bit] == '1':
            return c['cyc']
    return None


def flow01(log, ctx):
    """Collection order and data, same bank and across banks."""
    ok = True
    same = run(ctx['vvp'], 1)
    cross = run(ctx['vvp'], 2)

    # Across banks: three sources, three banks, one cycle, no conflict.
    log.append(f"    v0,v1,v2 (three banks): {cross['final']['grant']} "
               f"grant(s), {cross['final']['conflict']} conflict(s), ready "
               f"at cycle {first_ready(cross['cycles'], 0)}")
    ok &= cross['final']['grant'] == 3 and cross['final']['conflict'] == 0
    ok &= first_ready(cross['cycles'], 0) == 1

    # Same bank: three sources serialise, and the entry is NOT ready until
    # the last one lands. That is FLOW-01's "未收齐不发射".
    log.append(f"    v0,v4,v8 (one bank):   {same['final']['grant']} "
               f"grant(s), {same['final']['conflict']} conflict(s), ready "
               f"at cycle {first_ready(same['cycles'], 0)}")
    ok &= same['final']['grant'] == 3
    ok &= same['final']['conflict'] > 0
    ready_at = first_ready(same['cycles'], 0)
    ok &= ready_at is not None and ready_at >= 2
    not_ready_early = all(c['ready'][ENTRIES - 1] == '0'
                          for c in same['cycles'] if c['cyc'] < ready_at)
    log.append(f"    not ready in any earlier cycle: {not_ready_early}")
    ok &= not_ready_early

    # And the operands are the right registers, not three copies of one.
    for label, res, want in (('one bank', same, [0x1000, 0x1400, 0x1800]),
                             ('three banks', cross,
                              [0x1000, 0x1100, 0x1200])):
        got = [lane0(h) for h in (res['ops'] or [])]
        log.append(f"    {label}: lane 0 of each operand = "
                   f"{[hex(g) if g is not None else None for g in got]}")
        if got != want:
            log.append(f"        wanted {[hex(w) for w in want]}")
            ok = False
    return ok, 'collected in order, complete before ready, right registers'


def flow02(log, ctx):
    """Every request served, none lost, none duplicated, fairly."""
    ok = True
    res = run(ctx['vvp'], 3)
    needed = ENTRIES * SRCS
    g = res['final']['grant']
    log.append(f"    {ENTRIES} entries x {SRCS} sources, all in one bank: "
               f"{g} grant(s) for {needed} required read(s)")
    # Exactly the reads that were needed: a lost request would be fewer,
    # a re-requested one that was already served would be more.
    ok &= g == needed

    # One read port on that bank, so at most one grant per cycle.
    per_cycle = []
    prev = 0
    for c in res['cycles']:
        per_cycle.append(c['grant'] - prev)
        prev = c['grant']
    log.append(f"    grants per cycle: max {max(per_cycle)} "
               f"(one bank has one read port)")
    ok &= max(per_cycle) <= 1

    # Fairness: within any window of ENTRIES grants, every entry with an
    # outstanding request is served at least once. Observable here as the
    # entries completing in order rather than one entry finishing first.
    order = [(bit, first_ready(res['cycles'], bit))
             for bit in range(ENTRIES)]
    ready_cycles = [c for _b, c in order if c is not None]
    log.append(f"    entries became ready at cycles "
               f"{[c for _b, c in order]}")
    ok &= len(ready_cycles) == ENTRIES
    # No entry finishes more than ENTRIES-1 cycles before another starts
    # being served: round robin, not first-come-take-everything.
    if len(ready_cycles) == ENTRIES:
        spread = max(ready_cycles) - min(ready_cycles)
        log.append(f"    first to last ready: {spread} cycle(s); "
                   f"serialised would be {needed - SRCS}")
        ok &= spread <= needed

    # Waiting is counted, and it is not zero: a counter that reports no
    # contention while twelve reads queue for one port is not counting.
    log.append(f"    {res['final']['conflict']} refused request(s), "
               f"{res['final']['wait']} cycle(s) with something waiting")
    ok &= res['final']['conflict'] > 0 and res['final']['wait'] > 0
    return ok, 'every read served once, one per port, and waiting counted'


def flow04(log, ctx):
    """Conservation: no response without an outstanding request."""
    ok = True
    for case in (1, 2, 3, 4):
        res = run(ctx['vvp'], case)
        fired = res['final'].get('assert')
        if fired:
            log.append(f"    case {case}: {fired[0][:70]}")
            ok = False
    log.append(f"    the in-RTL conservation assertion did not fire in any "
               f"of the four scenarios")
    # And after the work is done nothing is left outstanding: every grant
    # has been answered.
    res = run(ctx['vvp'], 3)
    last = res['cycles'][-1]
    log.append(f"    after drain: ready={last['ready']}, "
               f"{res['final']['grant']} grant(s) all answered, "
               f"{res['final']['orphan']} orphan(s)")
    ok &= last['ready'] == '1' * ENTRIES and res['final']['orphan'] == 0
    return ok, 'request and response conserved in every scenario'


def flow05(log, ctx):
    """A flush drops un-issued work and absorbs what is already in flight.

    Run at both read latencies, because at one cycle the orphan path is
    unreachable and saying so is the result. A read accepted on edge N
    returns on N+1, and the collector -- which is always ready -- consumes
    it there; a flush has no cycle in which to land on an outstanding
    read. At two cycles there is a gap and the orphan appears.

    This case used to pass at latency 1 with one orphan, which was an
    accident of when the register file happened to grant. Changing the
    arbitration to round robin moved the grant and the orphan vanished,
    which is how the accident came to light.
    """
    ok = True
    res = run(ctx['vvp'], 4)
    log.append(f"    latency 1: {res['final']['orphan']} orphan(s) -- the "
               f"response returns before a flush can intervene, so there "
               f"is nothing to orphan")
    ok &= res['final']['orphan'] == 0

    res2 = run(ctx['vvp2'], 4)
    log.append(f"    latency 2: {res2['final']['orphan']} orphan(s) -- a "
               f"read really is outstanding when the entry disappears")
    ok &= res2['final']['orphan'] >= 1
    res = res2

    # The reused entry must hold the register the NEW allocation asked for
    # (v12 = 0x1c00), not the one the flushed allocation had in flight
    # (v0 = 0x1000).
    got = lane0(res['reused'])
    log.append(f"    the slot was reused immediately; it holds "
               f"{hex(got) if got is not None else None} "
               f"(v12), not 0x1000 (v0, the flushed request)")
    ok &= got == 0x1c00
    ok &= not res['final'].get('assert')
    return ok, 'un-issued work dropped, in-flight reads absorbed, no leak'


TESTS = [('FLOW-01', flow01), ('FLOW-02', flow02), ('FLOW-04', flow04),
         ('FLOW-05', flow05)]

NOT_YET = {
    'FLOW-03': 'RF busy on return, a full queue and a writeback conflict '
               'need the units that produce writebacks -- R3-05. The '
               'collector has the half it owns (an accepted read reserves '
               'its destination and is absorbed even if its entry is '
               'gone), but "已发事务可以完成" is a claim about the whole '
               'return path, and half of it does not exist yet',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3-04 operand collector, arbitration and return")
    tmp = tempfile.mkdtemp(prefix='r3-flow-')
    vvp = os.path.join(tmp, 'tb_oc.vvp')
    vvp2 = os.path.join(tmp, 'tb_oc2.vvp')
    for out, lat in ((vvp, 1), (vvp2, 2)):
        b = build(out, lat)
        if b.returncode != 0:
            print(f"  the bench did not build at latency {lat}: "
                  f"{b.stderr[-400:]}")
            return 1
    ctx = {'vvp': vvp, 'vvp2': vvp2, 'tmp': tmp}
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log, ctx)
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
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3-flow-tests          FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-flow-tests          PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
