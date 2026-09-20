#!/usr/bin/env python3
"""
MODEL-01 .. MODEL-05 -- doc/phase3_tasks_and_acceptance.md R3-02.

MODEL-01 asks for six small graphs whose cycle count and resource use can be
worked out by hand, and for the model to agree EXACTLY. Each case below states
the arithmetic in its docstring, so a disagreement is settled by reading
rather than by adjusting the model until it matches -- which is the failure
mode a calibration test has.

The graphs are small on purpose. A model that is right about four operations
and wrong about four hundred has a bug; a model that is right about four
hundred and cannot be checked by hand has no test.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

from model import (Machine, ModelError, Op, conservation,   # noqa: E402
                   simulate, sweep)


def check(log, label, ops, machine, want_cycles, why, **want):
    """One graph, one hand-computed answer."""
    r = simulate(ops, machine)
    ok = r.cycles == want_cycles
    log.append(f"    {label:<30s} {r.cycles:>4d} cycles  (hand: "
               f"{want_cycles})  {'agree' if ok else 'DISAGREE'}")
    log.append(f"        {why}")
    for key, value in want.items():
        got = r.stalls.get(key, r.issued.get(key))
        if got != value:
            log.append(f"        {key}: {got}, hand said {value}")
            ok = False
    problems = conservation(r, ops)
    if problems:
        log.append(f"        conservation: {problems}")
        ok = False
    return ok, r


def model01(log):
    """Six graphs, six hand calculations."""
    ok = True

    # 1. A serial chain. Four ALU operations, each depending on the last,
    #    each with latency 1: the fourth completes at cycle 4.
    ops = [Op(f"s{i}", 'alu', deps=(f"s{i - 1}",) if i else ())
           for i in range(4)]
    good, _ = check(log, 'serial chain of 4', ops, Machine(), 4,
                    'each op issues the cycle after its producer completes: '
                    '1+1+1+1')
    ok &= good

    # 2. Fully independent, with enough units. Four ALU ops, four ALUs,
    #    issue width 4: all issue in cycle 0 and complete at cycle 1.
    ops = [Op(f"i{i}", 'alu') for i in range(4)]
    good, _ = check(log, '4 independent, 4 ALUs', ops,
                    Machine(issue_width=4, units={'alu': (1, 1, 4)}), 1,
                    'nothing depends on anything and nothing is scarce: one '
                    'cycle')
    ok &= good

    # 3. The same graph with ONE ALU. Occupancy 1 each, so they serialise:
    #    issued at 0,1,2,3 and the last completes at 4.
    good, _ = check(log, '4 independent, 1 ALU', ops,
                    Machine(issue_width=4, units={'alu': (1, 1, 1)}), 4,
                    'one ALU, occupancy 1: four operations cannot overlap, '
                    'whatever the issue width says')
    ok &= good

    # 4. Same-bank conflict. Two operations, each reading two registers that
    #    land on banks 0 and 1 of a two-bank file with one port each. Either
    #    op alone fits; the two together want two ports on each bank, so they
    #    serialise: issued at 0 and 1, the second completes at 2.
    ops = [Op('c0', 'alu', reads=(0, 1)), Op('c1', 'alu', reads=(2, 3))]
    good, r = check(log, '2 ops, 2 banks, 1 port each', ops,
                    Machine(issue_width=2, rf_banks=2,
                            rf_read_ports_per_bank=1,
                            units={'alu': (1, 1, 2)}), 2,
                    'each op takes both banks\' single port, so the second '
                    'waits a cycle even with two ALUs and width 2')
    ok &= good
    if r.stalls['rf_bank'] == 0:
        log.append('        no bank conflict was recorded (bad)')
        ok = False

    # And the configuration that CANNOT work is refused rather than spun on:
    # one op needing two ports on one bank.
    try:
        simulate([Op('impossible', 'alu', reads=(0, 2))],
                 Machine(rf_banks=1, rf_read_ports_per_bank=1))
        log.append('    an op needing 2 ports on a 1-port bank: ACCEPTED (bad)')
        ok = False
    except ModelError as e:
        log.append(f"    2 reads on a 1-port bank         -> {str(e)[:62]}")

    # 5. Different banks. The same two operations across FOUR banks: c0
    #    takes banks 0 and 1, c1 takes 2 and 3, nothing is shared, and both
    #    issue in cycle 0.
    ops = [Op('d0', 'alu', reads=(0, 1)), Op('d1', 'alu', reads=(2, 3))]
    good, _ = check(log, '2 reads, 4 banks, 1 port each', ops,
                    Machine(issue_width=2, rf_banks=4,
                            rf_read_ports_per_bank=1,
                            units={'alu': (1, 1, 2)}), 1,
                    'four banks, four distinct registers, two units: both '
                    'ops issue in cycle 0')
    ok &= good

    # 6. A finite queue. Four DMA operations, a queue of depth 2, latency 4.
    #    Two issue at cycle 0 and 1 (one DMA engine, occupancy 1); they
    #    leave the queue at 4 and 5; the third issues at 4 and the fourth at
    #    5, completing at 8 and 9.
    ops = [Op(f"q{i}", 'dma', queue='dma') for i in range(4)]
    good, r = check(log, '4 transfers, queue depth 2', ops,
                    Machine(issue_width=1, queues={'dma': 2},
                            units={'dma': (1, 4, 1)}), 9,
                    'depth 2 and latency 4: two in flight, the next pair '
                    'waits for them to drain')
    ok &= good
    if r.stalls['queue'] == 0:
        log.append('        the queue never stalled, so its depth did '
                   'nothing (bad)')
        ok = False
    else:
        log.append(f"        the queue stalled {r.stalls['queue']} time(s), "
                   f"which is what a finite queue is for")

    # 7. ready/free reuse. A transfer, then a consumer that depends on it,
    #    then a second transfer that depends on the consumer finishing --
    #    the window cannot be refilled until its reader is done.
    ops = [Op('fill', 'dma', queue='dma'),
           Op('use', 'mma', deps=('fill',)),
           Op('refill', 'dma', deps=('use',), queue='dma')]
    good, _ = check(log, 'fill -> use -> refill', ops,
                    Machine(issue_width=1, queues={'dma': 2},
                            units={'dma': (1, 4, 1), 'mma': (1, 6, 1)}),
                    14,
                    'fill completes at 4, the MMA issues at 4 and completes '
                    'at 10, the refill issues at 10 and completes at 14')
    ok &= good
    return ok, 'six graphs, six hand calculations, exact agreement'


def model02(log):
    """Fewer ports and longer latencies change the answer, the right way."""
    ok = True
    ops = [Op(f"x{i}", 'alu', reads=(i * 2, i * 2 + 1)) for i in range(4)]

    wide = Machine(issue_width=4, rf_banks=8, rf_read_ports_per_bank=1,
                   units={'alu': (1, 1, 4)})
    narrow = Machine(issue_width=4, rf_banks=2, rf_read_ports_per_bank=1,
                     units={'alu': (1, 1, 4)})
    a = simulate(ops, wide).cycles
    b = simulate(ops, narrow).cycles
    log.append(f"    8 banks -> {a} cycles, 2 banks -> {b} cycles")
    if b <= a:
        log.append('        halving the banks did not cost anything (bad)')
        ok = False

    slow = Machine(issue_width=4, rf_banks=8, rf_read_ports_per_bank=1,
                   units={'alu': (1, 3, 4)})
    c = simulate(ops, slow).cycles
    log.append(f"    latency 1 -> {a} cycles, latency 3 -> {c} cycles")
    if c <= a:
        log.append('        tripling the latency did not cost anything (bad)')
        ok = False

    # The three things a flattering model would allow.
    for label, machine in (
            ('a queue with no depth', Machine(queues={'dma': None})),
            ('a unit that takes no time', Machine(units={'alu': (0, 1, 1)})),
            ('a bank with no read port',
             Machine(rf_read_ports_per_bank=0))):
        try:
            simulate([Op('z', 'alu')], machine)
            log.append(f"    {label:<28s} -> ACCEPTED (bad)")
            ok = False
        except ModelError as e:
            log.append(f"    {label:<28s} -> refused: "
                       f"{str(e).split(':')[0][:44]}")

    # And a consumer cannot issue before its producer's result exists.
    ops = [Op('p', 'dma', queue='dma'), Op('c', 'alu', deps=('p',))]
    r = simulate(ops, Machine(units={'dma': (1, 9, 1), 'alu': (1, 1, 1)}))
    log.append(f"    a consumer of a 9-cycle transfer completes at "
               f"{r.cycles} (hand: 10)")
    ok &= r.cycles == 10
    return ok, 'scarcity costs, and nothing is free'


def model03(log):
    """Conservation, acyclicity, and an illegal graph refused."""
    ok = True
    ops = [Op(f"a{i}", 'alu', deps=(f"a{i - 1}",) if i else ())
           for i in range(6)]
    r = simulate(ops)
    problems = conservation(r, ops)
    log.append(f"    6 operations issued {sum(r.issued.values())} time(s): "
               f"{problems or 'conserved'}")
    ok &= not problems

    for label, bad in (
            ('a dependency cycle',
             [Op('a', 'alu', deps=('b',)), Op('b', 'alu', deps=('a',))]),
            ('an unknown dependency', [Op('a', 'alu', deps=('ghost',))]),
            ('a unit the machine lacks', [Op('a', 'tensor-core')]),
            ('two operations with one name',
             [Op('a', 'alu'), Op('a', 'alu')]),
            ('a queue the machine lacks', [Op('a', 'dma', queue='nope')])):
        try:
            r = simulate(bad)
            log.append(f"    {label:<28s} -> {r.cycles} cycles (bad: a "
                       f"number where an error belongs)")
            ok = False
        except ModelError as e:
            log.append(f"    {label:<28s} -> {str(e)[:50]}")
    return ok, 'nothing lost, no cycles, and no zero-cycle answer for an '\
               'impossible machine'


def model04(log):
    """Deterministic, and a bounded sweep inside its time budget."""
    ok = True
    ops = [Op(f"a{i}", 'alu', reads=(i, i + 1),
              deps=(f"a{i - 1}",) if i % 3 == 0 and i else ())
           for i in range(24)]
    runs = [simulate(ops, Machine(issue_width=2, rf_banks=4,
                                  units={'alu': (1, 2, 2)})).cycles
            for _ in range(5)]
    log.append(f"    the same input five times: {runs} -> "
               f"{'identical' if len(set(runs)) == 1 else 'DIFFERENT (bad)'}")
    ok &= len(set(runs)) == 1

    candidates = []
    for banks in (2, 4, 8):
        for ports in (1, 2):
            for width in (1, 2, 4):
                for alus in (1, 2, 4):
                    candidates.append((
                        f"b{banks}p{ports}w{width}u{alus}",
                        Machine(issue_width=width, rf_banks=banks,
                                rf_read_ports_per_bank=ports,
                                units={'alu': (1, 2, alus)})))
    started = time.time()
    out = sweep(ops, candidates, limit=128)
    elapsed = time.time() - started
    best = min((o for o in out if o['ok']), key=lambda o: o['cycles'])
    log.append(f"    swept {len(out)} candidates in {elapsed:.2f}s "
               f"(budget: 128 candidates, 60 s)")
    log.append(f"    best: {best['candidate']} at {best['cycles']} cycles, "
               f"bottleneck {best['bottleneck']}")
    ok &= len(out) <= 128 and elapsed < 60

    # Three copies, because two of 54 is 108 and still inside the budget --
    # the first draft of this check asserted a refusal that was not owed.
    oversized = candidates * 3
    try:
        sweep(ops, oversized, limit=128)
        log.append(f"    a {len(oversized)}-candidate sweep: ACCEPTED (bad)")
        ok = False
    except ModelError as e:
        log.append(f"    a sweep past the budget -> {str(e)[:56]}")
    return ok, 'deterministic, bounded, and fast enough to be useful'


def model05(log):
    """What a result says about itself."""
    ops = [Op('f', 'dma', queue='dma'), Op('m', 'mma', deps=('f',))]
    r = simulate(ops, Machine(), trace=True)
    d = r.as_dict()
    log.append(f"    cycles={d['cycles']}  bottleneck={d['bottleneck']}  "
               f"issued={d['issued']}")
    log.append(f"    machine recorded: banks={d['machine']['rf_banks']}, "
               f"units={sorted(d['machine']['units'])}")
    log.append(f"    kind: {d['kind']!r}")
    ok = d['kind'] == 'model projection'
    if not ok:
        log.append('        a model result labelled as a measurement (bad)')
    # The rejection reason travels with the rejected candidate rather than
    # the candidate disappearing from the table.
    out = sweep(ops, [('impossible', Machine(units={'alu': (1, 1, 1)}))])
    log.append(f"    a rejected candidate: {out[0]['candidate']} -> "
               f"{out[0].get('reason', '')[:56]}")
    ok &= not out[0]['ok'] and bool(out[0].get('reason'))
    return ok, 'cycles, bottleneck, machine, and never "RTL measured"'


TESTS = [('MODEL-01', model01), ('MODEL-02', model02), ('MODEL-03', model03),
         ('MODEL-04', model04), ('MODEL-05', model05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3-02 finite-resource performance model")
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
            log.append('    ' + traceback.format_exc()[-700:].replace(
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
        print(f"  r3-model-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-model-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
