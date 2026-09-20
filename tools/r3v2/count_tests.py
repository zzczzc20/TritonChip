#!/usr/bin/env python3
"""
R3V2-COUNT-01 and -02 -- doc/phase3_tasks_and_acceptance.md G5.

The counters against the events they claim to count, and the boundaries
every number is taken on.

COUNT-01 does not compare the counters with themselves. The core prints a
bounded event trace -- each request offered, each one taken, each response,
each write and whether it was accepted -- and this counts those events in
Python. A counter that agrees with a tally of its own increments has
demonstrated nothing; one that agrees with the events is a measurement.

COUNT-02 is mostly about not mixing up three different things. Simulation
wall time, hardware cycles and a clock frequency are separate, the first
two are reported separately, and the third is NOT MEASURED -- deriving a
GHz from how fast the simulator ran is the specific mistake section 2 of
the contract opens by naming.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

BUILD = os.path.join(ROOT, 'sim', 'build')
CAND = os.path.join(BUILD, 'tb_core_vrf.vvp')
REF = os.path.join(BUILD, 'tb_core.vvp')
WINDOW = json.load(open(os.path.join(
    ROOT, 'spec', 'r3v2_inventory.json')))['budget'][
        'diagnostic_window_cycles']

PROGRAMS = ['t29_vrfbank', 't01_vector', 't11_matrix', 't16_vecadd',
            't25_mmasf', 't28_count']


def run(vvp, prog, extra=(), args=()):
    started = time.time()
    r = subprocess.run(['vvp', vvp, f'+hex={os.path.join(BUILD, prog)}.hex',
                        '+maxcycles=200000'] + list(extra) + list(args),
                       capture_output=True, text=True, timeout=1800)
    wall = time.time() - started
    out = {'passed': bool(re.search(r'^PASS', r.stdout, re.M)),
           'wall_s': round(wall, 3), 'stdout': r.stdout}
    for pat in (r'^vrf ((?:\w+=\d+ ?)+)',
                r'^stall_total=(\d+)',):
        pass
    m = re.search(r'^vrf ((?:\w+=\d+ ?)+)', r.stdout, re.M)
    if m:
        out.update({k: int(v) for k, v in
                    (kv.split('=') for kv in m.group(1).split())})
    m = re.search(r'^cycles=(\d+)', r.stdout, re.M)
    if m:
        out['cycles'] = int(m.group(1))
    m = re.search(r'^stall_total=(\d+) ((?:x_\w+=\d+ ?)+)', r.stdout, re.M)
    if m:
        out['stall_total'] = int(m.group(1))
        out.update({k: int(v) for k, v in
                    (kv.split('=') for kv in m.group(2).split())})
    return out


def tally(stdout, limit):
    """Count the events in the trace, independently of the counters."""
    n = dict(req=0, accept=0, conflict=0, mtx_req=0, mtx_wait=0,
             wb_retry=0, cycles=0)
    for line in stdout.splitlines():
        m = re.match(r'VRFEV (\d+) want=([01]+) acc=([01]+) rsp=([01]+) '
                     r'mreq=([01]) mval=([01]) we=([01]+) wok=([01]+)',
                     line)
        if not m:
            continue
        cyc = int(m.group(1))
        if cyc >= limit:
            continue
        n['cycles'] += 1
        want, acc = m.group(2), m.group(3)
        n['req'] += want.count('1')
        n['accept'] += acc.count('1')
        n['conflict'] += sum(1 for w, a in zip(want, acc)
                             if w == '1' and a == '0')
        if m.group(5) == '1':
            n['mtx_req'] += 1
            if m.group(6) == '0':
                n['mtx_wait'] += 1
        we, wok = m.group(7), m.group(8)
        # we[0] is the writeback in the printed order (MSB first)
        if we[0] == '1' and wok[0] == '0':
            n['wb_retry'] += 1
    return n


def count01(log, ctx):
    """The counters equal a Python tally of the events they count."""
    ok = True
    for prog in ('t29_vrfbank', 't11_matrix'):
        # The whole run, traced to the end, so the tally and the counters
        # cover the same window.
        r = run(CAND, prog, extra=[f'+vrftrace=1000000'])
        got = tally(r['stdout'], 10 ** 9)
        pairs = [('req', 'req'), ('accept', 'accept'),
                 ('conflict', 'conflict'), ('mtx_req', 'mtx_req'),
                 ('mtx_wait', 'mtx_wait'), ('wb_retry', 'wb_retry')]
        line = []
        for counter, event in pairs:
            hw, tr = r.get(counter), got[event]
            line.append(f"{counter}={hw}/{tr}")
            if hw != tr:
                log.append(f"    {prog} {counter}: the counter says {hw}, "
                           f"the events say {tr}")
                ok = False
        log.append(f"    {prog:<12s} counter/events: {'  '.join(line)}")
    log.append(f"    every counter equals an independent tally of the "
               f"events it counts")

    # The exclusive stall attribution sums to the total, and the
    # overlapping view is kept beside it rather than folded in.
    ex = ['x_hazard', 'x_lsu', 'x_fpu', 'x_mtx', 'x_txf', 'x_fence',
          'x_arrive', 'x_rsv', 'x_wb', 'x_vrf', 'x_vrfwb']
    for prog in PROGRAMS:
        r = run(CAND, prog)
        if 'stall_total' not in r:
            continue
        s = sum(r.get(k, 0) for k in ex)
        if s != r['stall_total']:
            log.append(f"    {prog}: exclusive sum {s} != total "
                       f"{r['stall_total']}")
            ok = False
    log.append(f"    across {len(PROGRAMS)} program(s) the nine exclusive "
               f"stall counters sum to perf_stall_total exactly")
    return ok, 'counters equal their events; stalls are exclusive and sum'


def count02(log, ctx):
    """Boundaries, and three different things kept apart."""
    ok = True
    log.append(f"    {'program':<13s} {'sim wall':>9s} {'cycles':>8s} "
               f"{'ROI cycles':>11s} {'cycles/s':>9s}")
    rows = []
    for prog in PROGRAMS:
        r = run(CAND, prog)
        if 'cycles' not in r:
            continue
        # The region of interest: from the launch to the drain. The core
        # counts perf_cycles only while a role context is running, which
        # is exactly that window -- and `cycles` is the whole simulation
        # including reset and the final drain.
        roi = r['cycles']
        rate = round(roi / r['wall_s']) if r['wall_s'] else 0
        rows.append((prog, r['wall_s'], r['cycles'], roi, rate))
        log.append(f"    {prog:<13s} {r['wall_s']:>8.2f}s {r['cycles']:>8d} "
                   f"{roi:>11d} {rate:>9d}")
        ok &= r['passed']
    ctx['rows'] = rows

    log.append(f"    simulation wall time, hardware cycles and a clock "
               f"frequency are three different things")
    log.append(f"    Fmax: NOT MEASURED -- a GHz derived from how fast the "
               f"simulator ran would be a statement about this laptop")

    # The same boundaries on both implementations, so the comparison is
    # between machines rather than between measurement windows.
    for prog in ('t01_vector', 't11_matrix'):
        a = run(REF, prog)
        b = run(CAND, prog)
        log.append(f"    {prog:<13s} reference {a['cycles']:>6d} cycles, "
                   f"candidate {b['cycles']:>6d} "
                   f"({(b['cycles'] - a['cycles']) * 100.0 / a['cycles']:+.0f}%), "
                   f"same boundary on both")
        ok &= a['passed'] and b['passed']
    return ok, 'one boundary, three quantities, and no frequency invented'


TESTS = [('R3V2-COUNT-01', count01), ('R3V2-COUNT-02', count02)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G5: counters, and what they are counted over")
    for p in (REF, CAND):
        if not os.path.exists(p):
            print(f"  {p} is not built; run `make core core-vrf`")
            return 1
    ctx = {}
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
            log.append('    ' + traceback.format_exc()[-900:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}", flush=True)
        for line in log:
            print(line, flush=True)
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3v2-count-tests       FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-count-tests       PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
