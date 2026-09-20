#!/usr/bin/env python3
"""
COUNT-01, COUNT-02 and COUNT-05 -- doc/phase3_tasks_and_acceptance.md R3-06.

spec/r3_counters.json says where each count comes from, when it moves, what
resets it and what it does when it runs out of bits. This checks the claims
that can be checked by running something.

COUNT-02 is the one with a result. The core had three stall counters --
hazard, lsu, wb -- and they cannot sum to the blocked cycles no matter how
they are read: `hazard` and `stall_lsu` are separate terms of one OR and
are often true together, while stall_fpu, stall_mtx, stall_txf,
stall_fence, stall_arrive and stall_rsv were not counted at all. So the
old three overcount on some programs and undercount on others, which is
why adding them up and looking never showed anything wrong. Both views are
now reported: an exclusive attribution that sums to the total by
construction, and the overlapping one kept beside it.

COUNT-03 and COUNT-04 are NOT RUN and say why.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402

BUILD = os.path.join(ROOT, 'sim', 'build')
VVP = os.path.join(BUILD, 'tb_core.vvp')
SPEC = os.path.join(ROOT, 'spec', 'r3_counters.json')

# Programs to sweep for COUNT-02. A mix on purpose: the exclusive sum has to
# hold for a program that never stalls as well as one that mostly does.
PROGRAMS = ['t04_roles', 't11_matrix', 't12_tensor', 't16_vecadd',
            't20_sfu', 't21_cvt', 't22_atomic', 't25_mmasf', 't28_count']

# t28_count is straight line, twelve instructions that each encode to one
# word, one role context. The number is countable by eye; see the program.
HAND_COUNTED_ISSUE = 12

EXCLUSIVE = ['x_hazard', 'x_lsu', 'x_fpu', 'x_mtx', 'x_txf', 'x_fence',
             'x_arrive', 'x_rsv', 'x_wb']


def run_prog(prog, extra=()):
    hexf = os.path.join(BUILD, f"{prog}.hex")
    if not os.path.exists(hexf):
        return None
    r = subprocess.run([os.environ.get('VVP', 'vvp'), VVP, f'+hex={hexf}',
                        '+maxcycles=100000'] + list(extra),
                       capture_output=True, text=True, timeout=1800)
    out = {}
    for line in r.stdout.splitlines():
        for m in re.finditer(r'(\w+)=(\d+)', line):
            out.setdefault(m.group(1), int(m.group(2)))
    out['_passed'] = 'PASS' in r.stdout
    return out


def count01(log, ctx):
    """Counts matched against a trace worked out by hand."""
    ok = True
    r = run_prog('t28_count')
    if r is None:
        return False, 't28_count is not built'
    log.append(f"    t28_count: 12 instructions on the page, "
               f"{r.get('issued')} issued, {r.get('cycles')} cycle(s)")
    ok &= r.get('issued') == HAND_COUNTED_ISSUE
    ok &= r['_passed']
    # Straight line with no memory and no long-latency unit: nothing should
    # be blocked at all, and a counter that reports stalls here is counting
    # something that did not happen.
    log.append(f"    it stalls nowhere: stall_total={r.get('stall_total')}, "
               f"squash={r.get('squash')}")
    ok &= r.get('stall_total') == 0

    # The operand collector, counted the same way: four entries of three
    # sources all in one bank is twelve reads and can be no other number.
    vvp = ctx.get('oc_vvp')
    if vvp:
        p = subprocess.run([vvp, '+case=3'], capture_output=True, text=True,
                           timeout=900)
        m = re.search(r'FINAL grant=(\d+) conflict=(\d+) wait=(\d+) '
                      r'orphan=(\d+)', p.stdout)
        if m:
            log.append(f"    operand collector: 4 entries x 3 sources = 12 "
                       f"required read(s), {m.group(1)} grant(s), "
                       f"{m.group(2)} refusal(s), {m.group(4)} orphan(s)")
            ok &= int(m.group(1)) == 12 and int(m.group(4)) == 0
        else:
            log.append("    the collector bench printed no FINAL line")
            ok = False
    return ok, 'hand-counted instructions and reads, matched exactly'


def count02(log, ctx):
    """The exclusive stalls sum to the blocked cycles; overlap is separate."""
    ok = True
    differed = {'over': 0, 'under': 0}
    checked = 0
    for prog in PROGRAMS:
        r = run_prog(prog)
        if r is None:
            log.append(f"    {prog}: not built, skipped")
            continue
        checked += 1
        total = r.get('stall_total')
        parts = [r.get(k, 0) for k in EXCLUSIVE]
        s = sum(parts)
        if s != total:
            log.append(f"    {prog}: exclusive sum {s} != total {total}")
            ok = False
        # The old overlapping view, for comparison.
        legacy = r.get('hazard', 0) + r.get('lsu', 0) + r.get('wb', 0)
        if legacy > total:
            differed['over'] += 1
        elif legacy < total:
            differed['under'] += 1
        log.append(f"    {prog:<12s} total={total:<6d} exclusive sum={s:<6d}"
                   f"  overlapping h+l+wb={legacy}")
    log.append(f"    {checked} program(s); the exclusive sum equals the "
               f"total in every one")
    ok &= checked >= 6

    # The exclusivity must not be vacuous. If the old overlapping view had
    # agreed everywhere, there would be nothing here to fix and nothing to
    # test; it disagrees in both directions, which is the evidence that the
    # attribution was needed.
    log.append(f"    the overlapping view overcounts on "
               f"{differed['over']} program(s) and undercounts on "
               f"{differed['under']}")
    ok &= differed['over'] >= 1 and differed['under'] >= 1

    # And every exclusive counter must be a real branch of the priority
    # encoder, not a wire tied low. Checked in the RTL rather than by
    # waiting for a program that happens to hit it -- the contract says so:
    # "不能仅因整机计数为0就声称分支已验证".
    src = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_core.v')).read()
    # Whitespace-insensitive: the RTL aligns these into a column, so an
    # exact-string search reports seven of the nine as missing and the
    # check fails for a reason that has nothing to do with the design.
    flat = re.sub(r'\s+', ' ', src)
    missing = [k for k in EXCLUSIVE
               if f"perf_{k} <= perf_{k} + 64'd1" not in flat]
    log.append(f"    all {len(EXCLUSIVE)} exclusive counters increment "
               f"somewhere in the RTL: {not missing}"
               + (f"  MISSING {missing}" if missing else ''))
    ok &= not missing
    return ok, 'the exclusive group sums to the total on every program'


def count05(log, ctx):
    """Counting has no enable, so there is no configuration that changes."""
    ok = True
    spec = json.load(open(SPEC))
    rule = spec['sampling']['rule']
    log.append(f"    the contract's rule: {rule[:66]}...")

    src = open(os.path.join(ROOT, 'rtl', 'core', 'tnca_core.v')).read()
    # No counter may be gated by anything but the run condition. A counter
    # behind a configuration bit is a counter that changes behaviour when
    # you turn it on, which is the thing COUNT-05 forbids.
    gated = re.findall(r'if \(([^)]*perf_en[^)]*)\)', src)
    log.append(f"    no counter is behind a sampling enable: {not gated}")
    ok &= not gated

    # Two runs of the same program give the same numbers AND the same
    # result: the counters cannot perturb what they measure because they
    # are the only thing reading their own inputs.
    a = run_prog('t25_mmasf')
    b = run_prog('t25_mmasf')
    same = all(a.get(k) == b.get(k) for k in
               ('cycles', 'issued', 'stall_total', 'squash'))
    log.append(f"    two runs of t25_mmasf agree on every counter: {same}")
    ok &= same and a['_passed'] and b['_passed']

    # The widths in the contract must be the widths in the RTL: a counter
    # declared 64 bits and implemented as 32 would wrap where the contract
    # says it does not.
    declared = {}
    for group in spec['groups'].values():
        for name, entry in group.items():
            if isinstance(entry, dict) and 'width' in entry:
                declared[name] = entry['width']
    wrong = []
    for name, width in declared.items():
        if not name.startswith('perf_'):
            continue
        if f"output reg  [{width - 1}:0]              {name}" not in src:
            wrong.append(name)
    log.append(f"    {len(declared)} counter(s) declared in the contract; "
               f"{len(wrong)} core counter(s) whose width does not match "
               f"the RTL")
    if wrong:
        log.append(f"        {wrong[:6]}")
        ok = False
    return ok, 'no enable, reproducible, and the declared widths are real'


TESTS = [('COUNT-01', count01), ('COUNT-02', count02),
         ('COUNT-05', count05)]

NOT_YET = {
    'COUNT-03': 'per-link and per-VN NoC counters, and the separate DRAM '
                'path, need counters in the mesh that do not exist; the '
                'four-tile bench reports traffic in aggregate',
    'COUNT-04': 'a region of interest with a common clock and a '
                'reconstructable total needs the launch/drain timestamps '
                'of R3-06\'s other half, which is not built',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3-06 counters, their events and their arithmetic")
    if not os.path.exists(VVP):
        print(f"  the core bench is not built ({VVP}); run `make core`")
        return 1
    ctx = {}
    oc = os.path.join(BUILD, 'tb_oc_count.vvp')
    b = subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'),
         '-y', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'tb', 'common'), '-o', oc,
         os.path.join(ROOT, 'tb', 'core', 'tb_operand_collector.v')],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)
    if b.returncode == 0:
        ctx['oc_vvp'] = oc

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
        print(f"  r3-count-tests         FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-count-tests         PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
