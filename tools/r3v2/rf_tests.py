#!/usr/bin/env python3
"""
R3V2-RF-01, -02 and -03 -- doc/phase3_tasks_and_acceptance.md G1.

The candidate's storage semantics: capacity and addressing, a per-request
comparison against tools/r3v2/vrf_model.py, and the initialisation
contract.

RF-03 is the one worth reading. It asks for two things that sound
contradictory:

    the architecture reads zero from a register nobody wrote
    the storage still holds X, so a 2-state zero is not hiding in it

The behavioural file this replaces satisfies the first by accident -- a
2-state Verilog array starts at zero -- and cannot demonstrate the second
at all. The candidate does both on purpose: per-lane valid bits, cleared
by reset, with the data array untouched. tb_r3v2_vrf.v elaborates the same
wrapper twice, with the substitution on and off, so both claims are
checked rather than one being asserted.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                        # noqa: E402
from vrf_model import CandidateVRF, LANES, MASK32        # noqa: E402

BENCH = os.path.join(ROOT, 'tb', 'core', 'tb_r3v2_vrf.v')
CONTRACT = json.load(open(os.path.join(ROOT, 'spec',
                                       'r3v2_contract.json')))
NUM_PORTS = 4
SEEDS = (1, 7, 42)
# R3V2-RF-02: "有界序列", and section 4.3 caps a component scenario at 256
# accepted requests.
MAX_REQUESTS = 256


def build(out):
    return subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'),
         '-y', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'tb', 'common'), '-o', out, BENCH],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)


def run_case(vvp, case, extra=()):
    r = subprocess.run([vvp, f'+case={case}'] + list(extra),
                       capture_output=True, text=True, timeout=900)
    return r.stdout


def lanes_of(hexstr):
    per = ELEN_HEX = 8
    total = LANES * per
    h = hexstr.rjust(total, '0')
    out = []
    for i in range(LANES):
        chunk = h[total - (i + 1) * per: total - i * per]
        out.append(None if any(c in 'xXzZ' for c in chunk)
                   else int(chunk, 16))
    return out


def rf01(log, ctx):
    """Capacity, banking, the inverse mapping, role aliasing, port limits."""
    c = CONTRACT['vrf']
    m = CandidateVRF()
    ok = True
    bits = c['physical_registers'] * c['lanes'] * c['element_bits']
    log.append(f"    {c['physical_registers']} register(s) x {c['lanes']} "
               f"lane(s) x {c['element_bits']} bit = {bits} bit = "
               f"{bits // 8192} KiB; the contract says {c['data_bits']}")
    ok &= bits == c['data_bits'] == 65536
    log.append(f"    {c['banks']} bank(s) x {c['rows_per_bank']} row(s) = "
               f"{c['banks'] * c['rows_per_bank']} register(s)")
    ok &= c['banks'] * c['rows_per_bank'] == c['physical_registers']

    bad = [p for p in range(c['physical_registers'])
           if m.unmap(*m.bank_row(p)) != p]
    log.append(f"    the address mapping is invertible for all "
               f"{c['physical_registers']}: {not bad}")
    ok &= not bad

    same = CandidateVRF(vbase=(0, 0))
    disj = CandidateVRF(vbase=(0, 64))
    alias = all(same.phys(0, i) == same.phys(1, i) for i in range(64))
    overlap = {disj.phys(0, i) for i in range(64)} & \
              {disj.phys(1, i) for i in range(64)}
    log.append(f"    equal role bases alias every name: {alias}; disjoint "
               f"bases share {len(overlap)}")
    ok &= alias and not overlap

    edge = CandidateVRF(vbase=(c['physical_registers'] - 4, 0))
    addrs = [edge.phys(0, i) for i in range(8)]
    log.append(f"    a base at the top wraps inside the file: {addrs}")
    ok &= all(0 <= a < c['physical_registers'] for a in addrs)

    # One accepted read and one accepted write per bank per cycle, and
    # every response delivered exactly once. Checked in the RTL, with four
    # requests aimed at one bank every cycle.
    out = run_case(ctx['vvp'], 5, [f"+stim={ctx['one_bank_stim']}"])
    accepts = [len([b for b in mm.group(2)[::-1] if b == '1'])
               for mm in re.finditer(r'ACC (\d+) ([01]+)', out)]
    responses = re.findall(r'RSP (\d+) (\d+) ', out)
    log.append(f"    four requests at one bank every cycle: at most "
               f"{max(accepts) if accepts else 0} accepted per cycle")
    ok &= (max(accepts) if accepts else 99) <= 1
    per_cycle = {}
    for cyc, port in responses:
        per_cycle[cyc] = per_cycle.get(cyc, 0) + 1
    log.append(f"    {len(responses)} response(s), at most "
               f"{max(per_cycle.values()) if per_cycle else 0} in any one "
               f"cycle, each appearing once")
    ok &= (max(per_cycle.values()) if per_cycle else 0) <= 1
    return ok, '8 KiB in 4 banks, invertible, one read and one response'


def gen_stim(seed, cycles, *, one_bank=False, stall=True):
    """A bounded stimulus, and the same program for the model."""
    rng = random.Random(seed)
    lines, program = [], []
    word = 0x1000
    accepted_budget = MAX_REQUESTS
    for _ in range(cycles):
        if one_bank:
            # phys 0, 4, 8, 12 are all bank 0 with base 0
            idxs = [0, 4, 8, 12]
        else:
            idxs = [rng.randrange(64) for _ in range(NUM_PORTS)]

        do_write = rng.random() < 0.55
        if do_write:
            wi = 3 if (one_bank or rng.random() < 0.4) else rng.randrange(64)
            wmask = rng.getrandbits(LANES)
            if rng.random() < 0.3:
                wmask = (1 << LANES) - 1
            word = (word + 0x40) & MASK32
            write = (0, wi, wmask, word)
        else:
            wi, wmask = 0, 0
            write = None

        # One port at a time keeps the accepted count inside the cap and
        # makes each property attributable; the one-bank case uses all four.
        live = NUM_PORTS if one_bank else 1 + (rng.random() < 0.5)
        rv = 0
        reads = []
        for p in range(live):
            if accepted_budget <= 0:
                break
            rv |= (1 << p)
            reads.append((p, 0, idxs[p]))
        ready_mask = (1 << NUM_PORTS) - 1
        if stall and rng.random() < 0.25:
            ready_mask = 0          # nobody takes a response this cycle
        accepted_budget -= len(reads)

        lines.append(f"{1 if write else 0} {wi} {wmask:04x} {word} "
                     f"{rv} " + ' '.join(str(x) for x in idxs) +
                     f" {ready_mask}")
        program.append((write, reads,
                        {p: bool(ready_mask >> p & 1)
                         for p in range(NUM_PORTS)}))
    return '\n'.join(lines) + '\n', program


def rf02(log, ctx):
    """Every accepted read, compared lane by lane against the model."""
    ok = True
    triggered = {'mask': 0, 'rdw': 0, 'conflict': 0, 'stall': 0,
                 'back_to_back': 0}
    for seed in SEEDS:
        stim, program = gen_stim(seed, 60)
        path = os.path.join(ctx['tmp'], f'stim{seed}.txt')
        open(path, 'w').write(stim)
        out = run_case(ctx['vvp'], 5, [f'+stim={path}'])

        model = CandidateVRF()
        got = {}
        for mm in re.finditer(r'RSP (\d+) (\d+) ([0-9a-fA-FxX]+)', out):
            got.setdefault(int(mm.group(1)), {})[int(mm.group(2))] = \
                mm.group(3)
        # Which ports the RTL accepted, per cycle. Comparing acceptance as
        # well as data localises a divergence to the cycle it starts in;
        # comparing only the data shows it several cycles later, wearing
        # the wrong cause.
        acc_rtl = {}
        for mm in re.finditer(r'ACC (\d+) ([01]+)', out):
            bits = mm.group(2)[::-1]
            acc_rtl[int(mm.group(1))] = {i for i, b in enumerate(bits)
                                         if b == '1'}

        problems, checked, unknown = [], 0, 0
        prev_read = False
        for n, (write, reads, ready) in enumerate(program):
            if write and reads:
                for _p, _r, idx in reads:
                    if idx == write[1]:
                        triggered['rdw'] += 1
            if write and write[2] not in (0, (1 << LANES) - 1):
                triggered['mask'] += 1
            if len(reads) > 1:
                triggered['conflict'] += 1
            if not all(ready.values()):
                triggered['stall'] += 1
            if reads and prev_read:
                triggered['back_to_back'] += 1
            prev_read = bool(reads)

            acc, delivered = model.cycle(write, reads, ready)
            if set(acc) != acc_rtl.get(n, set()):
                problems.append(
                    f"cycle {n}: the RTL accepted "
                    f"{sorted(acc_rtl.get(n, set()))} and the model "
                    f"{sorted(acc)}")
            for port, value in delivered:
                raw = got.get(n, {}).get(port)
                if raw is None:
                    problems.append(f"cycle {n} port {port}: the model "
                                    f"expected a response and none came")
                    continue
                lanes = lanes_of(raw)
                for lane in range(LANES):
                    want = value[lane]
                    if want is None:
                        unknown += 1
                        continue
                    checked += 1
                    if lanes[lane] != want:
                        problems.append(
                            f"cycle {n} port {port} lane {lane}: "
                            f"{lanes[lane]} != {want}")
        log.append(f"    seed {seed:<3d} {checked} lane comparison(s), "
                   f"{unknown} never-written (not compared), "
                   f"{len(problems)} problem(s)")
        for p in problems[:4]:
            log.append(f"        {p}")
        ok &= not problems

    log.append(f"    properties actually triggered: " +
               ', '.join(f"{k}={v}" for k, v in sorted(triggered.items())))
    # A sequence that never made a conflict has not tested conflicts.
    for prop, count in triggered.items():
        if count == 0:
            log.append(f"    {prop} never happened, so it was not tested")
            ok = False

    # The comparison has to be able to fail.
    stim, program = gen_stim(1, 20)
    path = os.path.join(ctx['tmp'], 'control.txt')
    open(path, 'w').write(stim)
    out = run_case(ctx['vvp'], 5, [f'+stim={path}'])
    control = CandidateVRF(zero_invalid=False)
    disagreed = False
    for n, (write, reads, ready) in enumerate(program):
        _a, delivered = control.cycle(write, reads, ready)
        for port, value in delivered:
            raw = got_l = re.search(rf'RSP {n} {port} (\S+)', out)
            if not raw:
                continue
            lanes = lanes_of(raw.group(1))
            for lane in range(LANES):
                if value[lane] is None and lanes[lane] == 0:
                    disagreed = True      # the model says X, the RTL zero
    log.append(f"    control: checked against a model WITHOUT the zero "
               f"substitution -> "
               f"{'disagrees, so the comparison is looking' if disagreed else 'AGREED (the comparison is not looking)'}")
    ok &= disagreed
    return ok, 'three seeds, every accepted read, every property triggered'


def rf03(log, ctx):
    """The initialisation contract: zero above, X underneath."""
    ok = True
    out = run_case(ctx['vvp'], 1)
    arch = re.findall(r'ARCH (\d+) v=(\d) (\S+)', out)
    raw = re.findall(r'RAW  (\d+) v=(\d) (\S+)', out)
    log.append(f"    cold reset, {len(arch)} register(s) never written:")
    for (idx, v, a), (_i, _v, r) in zip(arch, raw):
        a_zero = set(a) == {'0'}
        r_x = 'x' in r.lower()
        log.append(f"      v{idx:<3s} architecture reads "
                   f"{'zero' if a_zero else a[:16]}, storage reads "
                   f"{'X' if r_x else r[:16]}")
        ok &= a_zero and r_x
    ok &= len(arch) >= 3

    out = run_case(ctx['vvp'], 2)
    a = re.search(r'ARCH 7 v=\d (\S+)', out).group(1)
    r = re.search(r'RAW  7 v=\d (\S+)', out).group(1)
    al, rl = lanes_of(a), lanes_of(r)
    written = [i for i in range(LANES) if rl[i] is not None]
    log.append(f"    a 4-lane masked write to v7: the storage has real "
               f"data in lanes {written}")
    log.append(f"      the architecture reads {al[:6]}... -- the written "
               f"lanes, then zero")
    ok &= written == [0, 1, 2, 3]
    ok &= al[:4] == rl[:4] and all(v == 0 for v in al[4:])
    log.append(f"    a partial write did not disturb the initialisation of "
               f"the lanes it did not name")

    out = run_case(ctx['vvp'], 3)
    after = re.findall(r'AFTER \d+ arch_v=(\d+) raw_v=(\d+)', out)
    leaked = [a for a, _r in after if '1' in a]
    log.append(f"    a read in flight when reset asserts: "
               f"{len(after)} cycle(s) watched, "
               f"{len(leaked)} response(s) delivered into the new epoch")
    ok &= not leaked
    a = re.search(r'ARCH 9 v=\d (\S+)', out).group(1)
    r = re.search(r'RAW  9 v=\d (\S+)', out).group(1)
    log.append(f"    after the reset the architecture reads "
               f"{'zero' if set(a) == {'0'} else a[:16]}, while the array "
               f"still holds {r[:16]}...")
    ok &= set(a) == {'0'} and set(r) != {'0'}
    log.append(f"    so nothing leaks across the reset, and the data array "
               f"was not cleared to make that true")

    out = run_case(ctx['vvp'], 4)
    held = re.findall(r'(HELD|STILL) arch_v=(\d+) accept=(\d+) data=(\S+)',
                      out)
    log.append(f"    with rsp_ready low: {len(held)} cycle(s) holding")
    vals = set()
    for _w, v, acc, data in held:
        vals.add(data)
        ok &= '1' in v and set(acc) == {'0'}
    log.append(f"      the response stayed valid, the data stayed "
               f"{'stable' if len(vals) == 1 else 'UNSTABLE'}, and no "
               f"further read was accepted")
    ok &= len(vals) == 1
    rel = re.search(r'RELEASED arch_v=(\d+)', out)
    log.append(f"      released when the consumer was ready: "
               f"arch_v={rel.group(1) if rel else '?'}")
    ok &= bool(rel) and set(rel.group(1)) == {'0'}
    return ok, 'zero above, X underneath, nothing across a reset'


TESTS = [('R3V2-RF-01', rf01), ('R3V2-RF-02', rf02), ('R3V2-RF-03', rf03)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3 v2 G1: the candidate VRF's storage semantics")
    tmp = tempfile.mkdtemp(prefix='r3v2-rf-')
    vvp = os.path.join(tmp, 'tb.vvp')
    b = build(vvp)
    if b.returncode != 0:
        print(f"  the bench did not build: {b.stderr[-500:]}")
        return 1
    one_bank, _ = gen_stim(1, 24, one_bank=True, stall=False)
    ob = os.path.join(tmp, 'one_bank.txt')
    open(ob, 'w').write(one_bank)
    ctx = {'vvp': vvp, 'tmp': tmp, 'one_bank_stim': ob}

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
        print(f"  r3v2-rf-tests          FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3v2-rf-tests          PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
