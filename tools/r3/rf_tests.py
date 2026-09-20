#!/usr/bin/env python3
"""
RF-01 .. RF-05 -- doc/phase3_tasks_and_acceptance.md R3-03.

The stimulus is generated here, replayed by tb/core/tb_rf_bank.v, and
checked against tools/r3/rf_model.py, which was written from the contract
rather than from the RTL. Every accepted read is compared response by
response, at both read latencies, over three seeds with at most 256
requests each, exactly as RF-02 asks.

Where a register has never been written the model says "no claim" and the
RTL says X. Those are not compared -- an X is not a value -- but RF-03
turns that into its own case: an X that became a 0 would be a 2-state
simulation quietly covering up uninitialised state, and the contract names
that specifically ("不以2-state零初始化掩盖").
"""
import argparse
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
from rf_model import BankedRF, LANES, MASK32             # noqa: E402

BENCH = os.path.join(ROOT, 'tb', 'core', 'tb_rf_bank.v')
NUM_PORTS = 4
PHYS_VREGS = 128
NUM_BANKS = 4
VREGS_PER_RCTX = 64


def build(out):
    argv = ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
            '-y', os.path.join(ROOT, 'rtl', 'core'),
            '-y', os.path.join(ROOT, 'rtl', 'common'),
            '-y', os.path.join(ROOT, 'tb', 'common'),
            '-o', out, BENCH]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=1800,
                       cwd=ROOT)
    return r


def gen(seed, cycles, bases, *, hotspot=False, warm=40):
    """One stimulus program, and the requests it contains.

    The first `warm` cycles write and do not read. Without them most reads
    land on registers nothing has written yet, the model says "no claim",
    and a run of 139 requests compares 188 lanes out of 2224 -- a green
    result that checked almost nothing. Warming the file first is the
    difference between the comparison passing and the comparison happening.
    """
    rng = random.Random(seed)
    lines = [f"{bases[0]} {bases[1]}"]
    program = []
    word = 0x1000
    for n in range(warm):
        rctx = n % 2
        idx = (n // 2) % VREGS_PER_RCTX
        word = (word + 0x40) & MASK32
        lines.append(f"1 {rctx} {idx} ffff {word} " + ' '.join(['0'] * 12))
        program.append(((rctx, idx, (1 << LANES) - 1, word), []))
    for n in range(cycles):
        # A hot address makes same-bank conflicts and read-during-write
        # happen on purpose rather than by luck (RF-02 asks for that).
        def pick():
            if hotspot and rng.random() < 0.5:
                return 0, 3
            return rng.randrange(2), rng.randrange(VREGS_PER_RCTX)

        do_write = rng.random() < 0.6
        if do_write:
            wr, wi = pick()
            wmask = rng.getrandbits(LANES)
            if rng.random() < 0.3:
                wmask = (1 << LANES) - 1        # a full write, often
            word = (word + 0x40) & MASK32
            write = (wr, wi, wmask, word)
        else:
            wr, wi, wmask, word_ = 0, 0, 0, 0
            write = None

        reads = []
        fields = []
        for p in range(NUM_PORTS):
            if rng.random() < 0.55:
                rr, ri = pick()
                reads.append((p, rr, ri))
                fields += [1, rr, ri]
            else:
                fields += [0, 0, 0]

        w = write or (0, 0, 0, 0)
        lines.append(f"{1 if write else 0} {w[0]} {w[1]} "
                     f"{w[2]:04x} {w[3]} " + ' '.join(str(x) for x in fields))
        program.append((write, reads))
    return '\n'.join(lines) + '\n', program


def replay(vvp, stim_text, tmp):
    path = os.path.join(tmp, 'stim.txt')
    with open(path, 'w') as fh:
        fh.write(stim_text)
    r = subprocess.run(['vvp', vvp, f'+stim={path}'], capture_output=True,
                       text=True, timeout=1800)
    return r


def parse(out):
    acc, rsp = {}, {1: {}, 2: {}}
    for line in out.splitlines():
        m = re.match(r'ACC (\d+) (\d+) (\d+)', line)
        if m:
            acc.setdefault(int(m.group(1)), {})[int(m.group(2))] = \
                int(m.group(3))
            continue
        m = re.match(r'RSP([12]) (\d+) (\d+) ([0-9a-fA-FxXzZ]+)', line)
        if m:
            # keyed port -> cycle, which is how compare() reads it
            rsp[int(m.group(1))].setdefault(int(m.group(3)), {})[
                int(m.group(2))] = m.group(4)
    return acc, rsp


def lanes_of(hexstr):
    """The hex response split into lanes, with X lanes as None."""
    per = 8                       # 32 bits
    total = LANES * per
    h = hexstr.rjust(total, '0')
    out = []
    for i in range(LANES):
        # lane 0 is the low-order end
        chunk = h[total - (i + 1) * per: total - i * per]
        if any(c in 'xXzZ' for c in chunk):
            out.append(None)
        else:
            out.append(int(chunk, 16))
    return out


def compare(program, acc, rsp, latency, bases, log, label):
    """Every accepted read, response by response."""
    model = BankedRF(latency=latency, vbase=bases)
    problems = []
    checked = unknown = 0
    pending = {}
    for n, (write, reads) in enumerate(program):
        want_acc, due = model.cycle(n, write, reads)
        got_acc = acc.get(n, {})
        for port, _r, _i in reads:
            if got_acc.get(port, 0) != (1 if port in want_acc else 0):
                problems.append(
                    f"cycle {n} port {port}: accepted="
                    f"{got_acc.get(port)} model says "
                    f"{1 if port in want_acc else 0}")
        for port, value in due:
            pending[(n, port)] = value
        for port, value in due:
            got = rsp.get(port, {}).get(n)
            if got is None:
                problems.append(f"cycle {n} port {port}: no response, the "
                                f"model expected one")
                continue
            got_lanes = lanes_of(got)
            for lane in range(LANES):
                if value[lane] is None:
                    unknown += 1
                    continue
                checked += 1
                if got_lanes[lane] != value[lane]:
                    problems.append(
                        f"cycle {n} port {port} lane {lane}: "
                        f"{got_lanes[lane]} != {value[lane]}")
    # and the RTL must not answer when the model says nothing is due
    for port, by_cycle in rsp.items():
        for n in by_cycle:
            if (n, port) not in pending and n < len(program):
                problems.append(f"cycle {n} port {port}: a response the "
                                f"model did not expect")
    log.append(f"    {label}: {checked} lane comparison(s), "
               f"{unknown} against never-written state (not compared), "
               f"{len(problems)} problem(s)")
    for p in problems[:6]:
        log.append(f"        {p}")
    return not problems


def rf01(log, ctx):
    """Capacity, an invertible mapping, boundaries and role aliasing."""
    ok = True
    m = BankedRF()
    cap = NUM_BANKS * m.bank_depth * LANES * 4
    log.append(f"    {NUM_BANKS} bank(s) x {m.bank_depth} row(s) x "
               f"{LANES} lane(s) x 4 bytes = {cap} bytes "
               f"({cap // 1024} KiB)")
    ok &= cap == 8192
    bad = [p for p in range(PHYS_VREGS) if m.unmap(*m.bank_row(p)) != p]
    log.append(f"    the mapping is invertible for all {PHYS_VREGS} "
               f"registers: {not bad}")
    ok &= not bad
    # every bank is used, and used equally
    counts = {}
    for p in range(PHYS_VREGS):
        b, _r = m.bank_row(p)
        counts[b] = counts.get(b, 0) + 1
    log.append(f"    registers per bank: {sorted(counts.values())}")
    ok &= len(set(counts.values())) == 1 and len(counts) == NUM_BANKS

    # Role aliasing: equal bases alias completely, disjoint bases never do.
    same = BankedRF(vbase=(0, 0))
    dis = BankedRF(vbase=(0, 64))
    alias = all(same.phys(0, i) == same.phys(1, i)
                for i in range(VREGS_PER_RCTX))
    overlap = {dis.phys(0, i) for i in range(VREGS_PER_RCTX)} & \
              {dis.phys(1, i) for i in range(VREGS_PER_RCTX)}
    log.append(f"    equal bases alias every name: {alias}; disjoint bases "
               f"share {len(overlap)} register(s)")
    ok &= alias and not overlap

    # The boundary: a base near the top wraps within the physical file
    # rather than addressing past it.
    edge = BankedRF(vbase=(PHYS_VREGS - 4, 0))
    addrs = [edge.phys(0, i) for i in range(8)]
    log.append(f"    base={PHYS_VREGS - 4}, idx 0..7 -> {addrs}")
    ok &= all(0 <= a < PHYS_VREGS for a in addrs)
    return ok, f'{cap // 1024} KiB in {NUM_BANKS} banks, mapping invertible'


def rf02(log, ctx):
    """Random sequences, three seeds, both latencies, per request."""
    ok = True
    for seed in (1, 7, 42):
        for hot in (False, True):
            stim, program = gen(seed, 64, ctx['bases'], hotspot=hot)
            warm_cycles = len(program) - 64
            r = replay(ctx['vvp'], stim, ctx['tmp'])
            if r.returncode != 0:
                log.append(f"    seed {seed}: the bench exited "
                           f"{r.returncode}: {r.stderr[-200:]}")
                return False, 'the bench did not run'
            acc, rsp = parse(r.stdout)
            n_req = sum(len(reads) for _w, reads in program)
            for lat in (1, 2):
                good = compare(program, acc, rsp[lat], lat, ctx['bases'],
                               log,
                               f"seed {seed} {'hot' if hot else 'spread'} "
                               f"latency {lat} ({n_req} request(s), "
                               f"{warm_cycles} warm-up write(s))")
                ok &= good
            if not hot and seed == 1:
                quiet = []
                fooled = compare(program, acc, rsp[1], 2, ctx['bases'],
                                 quiet, 'control (latency 2 vs latency 1)')
                log.append(f"    control: latency-1 responses checked "
                           f"against a latency-2 model -> "
                           f"{'DISAGREE (good)' if not fooled else 'agreed, so the comparison is not looking (bad)'}")
                ok &= not fooled
            if n_req > 256:
                log.append(f"    seed {seed}: {n_req} requests exceeds the "
                           f"256 the contract allows")
                ok = False
    return ok, 'three seeds, two latencies, every request compared'


def rf03(log, ctx):
    """Never-written state reads as unknown, not as a convenient zero."""
    # One read of a register nothing has written, and nothing else.
    lines = [f"{ctx['bases'][0]} {ctx['bases'][1]}"]
    lines.append("0 0 0 0000 0 1 0 9 0 0 0 0 0 0 0 0 0")
    for _ in range(4):
        lines.append("0 0 0 0000 0 0 0 0 0 0 0 0 0 0 0 0 0")
    r = replay(ctx['vvp'], '\n'.join(lines) + '\n', ctx['tmp'])
    acc, rsp = parse(r.stdout)
    first = None
    for port_map in rsp[1].values():
        for _cyc, val in sorted(port_map.items()):
            first = val
            break
        if first:
            break
    unknown = first is not None and any(c in 'xXzZ' for c in first)
    log.append(f"    a register never written reads as "
               f"{'X (unknown)' if unknown else repr(first)}")
    if not unknown:
        log.append("    it reads as a value, which would hide exactly the "
                   "uninitialised state RF-03 is looking for")
    return unknown, 'uninitialised state stays unknown'


def rf04(log, ctx):
    """No more accepted than the ports exist, and no response goes astray."""
    ok = True
    # Four requests, all to the same bank, every cycle: at most one may be
    # accepted, and exactly one response may come back.
    lines = [f"{ctx['bases'][0]} {ctx['bases'][1]}"]
    # idx 0, 4, 8, 12 are all bank 0 with base 0
    for _ in range(8):
        lines.append("1 0 0 ffff 4096 1 0 0 1 0 4 1 0 8 1 0 12")
    for _ in range(4):
        lines.append("0 0 0 0000 0 0 0 0 0 0 0 0 0 0 0 0 0")
    r = replay(ctx['vvp'], '\n'.join(lines) + '\n', ctx['tmp'])
    acc, rsp = parse(r.stdout)
    worst = 0
    for n, ports in acc.items():
        taken = sum(ports.values())
        worst = max(worst, taken)
    log.append(f"    four requests to one bank, every cycle: at most "
               f"{worst} accepted per cycle")
    ok &= worst == 1
    for lat in (1, 2):
        per_cycle = {}
        for port, by_cycle in rsp[lat].items():
            for n in by_cycle:
                per_cycle[n] = per_cycle.get(n, 0) + 1
        most = max(per_cycle.values()) if per_cycle else 0
        log.append(f"    latency {lat}: at most {most} response(s) in any "
                   f"cycle, {sum(per_cycle.values())} in total")
        ok &= most <= 1
    return ok, 'one bank, one read, one response'


def rf05(log, ctx):
    """A configuration the storage cannot implement is refused, not fudged."""
    ok = True
    # NUM_BANKS that does not divide PHYS_VREGS: the banks would not sum
    # to the file, and the module must say so rather than address part of
    # it. 100 in 8 banks is 12 rows each, which reaches 96 registers and
    # leaves 4 of them unaddressable. (The first attempt at this case used
    # 100 in 4 banks, which divides exactly -- a configuration that is
    # fine, tested as though it were not.)
    src = os.path.join(ctx['tmp'], 'bad_cfg.v')
    with open(src, 'w') as fh:
        fh.write('''`timescale 1ns/1ps
module tb_bad;
  reg clk=0, rst_n=1;
  wire [3:0] acc, rv;
  wire [4*512-1:0] rd;
  tnca_vrf_banked #(.PHYS_VREGS(100), .NUM_BANKS(8), .BANK_W(3))
    u (.clk(clk), .rst_n(rst_n), .vbase(14'd0), .req_valid(4'd0),
       .req_rctx(4'd0), .req_idx(32'd0), .req_accept(acc),
       .rsp_valid(rv), .rsp_data(rd), .we(1'b0), .w_rctx(1'b0),
       .w_idx(8'd0), .wmask(16'd0), .wdata(512'd0));
  initial begin #100; $display("NOT REFUSED"); $finish; end
endmodule
''')
    out = os.path.join(ctx['tmp'], 'bad.vvp')
    b = subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'),
         '-y', os.path.join(ROOT, 'rtl', 'common'), '-o', out, src],
        capture_output=True, text=True, timeout=900, cwd=ROOT)
    if b.returncode == 0:
        r = subprocess.run(['vvp', out], capture_output=True, text=True,
                           timeout=300)
        refused = 'not PHYS_VREGS' in r.stdout
        log.append(f"    100 registers in 8 banks of 12: "
                   f"{'refused at elaboration' if refused else 'ACCEPTED (bad)'}")
        if refused:
            first = [l for l in r.stdout.splitlines() if 'banks' in l]
            if first:
                log.append(f"        {first[0].strip()}")
        ok &= refused
    else:
        log.append(f"    the bad configuration would not even compile: "
                   f"{b.stderr.strip().splitlines()[0][:70] if b.stderr else ''}")

    # And the write mask: every lane the mask does not name must keep its
    # old value, which is what makes a masked write a masked write rather
    # than a read-modify-write nobody costed.
    lines = [f"{ctx['bases'][0]} {ctx['bases'][1]}"]
    lines.append("1 0 5 ffff 8192 0 0 0 0 0 0 0 0 0 0 0 0")   # fill v5
    lines.append("1 0 5 000f 4096 0 0 0 0 0 0 0 0 0 0 0 0")   # 4 lanes only
    lines.append("0 0 0 0000 0 1 0 5 0 0 0 0 0 0 0 0 0")      # read it back
    for _ in range(4):
        lines.append("0 0 0 0000 0 0 0 0 0 0 0 0 0 0 0 0 0")
    r = replay(ctx['vvp'], '\n'.join(lines) + '\n', ctx['tmp'])
    _acc, rsp = parse(r.stdout)
    val = None
    for by_cycle in rsp[1].values():
        for _n, v in sorted(by_cycle.items()):
            val = v
            break
        if val:
            break
    if val is None:
        log.append("    the masked write case produced no response")
        return False, 'no response'
    got = lanes_of(val)
    want = [(4096 + i) if i < 4 else (8192 + i) for i in range(LANES)]
    log.append(f"    a 4-lane masked write over a full one: "
               f"lanes 0..3 = {got[:4]}, lanes 4..5 = {got[4:6]}")
    ok &= got == want
    if got != want:
        log.append(f"        wanted {want[:6]}...")
    return ok, 'an impossible configuration refused, a masked write masked'


TESTS = [('RF-01', rf01), ('RF-02', rf02), ('RF-03', rf03),
         ('RF-04', rf04), ('RF-05', rf05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    print("R3-03 synchronous banked VRF")
    tmp = tempfile.mkdtemp(prefix='r3-rf-')
    vvp = os.path.join(tmp, 'tb_rf_bank.vvp')
    b = build(vvp)
    if b.returncode != 0:
        print(f"  the bench did not build: {b.stderr[-400:]}")
        return 1
    ctx = {'vvp': vvp, 'tmp': tmp, 'bases': (0, 64)}
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
    if args.only and not results:
        print(f"  the --only filter {args.only} matched nothing; the known "
              f"cases are: {', '.join(t[0] for t in TESTS)}")
        return 2
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r3-rf-tests            FAIL   {', '.join(bad)}")
        return 1
    print(f"  r3-rf-tests            PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
