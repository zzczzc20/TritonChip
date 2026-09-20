#!/usr/bin/env python3
"""
The timing scout -- doc/phase3_tasks_and_acceptance.md R3-01 PLAN-03.

    "阶段开始必须实际测量每类 bench 的 cycles/second 和 wall time，并据此控制
     执行成本"

R3's whole shape follows from one fact: RTL simulation is slow, and how slow
is a measurement rather than a guess. This measures it, per bench, before any
budget is written -- because a budget derived from a remembered number is a
budget for a machine nobody is using.

Three numbers, kept apart on purpose, because the contract says they are
three different things and reports conflate them:

    wall time         how long the simulation took, in seconds
    device cycles     how many clock cycles the hardware model advanced
    cycles/second     the ratio, which is a property of THIS host and
                      this bench and nothing else

None of them is a clock frequency. A design that simulates at 4,000
cycles/second says nothing about what it would run at in silicon, and this
file never prints a number that could be mistaken for one.
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

CYCLES_RE = re.compile(r'cycles=(\d+)')

# One representative program per bench class. Small on purpose: the scout
# measures the RATE, and a long run measures the same rate more slowly.
PROBES = [
    ('core-fast', 'sim/build/tb_core.vvp', 'sim/build/t02_mem.hex', []),
    ('core-slow', 'sim/build/tb_core_slow.vvp', 'sim/build/t02_mem.hex', []),
    ('core-ooo', 'sim/build/tb_core_ooo.vvp', 'sim/build/t02_mem.hex', []),
    ('core-matrix', 'sim/build/tb_core.vvp', 'sim/build/t11_matrix.hex', []),
    ('core-tensor', 'sim/build/tb_core.vvp', 'sim/build/t12_tensor.hex', []),
    ('core4', 'sim/build/tb_core4.vvp', 'sim/build/t13_share.hex',
     ['+allroles', '+aliasvrf']),
    ('system-2x2', 'sim/build/tb_top.vvp', 'sim/build/t10_mesh.hex',
     ['+needevict', '+needmesh']),
]


def probe(name, bench, image, extra, repeats=1):
    bench_path = os.path.join(ROOT, bench)
    image_path = os.path.join(ROOT, image)
    if not os.path.exists(bench_path):
        return dict(name=name, error=f"{bench} not built")
    if not os.path.exists(image_path):
        return dict(name=name, error=f"{image} not built")

    best = None
    for _ in range(repeats):
        started = time.time()
        r = subprocess.run(
            ['vvp', bench_path, f'+hex={image_path}', '+maxcycles=2000000',
             *extra],
            capture_output=True, text=True, timeout=1800)
        wall = time.time() - started
        m = CYCLES_RE.search(r.stdout)
        cycles = int(m.group(1)) if m else None
        ok = 'PASS' in r.stdout
        rec = dict(name=name, bench=bench, image=os.path.basename(image),
                   wall_seconds=round(wall, 3), device_cycles=cycles,
                   passed=ok)
        if cycles and wall > 0:
            rec['cycles_per_second'] = round(cycles / wall)
        if best is None or (rec.get('cycles_per_second', 0) >
                            best.get('cycles_per_second', 0)):
            best = rec
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()

    print("R3-01 timing scout: how fast this host simulates, measured")
    print(f"    host: {os.uname().sysname} {os.uname().machine}, "
          f"{os.cpu_count()} cpu(s)")
    rows = []
    for name, bench, image, extra in PROBES:
        if args.only and name not in args.only:
            continue
        rec = probe(name, bench, image, extra, args.repeats)
        rows.append(rec)
        if rec.get('error'):
            print(f"  {name:<14s} -- {rec['error']}", flush=True)
            continue
        print(f"  {name:<14s} {rec['device_cycles']:>8d} cycles  "
              f"{rec['wall_seconds']:>7.2f} s  "
              f"{rec.get('cycles_per_second', 0):>8d} cycles/s  "
              f"{'PASS' if rec['passed'] else 'FAIL'}", flush=True)

    usable = [r for r in rows if r.get('cycles_per_second')]
    print()
    if usable:
        slowest = min(usable, key=lambda r: r['cycles_per_second'])
        fastest = max(usable, key=lambda r: r['cycles_per_second'])
        print(f"    slowest: {slowest['name']} at "
              f"{slowest['cycles_per_second']} cycles/s")
        print(f"    fastest: {fastest['name']} at "
              f"{fastest['cycles_per_second']} cycles/s")
        print()
        print("    These are simulation rates on this host. They are NOT a")
        print("    clock frequency and say nothing about silicon; the")
        print("    contract keeps wall time, device cycles and physical")
        print("    frequency as three separate numbers for that reason.")
    if args.json:
        import r1lib as R
        R.write_json(args.json, dict(
            host=dict(system=os.uname().sysname, machine=os.uname().machine,
                      cpus=os.cpu_count()),
            measured_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            probes=rows))
    return 0 if usable else 1


if __name__ == '__main__':
    sys.exit(main())
