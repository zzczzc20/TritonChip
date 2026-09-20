#!/usr/bin/env python3
"""
The Linux scout -- doc/phase3_tasks_and_acceptance.md G0, R3V2-BASE-03.

"阶段开始必须实际测量每类 bench 的 cycles/second 和 wall time，并据此控制
 执行成本." The budget numbers in section 4.3 are an engineering plan, not
a measurement, and the contract says so. This measures.

What it runs is the REFERENCE implementation -- core-sim, the profile that
is not changing -- on programs shaped like the ones the official cases
will use: an FP32 elementwise kernel and an FP16 matrix kernel, under both
the fast memory and the slow/reordering one, because C04 and C05 are
specified on slow-reorder and a rate measured only on fast memory would
not predict them.

It does NOT claim to be C01 and C03. Section 4.3 allows a scout record to
occupy two of the official twelve when the commit, artifact and config are
the same; these are existing bench programs at the right shape, not the
frozen workloads with their independent oracles, so they are recorded as
rate measurements and the official runs are still owed. Saying otherwise
would spend two of twelve on work that did not happen.
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
BUILD = os.path.join(ROOT, 'sim', 'build')

# (id, program, memory mode, what shape it stands in for)
PROBES = [
    ('vecadd-fast', 't16_vecadd', 'fast', 'C01/C02: FP32 elementwise'),
    ('matrix-fast', 't11_matrix', 'fast', 'C03: FP16 matrix'),
    ('matrix-slow', 't11_matrix', 'slow', 'C04/C05: matrix under a '
                                          'back-pressuring memory'),
    ('matrix-ooo', 't11_matrix', 'ooo', 'C04/C05: and under out-of-order '
                                        'completion'),
]

# The memory model is chosen by which bench was elaborated, not by a
# plusarg: tb_core_slow.vvp and tb_core_ooo.vvp are separate builds with
# different LATENCY and REORDER parameters.
BENCH = {
    'fast': 'tb_core.vvp',
    'slow': 'tb_core_slow.vvp',
    'ooo': 'tb_core_ooo.vvp',
}


def run_probe(probe_id, program, mode, stands_for, maxcycles):
    hexf = os.path.join(BUILD, f"{program}.hex")
    vvp = os.path.join(BUILD, BENCH[mode])
    if not (os.path.exists(hexf) and os.path.exists(vvp)):
        return {'id': probe_id,
                'error': f"{os.path.basename(vvp)} or {program}.hex is "
                         f"not built",
                'exit': None}
    argv = ['vvp', vvp, f"+hex={hexf}", f"+maxcycles={maxcycles}"]
    started = time.time()
    r = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - started

    cycles = None
    m = re.search(r'^cycles=(\d+)', r.stdout, re.M)
    if m:
        cycles = int(m.group(1))
    passed = 'PASS' in r.stdout
    return {
        'id': probe_id,
        'program': program,
        'memory': mode,
        'stands_for': stands_for,
        'command': ' '.join(argv),
        'exit': r.returncode,
        'passed': passed,
        'cycles': cycles,
        'elapsed_s': round(elapsed, 3),
        'cycles_per_second': round(cycles / elapsed) if cycles and elapsed
        else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(BUILD,
                                                   'r3v2_scout.json'))
    ap.add_argument('--maxcycles', type=int, default=50000)
    args = ap.parse_args()

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                          capture_output=True, text=True)
    commit = head.stdout.strip() if head.returncode == 0 else 'unknown'
    uname = subprocess.run(['uname', '-srm'], capture_output=True,
                           text=True).stdout.strip()

    print("R3 v2 G0 scout: what a run of this shape actually costs")
    print(f"  commit {commit[:12]}  on {uname}")
    runs = []
    for probe_id, program, mode, stands_for in PROBES:
        row = run_probe(probe_id, program, mode, stands_for, args.maxcycles)
        runs.append(row)
        if row.get('error'):
            print(f"  {probe_id:<14s} {row['error']}")
            continue
        print(f"  {probe_id:<14s} exit={row['exit']} "
              f"{'PASS' if row['passed'] else 'FAIL'} "
              f"cycles={row['cycles']:<7d} "
              f"elapsed={row['elapsed_s']:>7.2f}s  "
              f"{row['cycles_per_second']:>7d} cycles/s   "
              f"({row['stands_for']})")

    rates = [r['cycles_per_second'] for r in runs
             if r.get('cycles_per_second')]
    report = {
        'schema': 'r3v2-scout/1',
        'contract_id': 'r3-v2-core-vrf',
        'commit': commit,
        'platform': uname,
        'implementation': 'reference',
        'profile': 'core-sim',
        'runs': runs,
        'slowest_cycles_per_second': min(rates) if rates else None,
        'counts_toward_official_twelve': False,
        'why': ('These are existing bench programs at the right shape, not '
                'the frozen C01/C03 with their independent oracles. Section '
                '4.3 allows a scout record to occupy two of the official '
                'twelve only when the commit, artifact and config are the '
                'same; they are not, so the official runs are still owed.'),
    }
    with open(args.json, 'w') as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write('\n')

    if rates:
        slowest = min(rates)
        print(f"  slowest observed rate: {slowest} cycles/s")
        # What the budget means at this rate.
        print(f"  at that rate a 50,000-cycle core case costs "
              f"{50000 / slowest:.0f}s against a 180s cap, and the "
              f"1,000,000-cycle total costs {1000000 / slowest:.0f}s "
              f"against 1,800s")
    print(f"  wrote {args.json}")
    return 0 if all(r.get('exit') == 0 for r in runs) else 1


if __name__ == '__main__':
    sys.exit(main())
