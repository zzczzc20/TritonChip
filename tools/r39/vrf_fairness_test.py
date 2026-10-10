#!/usr/bin/env python3
"""R39-00 B2: every read port of the banked register file is served.

tb_r3v2_vrf case 7 has every port ask for the same bank in every cycle, so the
bank grants one of them per cycle and the round robin decides which. Built
with NUM_PORTS = 4 (the bench's old fixed width) and 5 (what the core
instantiates: three operands, the matrix, the transfer engine's store read),
each port must be granted at least once in every NUM_PORTS consecutive
cycles. With the round-robin pointer two bits wide, port 4 was never first in
the order and was never granted while the others kept asking.

Case 8 (R39-05) serves two banks in one cycle: port 0 alone on one, ports 1
and 3 on the other. The pointer used to move past the LAST port served, which
in the order 1 2 3 0 put it back on 1 every cycle, and port 3 -- the matrix --
never won its bank. It must now be served within NUM_PORTS cycles.

    python3 tools/r39/vrf_fairness_test.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(ROOT, 'tb', 'core', 'tb_r3v2_vrf.v')


def run(ports, tmp, case=7):
    out = os.path.join(tmp, f'vrf{ports}.vvp')
    b = subprocess.run(
        ['iverilog', '-g2005', '-I', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'rtl', 'core'), '-y', os.path.join(ROOT, 'rtl', 'common'),
         '-y', os.path.join(ROOT, 'tb', 'common'),
         '-P', f'tb_r3v2_vrf.NUM_PORTS={ports}', '-o', out, BENCH],
        capture_output=True, text=True, cwd=ROOT)
    if b.returncode != 0:
        return None, f'build failed: {b.stderr[-300:]}'
    r = subprocess.run(['vvp', out, f'+case={case}'], capture_output=True, text=True,
                       timeout=600, cwd=ROOT)
    grants = []
    for line in r.stdout.splitlines():
        m = re.match(r'GRANT (\d+) ([01]+)', line)
        if m:
            bits = m.group(2)[::-1]            # bit p is port p
            grants.append([p for p, b in enumerate(bits) if b == '1'])
    return grants, None


def check(ports, grants):
    served = [g for g in grants if g]
    if not served:
        return False, 'no grant at all'
    if any(len(g) != 1 for g in served):
        return False, f'more than one grant on one bank in a cycle: {served}'
    order = [g[0] for g in served]
    for i in range(len(order) - ports + 1):
        window = set(order[i:i + ports])
        if window != set(range(ports)):
            missing = sorted(set(range(ports)) - window)
            return False, (f'ports {missing} not served in grants {i}..{i + ports - 1}: '
                           f'order {order}')
    return True, f'{len(order)} grants, every window of {ports} serves every port'


def check_two_banks(ports, grants):
    """Case 8: ports 0, 1 and 3 ask every cycle, 1 and 3 on one bank. Port 3
    must be granted within every `ports` consecutive cycles."""
    gaps, last = [], -1
    for c, g in enumerate(grants):
        if 3 in g:
            gaps.append(c - last)
            last = c
    gaps.append(len(grants) - last)
    worst = max(gaps)
    if not any(3 in g for g in grants):
        return False, 'port 3 (the matrix) never granted while ports 0 and 1 kept asking'
    return worst <= ports, f'port 3 granted {sum(3 in g for g in grants)} times, longest wait {worst} cycles'


def main():
    ok = True
    with tempfile.TemporaryDirectory(prefix='r39-fair-') as tmp:
        for ports in (4, 5):
            grants, err = run(ports, tmp)
            if err:
                print(f'  {ports} ports: FAIL {err}')
                ok = False
                continue
            good, why = check(ports, grants)
            print(f"  {ports} ports: {'PASS' if good else 'FAIL'}  {why}")
            ok &= good
            grants, err = run(ports, tmp, case=8)
            good, why = (False, err) if err else check_two_banks(ports, grants)
            print(f"  {ports} ports, two banks: {'PASS' if good else 'FAIL'}  {why}")
            ok &= good
    print(f"vrf-fairness           {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
