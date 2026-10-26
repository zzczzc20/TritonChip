#!/usr/bin/env python3
"""Record vector-FPU results from the product RTL for the model's numeric check.

    rtl_oracle.py --root PRODUCT_TREE --out FRESH_DIR [--per-op N] [--seed S]

Offline only: iverilog runs tnca_vector_fpu over generated stimuli once, and
the recorded outputs become an oracle file. The TLM model never invokes an
RTL simulator; `tnca-numeric-probe vfpu ORACLE` then checks the executor's
own lane functions (include/tnca_vector_ops.hpp) against every recorded lane.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent

SPECIALS = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000, 0x3fc00000, 0x40200000,
            0xc0200000, 0xbfc00000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc12345, 0x7fbfffff,
            0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
            0x4b000000, 0x4b000001, 0xcb000001, 0x4b800000, 0x4b800001, 0x4b800003, 0x4c000001, 0x4e7fffff,
            0x4effffff, 0xceffffff, 0x4f000000, 0xcf000000, 0x4f7fffff, 0x4f800000, 0x3f400000, 0xbf400000,
            0x3e800000, 0xbe800000, 0x3f7fffff, 0xbf7fffff, 0x42fe0000, 0xc2fe0000, 0x43000000, 0xc3160000,
            0x34000000, 0x0c000000, 0x8c000000, 0x7e967699, 0x00400000]


def rand_float(rng):
    r = rng.random()
    if r < 0.2:
        return rng.choice(SPECIALS)
    if r < 0.45:
        return rng.getrandbits(32)
    if r < 0.6:   # integer-valued, around the conversion corners
        e = rng.randint(127, 160)
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
    if r < 0.7:   # tiny: subnormal results under products
        return (rng.getrandbits(1) << 31) | (rng.randint(0, 20) << 23) | rng.getrandbits(23)
    return (rng.getrandbits(1) << 31) | (rng.randint(100, 154) << 23) | rng.getrandbits(23)


def operations():
    ops = []
    for func in (1, 2, 3, 6, 7):
        for ftz in (0, 0x100):
            ops.append((0x11, func, ftz))
    for func in (0x10, 0x11):
        for mode in (0, 1, 2):
            ops.append((0x11, func, mode))
    ops += [(0x11, 0x14, 0), (0x11, 0x15, 0)]
    for cc in (0, 1, 2, 3, 4, 5, 10, 11):
        ops.append((0x11, 0x18, cc))
    ops += [(0x12, 0, 0), (0x12, 1, 0)]
    for func in (2, 3):
        for rnd in range(4):
            ops.append((0x12, func, rnd))
    for src in range(5):
        for dst in range(5):
            ops.append((0x12, 4, src | (dst << 3)))
    for func in (8, 9, 10):
        for broadcast in (0, 1):
            ops.append((0x15, func, broadcast))
    for func in range(4):
        ops.append((0x16, func, 0))
    return ops


def stimulus(path, per_op, seed):
    rng = random.Random(seed)
    lines = 0
    with path.open('x') as f:
        for major, func, mod in operations():
            for k in range(per_op):
                mask = rng.choice([0xffff, 0xffff, rng.getrandbits(16), 0x0001, 0x8000, 0])
                if major == 0x15 and k % 3 == 0:   # reductions over specials
                    a = [rng.choice(SPECIALS) for _ in range(16)]
                else:
                    a = [rand_float(rng) for _ in range(16)]
                if major == 0x12 and func in (0, 1):
                    a = [rng.choice([rng.getrandbits(32), rng.randint(-(1 << 25), 1 << 25) & 0xffffffff]) for _ in range(16)]
                b = [rand_float(rng) for _ in range(16)]
                c = [rand_float(rng) for _ in range(16)]
                if major == 0x11 and k % 4 == 0:   # equal / signed-zero pairs for min, max, cmp
                    b = [x if rng.random() < 0.5 else x ^ (0x80000000 if (x & 0x7fffffff) == 0 else 0) for x in a]
                f.write(f'{major:02x} {func:02x} {mod:03x} {mask:04x} ' + ' '.join(f'{x:08x}' for x in a + b + c) + '\n')
                lines += 1
    return lines


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-op', type=int, default=120)
    ap.add_argument('--seed', type=int, default=4051)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rtl = a.root / 'rtl'
    stim = a.out / 'stimulus.txt'
    n = stimulus(stim, a.per_op, a.seed)
    vvp = a.out / 'tb.vvp'
    iverilog = shutil.which('iverilog')
    cmd = [iverilog, '-g2012', '-I', str(rtl/'common'), '-y', str(rtl/'core'), '-y', str(rtl/'gen'),
           '-o', str(vvp), str(HERE/'tb_vector_fpu_oracle.v')]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    t = time.monotonic()
    subprocess.run(['vvp', '-n', str(vvp), f'+in={stim}', f'+out={a.out/"results.txt"}'], check=True,
                   stdout=(a.out/'vvp.log').open('w'), stderr=subprocess.STDOUT, timeout=7200)
    elapsed = time.monotonic() - t
    stim_lines = stim.read_text().splitlines()
    res_lines = (a.out/'results.txt').read_text().splitlines()
    if len(stim_lines) != n or len(res_lines) != n:
        raise SystemExit(f'RTL produced {len(res_lines)} results for {n} stimuli')
    with (a.out/'oracle.txt').open('x') as f:
        for s, r in zip(stim_lines, res_lines):
            f.write(s + ' | ' + r + '\n')
    used = sorted({str(p.relative_to(a.root)) for p in [rtl/'core/tnca_vector_fpu.v', rtl/'core/tnca_fp32_fma.v',
                   rtl/'core/tnca_sfu.v', rtl/'common/tnca_defs.vh'] + list((rtl/'gen').glob('*.v'))})
    report = dict(schema='tnca-rtl-oracle/1', unit='tnca_vector_fpu', vectors=n, per_op=a.per_op, seed=a.seed,
                  operations=len(operations()), simulator=subprocess.run([iverilog, '-V'], capture_output=True,
                  text=True).stdout.splitlines()[0], simulation_seconds=round(elapsed, 2),
                  rtl_sha256={p: sha(a.root/p) for p in used}, testbench_sha256=sha(HERE/'tb_vector_fpu_oracle.v'),
                  oracle_sha256=sha(a.out/'oracle.txt'),
                  scope='offline oracle capture; the model never runs RTL')
    (a.out/'oracle.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('vectors', 'operations', 'simulation_seconds')}))


if __name__ == '__main__':
    main()
