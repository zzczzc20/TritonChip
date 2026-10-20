#!/usr/bin/env python3
"""Record tnca_matrix accumulator contents from the product RTL (offline).

    matrix_oracle.py --root PRODUCT_TREE --out FRESH_DIR [--cases N] [--seed S] [--abuf 0|1]

A command stream (vector-file loads, matrix commands, accumulator dumps) is
run once through iverilog; `tnca-numeric-probe matrix ORACLE` replays the same
stream through include/tnca_matrix_ops.hpp and compares every accumulator
word. Each command line also records the engine's busy cycles, retained for
later timing calibration (not fitted here).
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def element(rng, dtype):
    bits = 16 if dtype < 2 else 8
    r = rng.random()
    if r < 0.08:
        return rng.getrandbits(bits)            # anything, including inf/NaN
    if dtype == 0:   # f16 moderate exponents
        return (rng.getrandbits(1) << 15) | (rng.randint(8, 20) << 10) | rng.getrandbits(10)
    if dtype == 1:   # bf16
        return (rng.getrandbits(1) << 15) | (rng.randint(110, 140) << 7) | rng.getrandbits(7)
    if dtype == 2:   # e5m2
        return (rng.getrandbits(1) << 7) | (rng.randint(8, 22) << 2) | rng.getrandbits(2)
    return (rng.getrandbits(1) << 7) | (rng.randint(2, 13) << 3) | rng.getrandbits(3)   # e4m3


def pack(elements, bytes_):
    regs = []
    per = 64 // bytes_
    for r in range(0, len(elements), per):
        chunk = elements[r:r + per] + [0] * (per - len(elements[r:r + per]))
        lanes = []
        for lane in range(16):
            v = 0
            for j in range(4 // bytes_):
                v |= chunk[lane * (4 // bytes_) + j] << (8 * bytes_ * j)
            lanes.append(v)
        regs.append(lanes)
    return regs


def f32(rng):
    r = rng.random()
    if r < 0.1:
        return rng.choice([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 1, 0x80000001, 0x7f7fffff])
    return (rng.getrandbits(1) << 31) | (rng.randint(110, 140) << 23) | rng.getrandbits(23)


def stream(path, cases, seed):
    rng = random.Random(seed)
    lines = []
    def cmd(op, slot, va=0, vb=0, k=0, dt=0, rowop=0, row=0, scale=0, mask=0, vdata=None):
        vdata = vdata or [0] * 16
        lines.append('C ' + ' '.join(f'{x:x}' for x in (op, slot, va, vb, k, dt, rowop, row, scale, mask, 0) + tuple(vdata)))
    for case in range(cases):
        for dt in range(4):
            for k in range(3):
                K = 16 << k
                bytes_ = 2 if dt < 2 else 1
                slot = rng.randrange(16)
                a = pack([element(rng, dt) for _ in range(16 * K)], bytes_)
                b = pack([element(rng, dt) for _ in range(K * 16)], bytes_)
                va = rng.choice([0, 64, 100])
                vb = va + len(a) + rng.randrange(4)
                for i, regs in enumerate(a):
                    lines.append(f'V {va + i:x} ' + ' '.join(f'{x:x}' for x in regs))
                for i, regs in enumerate(b):
                    lines.append(f'V {vb + i:x} ' + ' '.join(f'{x:x}' for x in regs))
                if case % 2:
                    cmd(0, slot)                                    # m.zero
                else:
                    for row in range(16):                           # a.vwrite a random C
                        cmd(2, slot, rowop=0, row=row, mask=0xffff, vdata=[f32(rng) for _ in range(16)])
                cmd(1, slot, va=va, vb=vb, k=k, dt=dt)
                if rng.random() < 0.5:
                    row = rng.randrange(16)
                    cmd(2, slot, rowop=1, row=row, scale=f32(rng), mask=rng.getrandbits(16))          # a.mul.row
                    cmd(2, slot, rowop=2, row=rng.randrange(16), scale=f32(rng), mask=rng.choice([0xffff, rng.getrandbits(16)]),
                        vdata=[f32(rng) for _ in range(16)])                                        # a.fma.row
                lines.append(f'R {slot:x}')
    path.write_text('\n'.join(lines) + '\n')
    return len(lines)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1709)
    ap.add_argument('--abuf', type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rtl = a.root/'rtl'
    stim = a.out/'stream.txt'
    n = stream(stim, a.cases, a.seed)
    vvp = a.out/'tb.vvp'
    subprocess.run([shutil.which('iverilog'), '-g2012', f'-DORACLE_ABUF={a.abuf}', '-I', str(rtl/'common'),
                    '-y', str(rtl/'core'), '-y', str(rtl/'gen'), '-o', str(vvp), str(HERE/'tb_matrix_oracle.v')],
                   check=True, capture_output=True, text=True)
    subprocess.run(['vvp', '-n', str(vvp), f'+in={stim}', f'+out={a.out/"results.txt"}'], check=True,
                   stdout=(a.out/'vvp.log').open('w'), stderr=subprocess.STDOUT, timeout=7200)
    used = sorted(str(p.relative_to(a.root)) for p in [rtl/'core/tnca_matrix.v', rtl/'core/tnca_acc_rf.v',
                  rtl/'core/tnca_fpu_multi.v', rtl/'core/tnca_fp32_fma.v'] + list((rtl/'gen').glob('*.v')))
    report = dict(schema='tnca-rtl-oracle/1', unit='tnca_matrix', stream_lines=n, cases=a.cases, seed=a.seed,
                  abuf=a.abuf, rtl_sha256={p: sha(a.root/p) for p in used},
                  testbench_sha256=sha(HERE/'tb_matrix_oracle.v'), stream_sha256=sha(stim),
                  results_sha256=sha(a.out/'results.txt'), scope='offline oracle capture; the model never runs RTL')
    (a.out/'oracle.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(stream_lines=n)))


if __name__ == '__main__':
    main()
