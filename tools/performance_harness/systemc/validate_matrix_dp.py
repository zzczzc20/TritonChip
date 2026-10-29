#!/usr/bin/env python3
"""Exact-oracle validation of the SystemC model's m.mma.dp.async arithmetic.

The oracle is tools/model/fdp.py (Fractions, independent of the C++ in
include/tnca_matrix_dp.hpp). Vectors cover every format pair of ISA sec 9C.4,
random encodings, exponent-spread cases (where RO_48 differs from RNE), the
specials and the signed-zero rules. Vectors are fixed by the seed before the
model is run; nothing here is adjusted after looking at a result.

    validate_matrix_dp.py --binary BUILD/tnca-numeric-probe --out DIR [--count N]
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/model'))
import fdp  # noqa: E402

CODE = {'f16': 2, 'bf16': 1, 'e5m2': 3, 'e4m3': 4}
WIDTH = {'f16': 16, 'bf16': 16, 'e5m2': 8, 'e4m3': 8}
NAMES = list(CODE)


def group_of(fa, fb):
    return 4 if (fa == 'e4m3' and fb == 'e4m3') else 2


def oracle(fa, fb, a, b, c):
    g = fdp.group_sum(fa, fb, a, b)
    if g[0] == 'fin':
        g = ('fin', fdp.ro(g[1]), g[2])
    return fdp.add_round(g, c)


def element_pool(fmt, rng):
    """Special and boundary encodings plus random ones."""
    w = WIDTH[fmt]
    top = (1 << w) - 1
    pool = [0, 1 << (w - 1), 1, top, top >> 1, (top >> 1) + 1, (1 << (w - 1)) | 1]
    pool += [rng.getrandbits(w) for _ in range(40)]
    return pool


EXP_BITS = {'f16': (5, 10), 'bf16': (8, 7), 'e5m2': (5, 2), 'e4m3': (4, 3)}


def with_exponent(fmt, rng, lo, hi):
    """A finite encoding whose exponent field is in [lo, hi] (clipped to the format)."""
    eb, fb = EXP_BITS[fmt]
    top = (1 << eb) - 1
    e = rng.randint(max(lo, 0), min(hi, top - 1))
    return (rng.getrandbits(1) << (eb + fb)) | (e << fb) | rng.getrandbits(fb)


def f32_pool(rng):
    pool = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
            0x00800000, 0x3f800000, 0xbf800000]
    pool += [rng.getrandbits(32) for _ in range(40)]
    pool += [(rng.randrange(0, 255) << 23) | rng.getrandbits(23) for _ in range(20)]
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--count', type=int, default=4000, help='vectors per format pair')
    ap.add_argument('--seed', type=int, default=4129)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    lines = []
    for fa in NAMES:
        for fb in NAMES:
            g = group_of(fa, fb)
            ea, eb, cp = element_pool(fa, rng), element_pool(fb, rng), f32_pool(rng)
            for i in range(args.count):
                mode = i % 8
                if mode in (2, 3):  # moderate exponents: finite, varied results
                    bias = {'f16': 15, 'bf16': 127, 'e5m2': 15, 'e4m3': 7}
                    a = [with_exponent(fa, rng, bias[fa] - 4, bias[fa] + 4) for _ in range(g)]
                    b = [with_exponent(fb, rng, bias[fb] - 4, bias[fb] + 4) for _ in range(g)]
                elif mode in (5, 6):  # exponent spread inside one group: RO_48 and sticky bits
                    bias = {'f16': 15, 'bf16': 127, 'e5m2': 15, 'e4m3': 7}
                    a = [with_exponent(fa, rng, 0, 255) for _ in range(g)]
                    b = [with_exponent(fb, rng, 0, 255) for _ in range(g)]
                    a[0] = with_exponent(fa, rng, bias[fa] + 2, 255)
                    b[0] = with_exponent(fb, rng, bias[fb] + 2, 255)
                elif i % 4 == 0:      # drawn from the boundary pools
                    a = [rng.choice(ea) for _ in range(g)]
                    b = [rng.choice(eb) for _ in range(g)]
                else:               # fully random encodings
                    a = [rng.getrandbits(WIDTH[fa]) for _ in range(g)]
                    b = [rng.getrandbits(WIDTH[fb]) for _ in range(g)]
                c = rng.choice(cp) if i % 2 == 0 else rng.getrandbits(32)
                if mode in (2, 3, 5, 6) and rng.random() < 0.5:   # C of comparable magnitude, incl. subnormal range
                    c = (rng.getrandbits(1) << 31) | (rng.randint(0, 160) << 23) | rng.getrandbits(23)
                if i % 8 == 1:      # C of about the group's magnitude: cancellation and ties
                    s = oracle(fa, fb, a, b, 0)
                    c = s ^ 0x80000000 if rng.random() < 0.5 else s
                want = oracle(fa, fb, a, b, c)
                lines.append(f"{CODE[fa]} {CODE[fb]} {g} " + ' '.join(f'{x:x}' for x in a) + ' '
                             + ' '.join(f'{x:x}' for x in b) + f' {c:08x} {want:08x}')
    vec = args.out/'matrix_dp.txt'
    vec.write_text('\n'.join(lines) + '\n')
    with (args.out/'probe.stdout').open('x') as so, (args.out/'probe.stderr').open('x') as se:
        r = subprocess.run([str(args.binary), 'matrix_dp', str(vec)], stdout=so, stderr=se, timeout=1800)
    result = json.loads((args.out/'probe.stdout').read_text())
    report = dict(schema='tnca-matrix-dp-validation/1', domain='SystemC/TLM model arithmetic vs exact Fraction oracle',
                  status='passed' if r.returncode == 0 and result['mismatches'] == 0 else 'failed',
                  returncode=r.returncode, seed=args.seed, vectors_per_pair=args.count, result=result)
    (args.out/'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], result['checked'], 'vectors', result['mismatches'], 'mismatches')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
