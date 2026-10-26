#!/usr/bin/env python3
"""Independent exact-oracle validation of native numeric helpers on the GB10."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from run import ROOT, sha

sys.path.insert(0, str(ROOT/'tools/model'))
from fp32 import classify, to_fraction


def rtl_float_to_integer(raw, signed, mode):
    """rtl/core/tnca_vector_fpu.v binary32 -> int32/uint32, transcribed in
    Python from the Verilog (8-bit `csh`, 32-bit `csh - 1`, 64-bit shifts).
    Independent of the C++ port; tools/performance_harness/systemc/rtl_oracle
    checks both against simulated RTL."""
    cexp, cneg, frac = (raw >> 23) & 255, raw >> 31, raw & 0x7fffff
    cman = (1 << 23) | frac
    if cexp == 255 and frac:
        return 0
    if cexp < 127:
        if mode == 1:
            return (0xffffffff if cneg else 1) if (cexp == 126 and frac) else 0
        if mode == 2:
            return 0xffffffff if (cneg and raw & 0x7fffffff) else 0
        if mode == 3:
            return 1 if (not cneg and raw & 0x7fffffff) else 0
        return 0
    if cexp >= 158:
        return (0x80000000 if cneg else 0x7fffffff) if signed else (0 if cneg else 0xffffffff)
    csh = (150 - cexp) & 255
    cmag = cman >> csh if csh < 64 else 0
    cfrac = cman & ((((1 << csh) if csh < 64 else 0) - 1) & ((1 << 64) - 1))
    half_shift = (csh - 1) & 0xffffffff
    chalf = (1 << half_shift) if half_shift < 64 else 0
    inc = {1: cfrac > chalf or (cfrac == chalf and cmag & 1), 2: bool(cneg and cfrac), 3: bool(not cneg and cfrac)}.get(mode, False)
    cmag += int(inc)
    fix = -cmag if cneg else cmag
    if signed:
        return 0x7fffffff if fix > 0x7fffffff else 0x80000000 if fix < -(1 << 31) else fix & 0xffffffff
    return 0 if cneg else min(cmag, 0xffffffff)


def validate(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(7319)
    command = [sys.executable, str(ROOT/'tools/model/gen_cvt_vectors.py'),
               '-o', str(out/'formats.txt'), '--f32-random', '20000', '--seed', '7319']
    with (out/'generator.log').open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
    command=[sys.executable,str(ROOT/'tools/model/gen_sfu_vectors.py'),
             '-o',str(out/'sfu.txt'),'--count','30000','--seed','7319']
    with (out/'sfu-generator.log').open('x') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    edges = [0, 1, 0xffffffff, 0x100000000, (1 << 63)-1, 1 << 63, (1 << 64)-1]
    pairs = [(a, b) for a in edges for b in edges]
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(20000))
    with (out/'mul.txt').open('x') as f:
        for a, b in pairs:
            sa = a if a < 1 << 63 else a-(1 << 64)
            sb = b if b < 1 << 63 else b-(1 << 64)
            f.write(f'{a:016x} {b:016x} {(a*b)>>64:016x} {((sa*sb)>>64)&((1<<64)-1):016x}\n')
    floats = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
              0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x40200000,
              0x4f000000, 0xcf000000, 0x4f800000, 0xcf800000]
    floats += [rng.getrandbits(32) for _ in range(20000)]
    floats += [(e << 23) | rng.getrandbits(23) | (rng.getrandbits(1) << 31) for e in range(150, 158) for _ in range(200)]
    divergent = []
    with (out/'ints.txt').open('x') as f:
        for raw in floats:
            kind, sign, _ = classify(raw)
            value = to_fraction(raw)
            for signed in (0, 1):
                low, high = (-(1 << 31), (1 << 31)-1) if signed else (0, (1 << 32)-1)
                for mode in range(4):
                    if kind == 'nan': want = 0
                    elif kind == 'inf': want = low if sign else high
                    else:
                        numerator, denominator = value.numerator, value.denominator
                        q, remainder = divmod(numerator, denominator)
                        if mode == 0: want = q+int(numerator < 0 and remainder != 0)
                        elif mode == 2: want = q
                        elif mode == 3: want = q+int(remainder != 0)
                        else: want = q+int(2*remainder > denominator or (2*remainder == denominator and q % 2))
                        want = max(low, min(high, want))
                    # The RTL is the authority for what a device computes; the
                    # ISA-text value is kept to record every divergence.
                    rtl = rtl_float_to_integer(raw, signed, mode)
                    if rtl != want & 0xffffffff:
                        divergent.append((signed, mode, raw, want & 0xffffffff, rtl))
                    f.write(f'{signed} {mode} {raw:08x} {rtl:08x}\n')
    results={}
    for mode in ('formats', 'mul', 'ints', 'sfu'):
        command = [str(binary), mode, str(out/f'{mode}.txt')]
        with (out/f'{mode}.stdout').open('x') as stdout, (out/f'{mode}.stderr').open('x') as stderr:
            r = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=120)
        results[mode] = dict(returncode=r.returncode,
            result=json.loads((out/f'{mode}.stdout').read_text()),
            vectors_sha256=sha(out/f'{mode}.txt'))
    report = dict(schema='tnca-native-numeric-validation/1',
        status='passed' if all(r['returncode']==0 for r in results.values()) else 'failed',
        results=results, binary_sha256=sha(binary),
        oracle_sha256={p.name:sha(p) for p in [ROOT/'tools/model/fp32.py',
            ROOT/'tools/model/floatfmt.py', ROOT/'tools/model/gen_cvt_vectors.py', ROOT/'tools/model/gen_sfu_vectors.py']},
        nan_comparison='class, matching the existing RTL conversion validator',
        float_to_integer_rtl_vs_isa=dict(
            vectors=len(floats) * 8, divergent=len(divergent),
            note='rtl/core/tnca_vector_fpu.v differs from ISA sec 8.4 in four places: (1) |x| in [2^24, 2^31) '
                 'converts to 0/+-1 in every mode (8-bit `csh` wraps), (2) odd |x| in [2^23, 2^24) round up under '
                 'rne (`chalf` is 0 when csh = 0), (3) unsigned x in [2^31, 2^32) saturates to 0xffffffff, '
                 '(4) unsigned negative x in (-1, 0) gives 0xffffffff under rne (x < -0.5) and rmi. The model '
                 'follows the RTL; none of these conversions occurs in the frozen suites (opcode inventory)',
            examples=[dict(signed=a, mode=b, input=f'{c:08x}', isa=f'{d:08x}', rtl=f'{e:08x}') for a, b, c, d, e in divergent[:12]]),
        scope='numeric semantics only; no timing or complete-chip qualification', performance_gate='not-measured')
    (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    result=validate(a.binary,a.out);print(result['status'])
    raise SystemExit(0 if result['status']=='passed' else 1)
