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
                    f.write(f'{signed} {mode} {raw:08x} {want&0xffffffff:08x}\n')
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
