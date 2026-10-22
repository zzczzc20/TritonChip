#!/usr/bin/env python3
"""Accuracy of the compiler's log1p, sin and cos over every binary32 input.

    transcendental_numeric.py --out report.json [--functions log1p sin cos]
        [--sample N] [--jobs 4]

tnca_cc.mathlib builds these functions from builder calls (vbin, vfma,
vcvt, vcmp, vselect, ...) and the compiler lowers each call to one ISA
instruction. `Emu` below is a builder whose calls EXECUTE that instruction
on numpy arrays instead of emitting it, so running mathlib.sin(Emu, ...)
evaluates the very sequence the compiler emits, constant for constant.
Every operation the functions use is exactly specified (no SFU): IEEE
binary32 add/sub/mul with round to nearest even and subnormals, a single-
rounding fma (emulated exactly: float64 product and TwoSum, then the
binary32 rounding with the sum's sign as sticky at a midpoint), and 32-bit
integer arithmetic. The device run of the same kernels
(tools/performance_harness/std_ops_cases.py) is compared bit for bit with
this emulation on its sample, which is what ties the two together.

The reference is float64 numpy (log1p, sin, cos of the exactly widened
input; glibc's float64 sin/cos reduce huge arguments exactly). The error of
a result y against the reference v is |y - v| in units of the binary32 ulp
at |v| (2**-149 at and below the subnormal range); "correctly rounded"
means y equals float32(v). NaN results must match NaN references.

Without --sample every one of the 2**32 bit patterns is evaluated.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler/python'))

U32, I32, F32, F64 = np.uint32, np.int32, np.float32, np.float64


class V:
    """An emulated vector register: 32-bit lane patterns and a dtype tag."""
    __slots__ = ('bits', 'dtype')

    def __init__(self, bits, dtype):
        self.bits = bits
        self.dtype = dtype

    @property
    def f(self):
        return self.bits.view(F32)

    @property
    def s(self):
        return self.bits.view(I32)


def _fv(arr):
    return V(np.ascontiguousarray(arr, dtype=F32).view(U32), 'f32')


def _iv(arr):
    return V(np.ascontiguousarray(arr).astype(U32, copy=False), 'i32')


def fma32(a, b, c):
    """binary32 a*b + c with ONE rounding (RNE), exactly."""
    a64, b64, c64 = a.astype(F64), b.astype(F64), c.astype(F64)
    p = a64 * b64                                   # exact: 24 + 24 bits
    with np.errstate(invalid='ignore', over='ignore'):
        s = p + c64
        bb = s - p
        e = (p - (s - bb)) + (c64 - bb)             # exact error of the sum
        r1 = s.astype(F32)
        toward = np.where(s > r1.astype(F64), np.inf, -np.inf).astype(F32)
        n = np.nextafter(r1, toward)
        mid = (r1.astype(F64) + n.astype(F64)) * 0.5
        is_mid = (s == mid) & (n != r1) & np.isfinite(s) & (e != 0)
        lo = np.minimum(r1, n)
        hi = np.maximum(r1, n)
        out = np.where(is_mid & (e > 0), hi, np.where(is_mid & (e < 0), lo, r1))
    return out.astype(F32)


class Emu:
    """The tnca_cc.ir.Builder calls mathlib and wide use, executed."""

    def __init__(self, n):
        self.n = n

    def vmovi(self, imm, dtype='i32'):
        return V(np.full(self.n, int(imm) & 0xFFFFFFFF, dtype=U32), dtype)

    def vbin(self, op, a, b, dtype=None, comment=None, fmm=None):
        dtype = dtype or a.dtype
        if dtype == 'f32':
            x, y = a.f, b.f
            with np.errstate(all='ignore'):
                if op == 'add':
                    r = x + y
                elif op == 'sub':
                    r = x - y
                elif op == 'mul':
                    r = x * y
                else:
                    raise NotImplementedError(f'f32 {op}')
            return _fv(r)
        x, y = a.bits, b.bits
        sh = (y & U32(31))
        if op == 'add':
            r = x + y
        elif op == 'sub':
            r = x - y
        elif op == 'mul':
            r = x * y
        elif op == 'mulhu':
            r = ((x.astype(np.uint64) * y.astype(np.uint64)) >> np.uint64(32)).astype(U32)
        elif op == 'and':
            r = x & y
        elif op == 'or':
            r = x | y
        elif op == 'xor':
            r = x ^ y
        elif op == 'shl':
            r = x << sh
        elif op == 'lshr':
            r = x >> sh
        elif op == 'ashr':
            r = (a.s >> sh.astype(I32)).view(U32)
        elif op == 'min':
            r = np.minimum(a.s, b.s).view(U32)
        elif op == 'max':
            r = np.maximum(a.s, b.s).view(U32)
        else:
            raise NotImplementedError(f'i32 {op}')
        return V(r.astype(U32, copy=False), 'i32')

    def vun(self, op, a, dtype=None, comment=None):
        dtype = dtype or a.dtype
        if op == 'neg':
            return V(a.bits ^ U32(0x80000000), dtype)
        if op == 'abs':
            return V(a.bits & U32(0x7FFFFFFF), dtype)
        if op == 'clz':
            _, e = np.frexp(a.bits.astype(F64))
            return V(np.where(a.bits == 0, 32, 32 - e).astype(U32), 'i32')
        raise NotImplementedError(op)

    def vfma(self, a, b, c, dtype=None, comment=None):
        return _fv(fma32(a.f, b.f, c.f))

    def vfms(self, a, b, c, dtype=None, comment=None):
        return _fv(fma32(a.f, b.f, -c.f))

    def vcvt(self, src, dst, a, comment=None):
        if dst != 'f32':
            raise NotImplementedError(f'{src}->{dst}')
        if src == 'u32':
            return _fv(a.bits.astype(F32))
        if src == 's32':
            return _fv(a.s.astype(F32))
        raise NotImplementedError(src)

    def vcmp(self, cc, a, b):
        if a.dtype == 'f32':
            x, y = a.f, b.f
            with np.errstate(invalid='ignore'):
                r = {'eq': x == y, 'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y,
                     'o': ~(np.isnan(x) | np.isnan(y)), 'uo': np.isnan(x) | np.isnan(y)}[cc]
            return r
        x, y = a.s, b.s
        ux, uy = a.bits, b.bits
        return {'eq': x == y, 'ne': x != y, 'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y,
                'ltu': ux < uy, 'leu': ux <= uy, 'gtu': ux > uy, 'geu': ux >= uy}[cc]

    def vselect(self, p, a, b):
        return V(np.where(p, a.bits, b.bits), a.dtype if a.dtype == b.dtype else 'f32')


def evaluate(name, bits):
    """The compiled function `name` on binary32 patterns `bits` (uint32)."""
    from tnca_cc import mathlib
    b = Emu(len(bits))
    fconst = {}

    def k(c):
        key = np.float32(c).view(U32)
        if int(key) not in fconst:
            fconst[int(key)] = b.vmovi(int(key), dtype='f32')
        return fconst[int(key)]
    out = getattr(mathlib, name)(b, k, V(bits.astype(U32), 'f32'), dtype='f32')
    return out.bits


REFERENCE = {'log1p': np.log1p, 'sin': np.sin, 'cos': np.cos}


def ulp32(v):
    """The binary32 grid spacing at |v| (2**-149 at and below 2**-126)."""
    a = np.abs(v)
    _, ex = np.frexp(np.where(a > 0, a, 1.0))
    e = np.where(a > 0, np.maximum(ex - 1, -126), -126)
    return np.ldexp(1.0, (e - 23).astype(np.int64))


def check(name, bits):
    y = evaluate(name, bits).view(F32)
    x = bits.view(F32).astype(F64)
    with np.errstate(all='ignore'):
        ref = REFERENCE[name](x)
    yn, rn = np.isnan(y), np.isnan(ref)
    nan_mismatch = int(np.count_nonzero(yn != rn))
    both = ~(yn | rn)
    yv, rv = y[both].astype(F64), ref[both]
    inf_mismatch = int(np.count_nonzero(np.isinf(yv) != np.isinf(rv)) +
                       np.count_nonzero(np.isinf(yv) & np.isinf(rv) & (np.sign(yv) != np.sign(rv))))
    fin = np.isfinite(yv) & np.isfinite(rv)
    with np.errstate(all='ignore'):
        err = np.abs(yv[fin] - rv[fin]) / ulp32(rv[fin])
    cr = rv[fin].astype(F32)
    not_cr = int(np.count_nonzero(cr != yv[fin].astype(F32)))
    # signed zero: a zero result must carry the reference's sign
    zero = (yv[fin] == 0) & (rv[fin].astype(F32) == 0)
    sz = int(np.count_nonzero(zero & (np.signbit(yv[fin]) != np.signbit(rv[fin].astype(F32)))))
    worst = int(np.argmax(err)) if err.size else None
    return dict(n=int(bits.size), nan_mismatch=nan_mismatch, inf_mismatch=inf_mismatch,
                signed_zero_mismatch=sz, max_ulp=float(err.max()) if err.size else 0.0,
                over_half_ulp=int(np.count_nonzero(err > 0.5)), over_1_ulp=int(np.count_nonzero(err > 1.0)),
                not_correctly_rounded=not_cr,
                worst_input=(f'0x{int(bits[both][fin][worst]):08x}' if worst is not None else None))


def merge(a, b):
    out = dict(a)
    for key in ('n', 'nan_mismatch', 'inf_mismatch', 'signed_zero_mismatch', 'over_half_ulp',
                'over_1_ulp', 'not_correctly_rounded'):
        out[key] = a[key] + b[key]
    if b['max_ulp'] > a['max_ulp']:
        out['max_ulp'], out['worst_input'] = b['max_ulp'], b['worst_input']
    return out


def _chunk(args):
    name, start, count = args
    bits = np.arange(start, start + count, dtype=np.uint64).astype(U32)
    return check(name, bits)


def structured():
    """Edge values over every exponent plus random patterns."""
    rng = np.random.default_rng(20261022)
    vals = []
    for e in range(256):
        base = e << 23
        vals += [base, base | 1, base | 0x7FFFFF, base | 0x400000, base | 0x3FFFFF]
    vals += [0x3f800000, 0xbf800000, 0xbf7fffff, 0x7f800000, 0xff800000, 0x7fc00000]
    v = np.array(vals, dtype=np.uint64).astype(U32)
    v = np.concatenate([v, v | U32(0x80000000), rng.integers(0, 2**32, 1 << 20, dtype=np.uint64).astype(U32)])
    return np.unique(v)


def check_i64(signed, count=1 << 22, seed=7):
    """tnca_cc.wide.to_f32 on 64-bit integers: bit for bit against numpy's
    int64/uint64 -> float32 conversion (one rounding, RNE). Random patterns
    with every bit length, both signs, plus the edges around 2**24, 2**32,
    2**53, 2**63 and the round-to-even ties."""
    from tnca_cc import wide
    rng = np.random.default_rng(seed)
    length = rng.integers(0, 65, count)
    raw = rng.integers(0, 2**63, count, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, count, dtype=np.uint64)
    mask = np.where(length >= 64, np.uint64(0xFFFFFFFFFFFFFFFF),
                    (np.uint64(1) << length.astype(np.uint64)) - np.uint64(1))
    vals = raw & mask
    edges = []
    for p in (24, 25, 31, 32, 33, 53, 54, 62, 63):
        for d in (-3, -2, -1, 0, 1, 2, 3):
            edges.append((1 << p) + d)
        # ties at the 24-bit rounding point, odd and even
        edges += [(1 << p) + (1 << (p - 24)) * k for k in (1, 3)] if p > 24 else []
        edges += [(1 << p) + (1 << (p - 25)), (1 << p) + 3 * (1 << (p - 25))] if p > 25 else []
    edges += [0, 1, 2**64 - 1, 2**63, 2**63 - 1]
    e = np.array([x % (1 << 64) for x in edges], dtype=np.uint64)
    vals = np.concatenate([vals, e, (~e) + np.uint64(1)])
    lo = V((vals & np.uint64(0xFFFFFFFF)).astype(U32), 'i32')
    hi = V((vals >> np.uint64(32)).astype(U32), 'i32')
    got = wide.to_f32(Emu(len(vals)), lo, hi, signed=signed).bits
    ref = (vals.view(np.int64) if signed else vals).astype(F32).view(U32)
    bad = got != ref
    return dict(n=int(vals.size), mismatches=int(np.count_nonzero(bad)),
                examples=[dict(value=int((vals.view(np.int64) if signed else vals)[i]),
                               got=f'0x{int(got[i]):08x}', ref=f'0x{int(ref[i]):08x}')
                          for i in np.flatnonzero(bad)[:5]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--functions', nargs='+', default=['log1p', 'sin', 'cos'])
    ap.add_argument('--sample', action='store_true', help='structured + random sample instead of all 2**32')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=1 << 21)
    a = ap.parse_args()
    report = dict(schema='tnca-transcendental-numeric/1', mode='sample' if a.sample else 'exhaustive',
                  numpy=np.__version__, results={})
    from tnca_cc import mathlib
    report['accuracy_claims'] = {f: mathlib.ACCURACY.get(f) for f in a.functions if f in mathlib.ACCURACY}
    for name in a.functions:
        t = time.monotonic()
        if name in ('i64_to_f32', 'u64_to_f32'):
            res = check_i64(name == 'i64_to_f32')
            res['seconds'] = round(time.monotonic() - t, 1)
            report['results'][name] = res
            print(name, json.dumps(res), flush=True)
            a.out.write_text(json.dumps(report, indent=1))
            continue
        if a.sample:
            res = check(name, structured())
        else:
            jobs = [(name, s, a.chunk) for s in range(0, 1 << 32, a.chunk)]
            res = None
            with mp.get_context('fork').Pool(a.jobs) as pool:
                for part in pool.imap_unordered(_chunk, jobs):
                    res = part if res is None else merge(res, part)
        res['seconds'] = round(time.monotonic() - t, 1)
        report['results'][name] = res
        print(name, json.dumps(res), flush=True)
        a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
