#!/usr/bin/env python3
"""kfuse_rtl_vectors.py -- test vectors for rtl/core/tnca_kfuse_lane.v from tools/r42/kfuse_model.py.

Every expected value is `kfuse_model.scalar_step` (the definition: Fractions, the independent oracles of
tools/model) for the lane's configuration: variant 'sum_fma', rounding 'trunc', element format E4M3 x E4M3,
group size G, window W. Nothing in this file computes a result by itself.

Output: one hex word per vector, {y[31:0], c[31:0], b[8G-1:0], a[8G-1:0]} (element k of a is bits [8k +: 8]),
as `$readmemh` reads it. The vector set is a fixed directed prefix followed by N random vectors drawn from
weighted classes (see CLASSES); the whole file is a pure function of (G, W, N, seed).

  python3 kfuse_rtl_vectors.py --G 8 --W 0 --N 1000000 --seed 1 --out vec_G8_W0.hex [--procs 4]
"""
import argparse
import hashlib
import math
import multiprocessing as mp
import os
import random
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
import kfuse_model as km          # noqa: E402
import fp32 as f32                # noqa: E402
import fdp                        # noqa: E402

NAN_CODES = (0x7F, 0xFF)
QNAN = 0x7FC00000


def e4m3_value(byte):
    k, s, v = fdp.element('e4m3', byte)
    return k, s, v


# non-negative finite magnitudes, ascending, for quantisation
_POS = sorted(((float(e4m3_value(b)[2]), b) for b in range(0x7F) if e4m3_value(b)[0] in ('zero', 'fin')))
_POS_V = [v for v, _ in _POS]


def quant(x):
    """nearest E4M3 byte to float x (saturating), for generating 'typical' operands"""
    s = 0x80 if x < 0 else 0
    ax = abs(x)
    import bisect
    i = bisect.bisect_left(_POS_V, ax)
    if i == 0:
        b = _POS[0][1]
    elif i >= len(_POS):
        b = _POS[-1][1]
    else:
        lo, hi = _POS[i - 1], _POS[i]
        b = lo[1] if (ax - lo[0]) <= (hi[0] - ax) else hi[1]
    return s | b


def nonan(rng, b):
    return b if b not in NAN_CODES else (b ^ 0x01)


def rbyte(rng):
    return rng.getrandbits(8)


def mk(rng, e, f=None, s=None):
    f = rng.getrandbits(3) if f is None else f
    s = rng.getrandbits(1) if s is None else s
    return nonan(rng, (s << 7) | (e << 3) | f)


def fp32_from_fraction(v):
    return f32.round_to_f32(v, 0)


def products(a, b):
    out = []
    for x, y in zip(a, b):
        kx, sx, vx = e4m3_value(x)
        ky, sy, vy = e4m3_value(y)
        if 'nan' in (kx, ky):
            return None
        out.append(vx * vy)
    return out


def group_ref(cfg, a, b, c):
    return km.scalar_step(cfg, 'e4m3', a, b, c)


def window_sum(cfg, a, b):
    """S * 2^L as a Fraction exactly as the model forms it (None when nothing non-zero / NaN present)."""
    p = products(a, b)
    if p is None:
        return None
    prods = [v for v in p if v != 0]
    if not prods:
        return None
    S, L = km._aligned_sum(prods, cfg.W, cfg.rnd)
    return Fraction(S) * km._pow2(L)


C_SPECIAL = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF,
             0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
             0x3F800000, 0xBF800000, 0x4B800000, 0x33800000]


def gen_c(rng, cfg, a, b, mode=None):
    """an accumulator for the group (a, b): the interesting alignments are c's exponent within +-40 of the group's"""
    ws = window_sum(cfg, a, b)
    mode = mode or rng.choices(['near', 'uniform', 'special', 'cancel', 'zero', 'tiny', 'huge'],
                               weights=[40, 15, 12, 12, 8, 6, 7])[0]
    if mode == 'uniform':
        return rng.getrandbits(32)
    if mode == 'special':
        return rng.choice(C_SPECIAL)
    if mode == 'zero':
        return rng.choice((0, 0x80000000))
    if mode == 'tiny':                       # subnormal / lowest exponents
        return (rng.getrandbits(1) << 31) | (rng.randrange(0, 4) << 23) | rng.getrandbits(23)
    if mode == 'huge':
        return (rng.getrandbits(1) << 31) | (rng.randrange(250, 255) << 23) | rng.getrandbits(23)
    if mode == 'cancel' and ws is not None:
        # c = -(S 2^L), rounded to binary32 (exact when S fits 24 bits); sometimes perturbed by one ulp
        c = fp32_from_fraction(-ws)
        if rng.random() < 0.3:
            c ^= 1 << rng.randrange(0, 3)
        return c
    # near: exponent within +-40 of the group's top
    if ws is None or ws == 0:
        top = rng.randrange(-20, 18)
    else:
        top = math.frexp(float(abs(ws)))[1] - 1
    e = max(-149, min(127, top + rng.randrange(-40, 41)))
    m = (1 << 23) | rng.getrandbits(23)
    if e >= -126:
        bits = ((e + 127) << 23) | (m & 0x7FFFFF)
    else:
        bits = m >> (-126 - e) if (-126 - e) < 24 else 1
    return (rng.getrandbits(1) << 31) | (bits & 0x7FFFFFFF)


# ---------------------------------------------------------------------------------------------
# operand classes
# ---------------------------------------------------------------------------------------------

def cls_uniform_nonan(rng, G):
    return [nonan(rng, rbyte(rng)) for _ in range(G)], [nonan(rng, rbyte(rng)) for _ in range(G)]


def cls_uniform(rng, G):
    return [rbyte(rng) for _ in range(G)], [rbyte(rng) for _ in range(G)]


def cls_normal(rng, G):
    sa = 10 ** rng.uniform(-2, 1.5)
    sb = 10 ** rng.uniform(-2, 1.5)
    return ([quant(rng.gauss(0, 1) * sa) for _ in range(G)],
            [quant(rng.gauss(0, 1) * sb) for _ in range(G)])


def split_exp(rng, X):
    """effective exponents ea + eb = X, each in 1..15 (a zero exponent field is effective exponent 1)"""
    lo = max(1, X - 15)
    hi = min(15, X - 1)
    ea = rng.randint(lo, hi)
    return ea, X - ea


def elem_with_eff(rng, eff, subn_ok=True):
    """a byte whose effective exponent is eff (1 may be a subnormal)"""
    if eff == 1 and subn_ok and rng.random() < 0.5:
        return mk(rng, 0)
    return mk(rng, eff)


def cls_spread(rng, G):
    """product exponents X_i spread: one near the top, the rest anywhere (truncation in the window)"""
    style = rng.choice(['top_low', 'random', 'close', 'two_levels'])
    base = rng.randint(2, 30)
    a, b = [], []
    for i in range(G):
        if style in ('top_low', 'random'):
            X = rng.randint(2, 30)
        elif style == 'close':
            X = max(2, min(30, base + rng.randint(-4, 4)))
        else:
            X = rng.choice([base, max(2, base - rng.randint(8, 28))])
        ea, eb = split_exp(rng, X)
        a.append(elem_with_eff(rng, ea))
        b.append(elem_with_eff(rng, eb))
    if style == 'top_low':
        k = rng.randrange(G)
        ea, eb = split_exp(rng, rng.randint(26, 30))
        a[k] = mk(rng, ea)
        b[k] = mk(rng, eb)
    return a, b


def cls_equal_exp(rng, G):
    ea, eb = rng.randint(1, 15), rng.randint(1, 15)
    return [mk(rng, ea) for _ in range(G)], [mk(rng, eb) for _ in range(G)]


def cls_cancel(rng, G):
    """pairs of equal-magnitude products of opposite sign; the rest small or zero"""
    a = [0] * G
    b = [0] * G
    idx = list(range(G))
    rng.shuffle(idx)
    npair = rng.randint(1, G // 2)
    for p in range(npair):
        i, j = idx[2 * p], idx[2 * p + 1]
        x, y = nonan(rng, rbyte(rng)), nonan(rng, rbyte(rng))
        a[i], b[i] = x, y
        a[j], b[j] = x ^ 0x80, y                  # same magnitude, opposite sign
    for q in idx[2 * npair:]:
        if rng.random() < 0.5:
            a[q], b[q] = nonan(rng, rbyte(rng)), nonan(rng, rbyte(rng))
        else:
            a[q], b[q] = rng.choice((0, 0x80)), nonan(rng, rbyte(rng))
    return a, b


def cls_subnormal(rng, G):
    def el():
        r = rng.random()
        if r < 0.55:
            return mk(rng, 0)                      # subnormal or zero
        if r < 0.8:
            return mk(rng, rng.randint(1, 3))
        return nonan(rng, rbyte(rng))
    return [el() for _ in range(G)], [el() for _ in range(G)]


def cls_zeros(rng, G):
    a, b = cls_uniform_nonan(rng, G)
    for i in range(G):
        r = rng.random()
        if r < 0.45:
            a[i] = rng.choice((0x00, 0x80))
        elif r < 0.6:
            b[i] = rng.choice((0x00, 0x80))
    if rng.random() < 0.35:                         # all products zero, signs random
        for i in range(G):
            a[i] = rng.choice((0x00, 0x80)) if rng.random() < 0.7 else a[i]
            b[i] = nonan(rng, rbyte(rng)) if a[i] in (0, 0x80) else rng.choice((0x00, 0x80))
    return a, b


def cls_negzero(rng, G):
    """all products are zeros, mostly all negative"""
    a, b = [], []
    for i in range(G):
        s = 1 if rng.random() < 0.85 else rng.getrandbits(1)
        if rng.random() < 0.5:
            a.append(0x00 | (rng.getrandbits(1) << 7))
            b.append(nonan(rng, rbyte(rng)))
        else:
            a.append(nonan(rng, rbyte(rng)))
            b.append(0x00 | (rng.getrandbits(1) << 7))
        # force the product sign
        sa, sb = a[-1] >> 7, b[-1] >> 7
        if (sa ^ sb) != s:
            a[-1] ^= 0x80
    return a, b


def cls_nan(rng, G):
    a, b = cls_uniform_nonan(rng, G)
    for _ in range(rng.randint(1, 2)):
        if rng.random() < 0.5:
            a[rng.randrange(G)] = rng.choice(NAN_CODES)
        else:
            b[rng.randrange(G)] = rng.choice(NAN_CODES)
    return a, b


def cls_trunc_sens(rng, G):
    """one product at the top exponent, the others a few bits below the window's last bit, full mantissas"""
    a, b = [], []
    k = rng.randrange(G)
    top = rng.randint(24, 30)
    for i in range(G):
        if i == k:
            X = top
        else:
            X = max(2, top - rng.randint(14, 26))
        ea, eb = split_exp(rng, X)
        a.append(mk(rng, ea, f=rng.choice((7, 5, 3, rng.getrandbits(3)))))
        b.append(mk(rng, eb, f=rng.choice((7, 5, 3, rng.getrandbits(3)))))
    return a, b


CLASSES = [
    # name, weight, operand generator, c-mode override
    ('uniform_nonan', 14, cls_uniform_nonan, None),
    ('uniform',        3, cls_uniform, None),
    ('normal',        14, cls_normal, None),
    ('spread',        14, cls_spread, None),
    ('equal_exp',      6, cls_equal_exp, None),
    ('cancel',         8, cls_cancel, None),
    ('subnormal',      8, cls_subnormal, None),
    ('zeros',          7, cls_zeros, None),
    ('negzero',        4, cls_negzero, 'zero'),
    ('nan',            3, cls_nan, None),
    ('trunc_sens',    13, cls_trunc_sens, None),
    ('ctrunc',         6, cls_spread, 'huge'),
]


def pack(vals):
    r = 0
    for i, v in enumerate(vals):
        r |= (v & 0xFF) << (8 * i)
    return r


def vec(cfg, a, b, c):
    y = group_ref(cfg, a, b, c)
    return (a, b, c, y)


def line(G, v):
    a, b, c, y = v
    w = (y << (32 + 16 * G)) | (c << (16 * G)) | (pack(b) << (8 * G)) | pack(a)
    return '%0*x' % ((16 * G + 64) // 4, w)


# ---------------------------------------------------------------------------------------------
# directed set
# ---------------------------------------------------------------------------------------------

def directed(cfg, rng):
    G = cfg.G
    out = []
    Z = [0] * G
    # D1: every (a, b) byte pair alone in element 0 (and in the last element), c in a few values
    cs = [0, 0x3F800000, 0xBF800000, 0x80000000, 0x00000001]
    for x in range(256):
        for y in range(256):
            pos = 0 if (x + y) % 2 == 0 else G - 1
            a, b = list(Z), list(Z)
            a[pos], b[pos] = x, y
            out.append((a, b, cs[(x * 7 + y) % len(cs)]))
    # D2: all exponents equal (every exponent field x every fraction pair at random positions), random c near
    for e in range(16):
        for _ in range(40):
            a = [mk(rng, e) for _ in range(G)]
            b = [mk(rng, rng.randint(1, 15)) for _ in range(G)]
            out.append((a, b, None))
    # D3: maximum and minimum exponent spread
    for big in (0x7E, 0xFE, 0x7D, 0x77):
        for small in (0x01, 0x81, 0x02, 0x08, 0x00):
            a = [small] * G
            b = [0x01] * G
            a[0], b[0] = big, 0x7E
            out.append((list(a), list(b), None))
            a2 = [big] * G
            b2 = [0x7E] * G
            a2[G - 1], b2[G - 1] = small, 0x01
            out.append((a2, b2, None))
    # D4: full cancellation: p and -p, with and without c = 0, c = exact cancel
    for _ in range(400):
        a, b = cls_cancel(rng, G)
        out.append((a, b, 'cancel'))
        out.append((a, b, 0))
    # D5: NaN at every position of a and of b, c finite / NaN / inf
    for i in range(G):
        for nan in NAN_CODES:
            for side in (0, 1):
                for c in (0, 0x3F800000, QNAN, 0x7F800000, 0xFF800000):
                    a, b = cls_uniform_nonan(rng, G)
                    (a if side == 0 else b)[i] = nan
                    out.append((a, b, c))
    # NaN times zero
    a, b = list(Z), list(Z)
    a[0], b[0] = 0x7F, 0x00
    out.append((a, b, 0))
    a[0], b[0] = 0x00, 0xFF
    out.append((list(a), list(b), 0x80000000))
    # D6: zero groups (all sign patterns of the first few products) against every special c
    for c in C_SPECIAL:
        for sgn in range(1 << min(G, 4)):
            a = [(0x80 if (sgn >> (i % 4)) & 1 else 0x00) for i in range(G)]
            b = [0x01] * G
            out.append((a, b, c))
            out.append(([0x80] * G, [0x03] * G, c))
            out.append(([0x00] * G, [0x83] * G, c))
            out.append(([0x80] * G, [0x83] * G, c))
        for _ in range(8):
            a, b = cls_uniform_nonan(rng, G)
            out.append((a, b, c))
    # D7: c huge and tiny against random groups
    for c in (0x7F7FFFFF, 0xFF7FFFFF, 0x7F000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x1A000000):
        for _ in range(20):
            a, b = cls_uniform_nonan(rng, G)
            out.append((a, b, c))
    # D8: max-magnitude products, both signs, c = +-max
    for s in range(4):
        a = [0x7E if (s & 1) == 0 else 0xFE] * G
        b = [0x7E if (s & 2) == 0 else 0xFE] * G
        for c in (0, 0x7F7FFFFF, 0x4A800000, 0xCA800000):
            out.append((list(a), list(b), c))
    return out


def build_one(args):
    G, W, seed, n, directed_set = args
    cfg = km.Cfg(G=G, W=W, rnd='trunc', variant='sum_fma')
    rng = random.Random(seed)
    names = [c[0] for c in CLASSES]
    weights = [c[1] for c in CLASSES]
    lines = []
    stat = {k: 0 for k in names}
    if directed_set:
        for a, b, c in directed(cfg, rng):
            if c is None:
                c = gen_c(rng, cfg, a, b, 'near')
            elif c == 'cancel':
                c = gen_c(rng, cfg, a, b, 'cancel')
            lines.append(line(G, vec(cfg, a, b, c)))
        stat['directed'] = len(lines)
    for _ in range(n):
        i = rng.choices(range(len(CLASSES)), weights=weights)[0]
        name, _, gen, cmode = CLASSES[i]
        a, b = gen(rng, G)
        c = gen_c(rng, cfg, a, b, cmode)
        lines.append(line(G, vec(cfg, a, b, c)))
        stat[name] += 1
    return lines, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--G', type=int, required=True)
    ap.add_argument('--W', type=int, default=0)
    ap.add_argument('--N', type=int, default=1000000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--procs', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-directed', action='store_true')
    ap.add_argument('--chunk', type=int, default=50000)
    a = ap.parse_args()
    jobs = []
    rem = a.N
    k = 0
    if not a.no_directed:
        jobs.append((a.G, a.W, a.seed * 1000003 + 7, 0, True))
    while rem > 0:
        n = min(a.chunk, rem)
        jobs.append((a.G, a.W, a.seed * 1000003 + 101 + k, n, False))
        rem -= n
        k += 1
    with mp.Pool(a.procs) as pool:
        res = pool.map(build_one, jobs)
    tot = {}
    nl = 0
    h = hashlib.sha256()
    with open(a.out, 'w') as f:
        for lines, stat in res:
            for l in lines:
                f.write(l + '\n')
                h.update((l + '\n').encode())
            nl += len(lines)
            for k2, v in stat.items():
                tot[k2] = tot.get(k2, 0) + v
    print('G=%d W=%d vectors=%d sha256=%s' % (a.G, a.W, nl, h.hexdigest()))
    print('classes:', ' '.join('%s=%d' % kv for kv in sorted(tot.items())))


if __name__ == '__main__':
    main()
