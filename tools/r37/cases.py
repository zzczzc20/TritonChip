"""What each TileBench kernel is given, and what it should produce.

One entry per operator. The reference is written out in Python from the
kernel's own text rather than borrowed from a library, because the point
of the check is that the compiled kernel computes what the kernel SAYS
-- a reference that came from somewhere else would agree with the source
and disagree with the compiler for reasons neither of them caused.

`program_id` is 0 on this bench, which launches one context rather than
a grid. A kernel indexed by it therefore computes block 0 or row 0, and
`expect` covers exactly what it wrote.

Tolerances are the ISA's, not fitted. A kernel whose arithmetic is
add/multiply/min/max/select only is compared bit for bit; one that
divides or calls the SFU is compared to the bound mathlib.py states for
what it used (sec 8.8's 2**-22, plus a rounding or two), and the entry
says which operation put it there.
"""
import math
import random
import struct

VLEN = 16
T = 16          # one matrix tile

# Relative tolerances, by what the kernel does. ISA sec 8.8 specifies
# the SFU to 2**-22 relative; mathlib.py refines the reciprocal and the
# inverse square root and does not refine the exponential, so a divide
# lands near the format's own rounding and an exp lands near the SFU's.
EXACT = 0.0
TOL_DIV = 2 ** -21       # rcp + one Newton step, then a multiply
TOL_SQRT = 2 ** -21      # rsqrt + one Newton step, then a multiply
TOL_EXP = 2 ** -19       # sec 8.8's exp2 bound, through a change of base
TOL_CHAIN = 2 ** -17     # several of the above composed


def f16(x):
    """`x` rounded to binary16, as a Python float."""
    return struct.unpack('<e', struct.pack('<e', float(x)))[0]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def ramp(n, a=0.5, b=0.0):
    return [i * a + b for i in range(n)]


def wavy(n, scale=0.25, offset=-2.0, period=17):
    """Something that is not monotonic, so an argmax has work to do."""
    return [((i * 13) % period) * scale + offset for i in range(n)]


def small_ints(n, period=11, shift=5):
    return [float((i * 7) % period - shift) for i in range(n)]


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def build(Case, seed=None):
    """Every case, built against the harness's `Case`."""
    C = {}
    rng = random.Random(seed)

    def vary(values, kind='float'):
        if seed is None:
            return values
        if kind == 'permutation':
            result = list(values)
            rng.shuffle(result)
            return result
        if kind == 'positive':
            return [rng.randint(1, 16) * 0.25 for _ in values]
        if kind == 'int8':
            return [rng.randint(-128, 127) for _ in values]
        if kind == 'keys':
            return [rng.randrange(32) for _ in values]
        if kind == 'mask':
            return [rng.randrange(2) for _ in values]
        return [rng.randint(-16, 16) * 0.25 for _ in values]

    # ---- point-wise ------------------------------------------------------
    x = ramp(VLEN)
    x = vary(x)
    y = [i * 0.25 + 1.0 for i in range(VLEN)]
    y = vary(y)
    C['vector_add'] = Case(
        'vector_add',
        buffers={'x_ptr': ('fp32', x), 'y_ptr': ('fp32', y),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[a + b for a, b in zip(x, y)])

    C['mul2'] = Case(
        'mul2',
        buffers={'x_ptr': ('fp32', x), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=[v * 2.0 for v in x])

    signed = [v - 4.0 for v in x]
    C['relu'] = Case(
        'relu',
        buffers={'x_ptr': ('fp32', signed), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=[max(v, 0.0) for v in signed])

    C['leaky_relu'] = Case(
        'leaky_relu',
        buffers={'x_ptr': ('fp32', signed), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN, 'slope': 0.25}, out='o_ptr',
        expect=[v if v > 0 else v * 0.25 for v in signed])

    C['sigmoid'] = Case(
        'sigmoid',
        buffers={'x_ptr': ('fp32', signed), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[sigmoid(v) for v in signed], rtol=TOL_EXP,
        note='tl.exp, then a reciprocal')

    b = [1.0 + i * 0.125 for i in range(VLEN)]
    b = vary(b)
    C['swiglu'] = Case(
        'swiglu',
        buffers={'a_ptr': ('fp32', signed), 'b_ptr': ('fp32', b),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[a * sigmoid(a) * bb for a, bb in zip(signed, b)],
        rtol=TOL_EXP, note='tl.sigmoid')

    # A scale that is a power of two, so the division is exact and the
    # truncation to int8 cannot land on the wrong side of a boundary
    # because of the reciprocal's last bit.
    qx = [float(v) for v in range(-8, 8)]
    C['quantize_global'] = Case(
        'quantize_global',
        buffers={'x_ptr': ('fp32', qx), 'o_ptr': ('i8', [0] * VLEN)},
        scalars={'scale': 0.5, 'n': VLEN}, out='o_ptr',
        expect=[int(v / 0.5) for v in qx])

    bias = [0.25 * i for i in range(VLEN)]
    bias = vary(bias)

    def gelu_tanh(v):
        inner = 0.7978845608 * (v + 0.044715 * v * v * v)
        t = 1.0 - 2.0 / (math.exp(2.0 * inner) + 1.0)
        return 0.5 * v * (1.0 + t)

    fa_in = [v * 0.25 for v in signed]
    C['fused_activation'] = Case(
        'fused_activation',
        buffers={'x_ptr': ('fp32', fa_in), 'b_ptr': ('fp32', bias),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[gelu_tanh(a + bb) for a, bb in zip(fa_in, bias)],
        rtol=TOL_CHAIN, atol=2 ** -20,
        note="GELU's tanh form: an exp and a divide")

    keep = [1.0 if i % 3 else 0.0 for i in range(VLEN)]
    keep = vary(keep, 'mask')
    C['dropout'] = Case(
        'dropout',
        buffers={'x_ptr': ('fp32', x), 'keep_ptr': ('fp32', keep),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN, 'p': 0.5}, out='o_ptr',
        expect=[v / 0.5 if k > 0 else 0.0 for v, k in zip(x, keep)],
        rtol=TOL_DIV)

    q8 = [(i * 5) % 61 - 30 for i in range(VLEN)]
    q8 = vary(q8, 'int8')
    C['dequantize_rowwise'] = Case(
        'dequantize_rowwise',
        buffers={'q_ptr': ('i8', q8), 's_ptr': ('fp32', [0.25] * 4),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[v * 0.25 for v in q8])

    zeros = [7] * 4
    C['weight_dequant'] = Case(
        'weight_dequant',
        buffers={'q_ptr': ('i8', q8), 's_ptr': ('fp32', [0.125] * 4),
                 'z_ptr': ('i8', zeros), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[(v - 7) * 0.125 for v in q8])

    # rope reads x[0:n] and x[n:2n] and writes both halves.
    xr = ramp(VLEN, 0.25)
    xr = vary(xr)
    xi = ramp(VLEN, -0.125, 1.0)
    xi = vary(xi)
    cs = [0.5] * VLEN
    cs = vary(cs)
    sn = [0.25] * VLEN
    sn = vary(sn)
    C['rope'] = Case(
        'rope',
        buffers={'x_ptr': ('fp32', xr + xi), 'cos_ptr': ('fp32', cs),
                 'sin_ptr': ('fp32', sn),
                 'o_ptr': ('fp32', [0] * (2 * VLEN))},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[r * c - im * s for r, im, c, s in zip(xr, xi, cs, sn)]
        + [r * s + im * c for r, im, c, s in zip(xr, xi, cs, sn)])

    # ---- reduction and normalisation --------------------------------------
    rx = wavy(VLEN)
    rx = vary(rx)
    peak = max(rx)
    ex = [math.exp(v - peak) for v in rx]
    tot = sum(ex)
    C['softmax'] = Case(
        'softmax',
        buffers={'x_ptr': ('fp32', rx), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[v / tot for v in ex], rtol=TOL_EXP,
        note='tl.max, tl.exp, tl.sum and a divide')

    w = [1.0 + 0.125 * i for i in range(VLEN)]
    w = vary(w)
    eps = 0.0
    rms = math.sqrt(sum(v * v for v in rx) / VLEN + eps)
    C['rmsnorm'] = Case(
        'rmsnorm',
        buffers={'x_ptr': ('fp32', rx), 'w_ptr': ('fp32', w),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN, 'eps': eps}, out='o_ptr',
        expect=[(v / rms) * ww for v, ww in zip(rx, w)],
        rtol=TOL_CHAIN, note='a sqrt and a divide')

    mean = sum(rx) / VLEN
    d = [v - mean for v in rx]
    var = sum(v * v for v in d) / VLEN
    bias2 = [0.25 * i for i in range(VLEN)]
    bias2 = vary(bias2)
    C['layernorm'] = Case(
        'layernorm',
        buffers={'x_ptr': ('fp32', rx), 'w_ptr': ('fp32', w),
                 'b_ptr': ('fp32', bias2),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN, 'eps': 0.0}, out='o_ptr',
        expect=[(v / math.sqrt(var)) * ww + bb
                for v, ww, bb in zip(d, w, bias2)],
        rtol=TOL_CHAIN, atol=2 ** -20, note='two reductions, a sqrt, a divide')

    target = 5
    lse = peak + math.log(sum(math.exp(v - peak) for v in rx))
    C['cross_entropy'] = Case(
        'cross_entropy',
        buffers={'logit_ptr': ('fp32', rx), 'o_ptr': ('fp32', [0] * 4)},
        scalars={'target': target, 'n': VLEN}, out='o_ptr',
        expect=[lse - rx[target]], rtol=TOL_CHAIN,
        note='an exp, a sum and a log')

    pp = [0.0625 * ((i % 7) + 1) for i in range(VLEN)]
    pp = vary(pp, 'positive')
    qq = [0.0625 * ((i % 5) + 2) for i in range(VLEN)]
    qq = vary(qq, 'positive')
    C['kl_divergence'] = Case(
        'kl_divergence',
        buffers={'p_ptr': ('fp32', pp), 'q_ptr': ('fp32', qq),
                 'o_ptr': ('fp32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[sum(a * math.log(a / bq) for a, bq in zip(pp, qq))],
        rtol=TOL_CHAIN, atol=2 ** -18, note='a divide and a log per element')

    mu = [0.5] * VLEN
    mu = vary(mu)
    va = [4.0] * VLEN
    va = vary(va, 'positive')
    C['batch_normalization'] = Case(
        'batch_normalization',
        buffers={'x_ptr': ('fp32', x), 'mean_ptr': ('fp32', mu),
                 'var_ptr': ('fp32', va), 'w_ptr': ('fp32', w),
                 'b_ptr': ('fp32', bias2), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN, 'eps': 0.0}, out='o_ptr',
        expect=[((v - m) / math.sqrt(s)) * ww + bb
                for v, m, s, ww, bb in zip(x, mu, va, w, bias2)],
        rtol=TOL_SQRT, note='a sqrt and a divide')

    gex = [math.exp(v - peak) for v in rx]
    C['moe_topk_gating'] = Case(
        'moe_topk_gating',
        buffers={'logit_ptr': ('fp32', rx), 'o_ptr': ('fp32', [0] * 4),
                 'idx_ptr': ('i32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[max(gex) / sum(gex)], rtol=TOL_EXP,
        extra_outputs=[('idx_ptr', 0, [max(i for i, v in enumerate(rx)
                                         if v == max(rx))])],
        note='the winning softmax weight and expert index')

    hv = [(i * 5) % 4 for i in range(VLEN)]
    C['histogramming'] = Case(
        'histogramming',
        buffers={'x_ptr': ('i32', hv), 'o_ptr': ('i32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[sum(1 for v in hv if v == 0)])

    C['mean_reduction'] = Case(
        'mean_reduction',
        buffers={'x_ptr': ('fp32', rx), 'o_ptr': ('fp32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[sum(rx) / VLEN], rtol=TOL_DIV)

    C['l2_norm'] = Case(
        'l2_norm',
        buffers={'x_ptr': ('fp32', rx), 'o_ptr': ('fp32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[math.sqrt(sum(v * v for v in rx))], rtol=TOL_SQRT)

    C['argmax'] = Case(
        'argmax',
        buffers={'x_ptr': ('fp32', rx), 'o_ptr': ('i32', [0] * 4)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[rx.index(max(rx))])

    # ---- matmul and attention ---------------------------------------------
    A = [f16(small_ints(1, shift=5)[0] if False else
             float((i * 7 + j * 3) % 11 - 5))
         for i in range(T) for j in range(T)]
    A = vary(A)
    B = [f16(float((i * 5 + j * 2) % 9 - 4))
         for i in range(T) for j in range(T)]
    B = vary(B)

    def matmul(a, bm, M=T, N=T, K=T):
        return [sum(a[m * K + k] * bm[k * N + n] for k in range(K))
                for m in range(M) for n in range(N)]

    C['matmul_fp32_fp16_fp8'] = Case(
        'matmul_fp32_fp16_fp8',
        buffers={'a_ptr': ('fp16', A), 'b_ptr': ('fp16', B),
                 'c_ptr': ('fp32', [0] * (T * T))},
        out='c_ptr', expect=matmul(A, B),
        note='a real tl.dot, onto one m.mma.async')

    C['batched_matmul'] = Case(
        'batched_matmul',
        buffers={'a_ptr': ('fp16', A), 'b_ptr': ('fp16', B),
                 'c_ptr': ('fp16', [0] * (T * T))},
        out='c_ptr', expect=[f16(v) for v in matmul(A, B)],
        note='tl.make_tensor_descriptor, onto td.load and t.ld.async')

    Ai = [(i * 7 + j * 3) % 11 - 5 for i in range(T) for j in range(T)]
    Ai = vary(Ai, 'int8')
    Bi = [(i * 5 + j * 2) % 9 - 4 for i in range(T) for j in range(T)]
    Bi = vary(Bi, 'int8')
    C['matmul_int8'] = Case(
        'matmul_int8',
        buffers={'a_ptr': ('i8', Ai), 'b_ptr': ('i8', Bi),
                 'c_ptr': ('fp32', [0] * (T * T))},
        scalars={'scale': 0.25}, out='c_ptr',
        expect=[v * 0.25 for v in matmul([float(v) for v in Ai],
                                         [float(v) for v in Bi])],
        note='INT8 through the FP16 MMA, exactly (ISA sec 9C.1)')

    # Split-k: the bench runs program 0, so exactly one k slice lands in
    # C -- through 256 atomics.
    SPLIT = 2
    A2 = [f16(float((i * 7 + j * 3) % 11 - 5))
          for i in range(T) for j in range(T * SPLIT)]
    A2 = vary(A2)
    B2 = [f16(float((i * 5 + j * 2) % 9 - 4))
          for i in range(T * SPLIT) for j in range(T)]
    B2 = vary(B2)
    slice0_a = [A2[m * (T * SPLIT) + k] for m in range(T) for k in range(T)]
    slice0_b = [B2[k * T + n] for k in range(T) for n in range(T)]
    C['streamk_matmul'] = Case(
        'streamk_matmul',
        buffers={'a_ptr': ('fp16', A2), 'b_ptr': ('fp16', B2),
                 'c_ptr': ('fp32', [0] * (T * T))},
        out='c_ptr', expect=matmul(slice0_a, slice0_b), preserve_output=True,
        note='256 atomic adds, each a compare-and-swap loop')

    sm_scale = 0.25
    Q = [f16(float((i + j) % 5 - 2) * 0.5) for i in range(T) for j in range(T)]
    Q = vary(Q)
    K_ = [f16(float((i * 3 + j) % 7 - 3) * 0.25)
          for i in range(T) for j in range(T)]
    K_ = vary(K_)
    V = [f16(float((i * 2 + j * 5) % 6 - 3) * 0.5)
         for i in range(T) for j in range(T)]
    V = vary(V)

    def attention(q, k, v, scale):
        kt = [k[n * T + d] for d in range(T) for n in range(T)]
        s = [w * scale for w in matmul(q, kt)]
        out = []
        for m in range(T):
            row = s[m * T:(m + 1) * T]
            mx = max(row)
            p = [math.exp(w - mx) for w in row]
            tot = sum(p)
            p16 = [f16(w) for w in p]
            for n in range(T):
                out.append(sum(p16[kk] * v[kk * T + n] for kk in range(T))
                           / tot)
        return out

    C['flash_attention'] = Case(
        'flash_attention',
        buffers={'q_ptr': ('fp16', Q), 'k_ptr': ('fp16', K_),
                 'v_ptr': ('fp16', V), 'o_ptr': ('fp32', [0] * (T * T))},
        scalars={'sm_scale': sm_scale}, out='o_ptr',
        expect=attention(Q, K_, V, sm_scale), rtol=TOL_CHAIN, atol=2 ** -16,
        note='two tl.dots, an online softmax between them')

    mask = [1 if (i + j) % 3 else 0 for i in range(T) for j in range(T)]

    def sparse_attention(q, k, v, scale, keep):
        kt = [k[n * T + d] for d in range(T) for n in range(T)]
        s = [w * scale for w in matmul(q, kt)]
        out = []
        for m in range(T):
            row = [s[m * T + n] if keep[m * T + n] else -math.inf
                   for n in range(T)]
            mx = max(row)
            p = [math.exp(w - mx) if w != -math.inf else 0.0 for w in row]
            tot = sum(p)
            p16 = [f16(w) for w in p]
            for n in range(T):
                out.append(sum(p16[kk] * v[kk * T + n] for kk in range(T))
                           / tot)
        return out

    C['block_sparse_attention'] = Case(
        'block_sparse_attention',
        buffers={'q_ptr': ('fp16', Q), 'k_ptr': ('fp16', K_),
                 'v_ptr': ('fp16', V), 'mask_ptr': ('i32', mask),
                 'o_ptr': ('fp32', [0] * (T * T))},
        scalars={'sm_scale': sm_scale}, out='o_ptr',
        expect=sparse_attention(Q, K_, V, sm_scale, mask),
        rtol=TOL_CHAIN, atol=2 ** -16,
        note='a 32-bit mask tile, loaded a row at a time')

    def linear_attention(q, k, v):
        kt = [k[n * T + d] for d in range(T) for n in range(T)]
        kv = [f16(w) for w in matmul(kt, v)]
        return matmul(q, kv)

    C['linear_self_attention'] = Case(
        'linear_self_attention',
        buffers={'q_ptr': ('fp16', Q), 'k_ptr': ('fp16', K_),
                 'v_ptr': ('fp16', V), 'o_ptr': ('fp32', [0] * (T * T))},
        out='o_ptr', expect=linear_attention(Q, K_, V),
        note='a tl.dot whose operand is another one, repacked through '
             'memory')

    qd = [f16(float((i % 5) - 2) * 0.5) for i in range(T)]
    qd = vary(qd)

    def decode(q, k, v, scale):
        s = [sum(k[n * T + d] * q[d] for d in range(T)) * scale
             for n in range(T)]
        mx = max(s)
        p = [math.exp(w - mx) for w in s]
        tot = sum(p)
        return [sum(v[n * T + d] * p[n] for n in range(T)) / tot
                for d in range(T)]

    C['flash_decode'] = Case(
        'flash_decode',
        buffers={'q_ptr': ('fp32', [float(w) for w in qd]),
                 'k_ptr': ('fp32', [float(w) for w in K_]),
                 'v_ptr': ('fp32', [float(w) for w in V]),
                 'o_ptr': ('fp32', [0] * T)},
        scalars={'sm_scale': sm_scale}, out='o_ptr',
        expect=decode([float(w) for w in qd], [float(w) for w in K_],
                      [float(w) for w in V], sm_scale),
        rtol=TOL_CHAIN, atol=2 ** -18,
        note='one query row: a reduction down the tile')

    # ---- stencil and convolution ------------------------------------------
    N = 16
    conv_w = [0.5, 0.25, 0.125]
    conv_w = vary(conv_w)
    cx = ramp(N + 4, 0.25)
    cx = vary(cx)
    C['1d_conv'] = Case(
        'conv1d' if False else '1d_conv',
        buffers={'x_ptr': ('fp32', cx), 'w_ptr': ('fp32', conv_w),
                 'o_ptr': ('fp32', [0] * N)},
        scalars={'n': N, }, out='o_ptr',
        expect=[sum(conv_w[r] * (cx[i + r] if i + r < N else 0.0)
                    for r in range(3)) for i in range(N)])

    plane = [float((i * 3 + j) % 7) * 0.25
             for i in range(N + 4) for j in range(N + 4)]
    plane = vary(plane)

    def at(row, col):
        return plane[row * N + col] if 0 <= row * N + col < len(plane) else 0.0

    w2 = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
          0.5, 0.25, 0.125]
    w2 = vary(w2)
    C['2d_conv'] = Case(
        '2d_conv',
        buffers={'x_ptr': ('fp32', plane), 'w_ptr': ('fp32', w2),
                 'o_ptr': ('fp32', [0] * N)},
        out='o_ptr',
        expect=[sum(plane[(0 + dy) * N + c + dx] * w2[dy * 3 + dx]
                    for dy in range(3) for dx in range(3))
                for c in range(N)])

    w3 = [0.5 ** ((i % 5) + 1) for i in range(27)]
    w3 = vary(w3)
    cube = [float((i * 3 + j * 5) % 7) * 0.25
            for i in range(N * (N + 4)) for j in range(1)]
    cube = vary(cube)
    cube = [float((i * 3) % 7) * 0.25 for i in range(N * N * 3)]
    cube = vary(cube)
    C['3d_conv'] = Case(
        '3d_conv',
        buffers={'x_ptr': ('fp32', cube), 'w_ptr': ('fp32', w3),
                 'o_ptr': ('fp32', [0] * N)},
        out='o_ptr',
        expect=[sum(cube[((0 + dz) * N + 0 + dy) * N + c + dx]
                    * w3[(dz * 3 + dy) * 3 + dx]
                    for dz in range(3) for dy in range(3) for dx in range(3))
                for c in range(N)])

    PN = 32
    pool = [float((i * 5 + j * 3) % 13) * 0.25
            for i in range(4) for j in range(PN)]
    pool = vary(pool)
    C['2d_max_pooling'] = Case(
        '2d_max_pooling',
        buffers={'x_ptr': ('fp32', pool), 'o_ptr': ('fp32', [0] * VLEN)},
        out='o_ptr',
        expect=[max(pool[(0 * 2 + dy) * PN + c * 2 + dx]
                    for dy in range(2) for dx in range(2))
                for c in range(VLEN)])

    blur = [float((i * 7) % 11) * 0.25 for i in range(N * 3)]
    blur = vary(blur)
    C['gaussian_blur'] = Case(
        'gaussian_blur',
        buffers={'x_ptr': ('fp32', blur), 'o_ptr': ('fp32', [0] * N)},
        out='o_ptr',
        expect=[blur[c] * 0.5 + blur[c + 1] * 0.25 + blur[c + 2] * 0.25
                for c in range(N)])

    # The stencil reads the row above the one it writes, and with
    # program_id 0 that row is at a negative offset -- so the pointer
    # points one row into its buffer.
    jac = [float((i * 3) % 5) * 0.25 for i in range(N * 3)]
    jac = vary(jac)
    C['jacobi_stencil_2d'] = Case(
        'jacobi_stencil_2d',
        buffers={'x_ptr': ('fp32', jac), 'o_ptr': ('fp32', [0] * (N * 3))},
        out='o_ptr', offsets={'x_ptr': N, 'o_ptr': N}, out_offset=N,
        expect=[(jac[N + c] + jac[c] + jac[2 * N + c]) / 3.0
                for c in range(N)], rtol=TOL_DIV)

    # ---- data layout ------------------------------------------------------
    C['matrix_copy'] = Case(
        'matrix_copy',
        buffers={'x_ptr': ('fp32', x), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=list(x))

    M = [float((i * 7 + j) % 13) * 0.25 for i in range(T) for j in range(T)]
    M = vary(M)
    C['matrix_transpose'] = Case(
        'matrix_transpose',
        buffers={'x_ptr': ('fp32', M), 'o_ptr': ('fp32', [0] * (T * T))},
        out='o_ptr',
        expect=[M[n * T + m] for m in range(T) for n in range(T)],
        note='tl.trans, as ISA sec 9E mod[5]')

    C['reverse_array'] = Case(
        'reverse_array',
        buffers={'x_ptr': ('fp32', x), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=list(reversed(x)))

    perm = [(i * 7) % VLEN for i in range(VLEN)]
    perm = vary(perm, 'permutation')
    dest = [0.0] * VLEN
    for i, pv in enumerate(perm):
        dest[pv] = x[i]
    C['destindex'] = Case(
        'destindex',
        buffers={'x_ptr': ('fp32', x), 'idx_ptr': ('i32', perm),
                 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=dest)

    ia = ramp(VLEN, 0.5)
    ia = vary(ia)
    ib = ramp(VLEN, 0.25, 100.0)
    ib = vary(ib)
    inter = []
    for a_, b_ in zip(ia, ib):
        inter += [a_, b_]
    C['interleave'] = Case(
        'interleave',
        buffers={'a_ptr': ('fp32', ia), 'b_ptr': ('fp32', ib),
                 'o_ptr': ('fp32', [0] * (2 * VLEN))},
        scalars={'n': VLEN}, out='o_ptr', expect=inter)

    sv = [float((i * 11) % 16) for i in range(VLEN)]
    sv = vary(sv)
    C['bitonic_sort'] = Case(
        'bitonic_sort',
        buffers={'x_ptr': ('fp32', sv), 'o_ptr': ('fp32', [0] * VLEN)},
        out='o_ptr', expect=sorted(sv),
        note='four bitonic stages, entirely in registers')

    keys = [(i * 5 + 3) % 16 for i in range(VLEN)]
    keys = vary(keys, 'keys')
    zeros_before = 0
    ones_total = sum(1 for k in keys if k & 1)
    ones_before = 0
    rs = [0] * VLEN
    for i, k in enumerate(keys):
        if k & 1:
            rs[VLEN - ones_total + ones_before] = k
            ones_before += 1
        else:
            rs[zeros_before] = k
            zeros_before += 1
    C['radix_sort'] = Case(
        'radix_sort',
        buffers={'key_ptr': ('i32', keys), 'o_ptr': ('i32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr', expect=rs,
        note='one radix pass: a tt.scan, which is a fold with every '
             'partial kept')

    tv = wavy(VLEN, 0.5, -4.0)
    tv = vary(tv)
    ranks = [sum(1 for o in tv if o > v) for v in tv]
    C['top_k_selection'] = Case(
        'top_k_selection',
        buffers={'x_ptr': ('fp32', tv), 'o_ptr': ('fp32', [0] * VLEN)},
        scalars={'n': VLEN}, out='o_ptr',
        expect=[v if r < 4 else float('-inf') for v, r in zip(tv, ranks)],
        note='a 16x16 comparison, recomputed a row at a time')

    return C
