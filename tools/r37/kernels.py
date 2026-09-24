"""Representative Triton kernels for TileBench's operators.

These are written the way somebody would write them, not the way this
compiler would prefer. That is the point of the exercise: the goal is
coverage with the kernel unchanged, so a kernel bent around a missing
feature would measure the bend rather than the feature.

Each entry is (kernel fn, signature, constexprs). Adding an operator
means adding a kernel here and setting its `kernel` field in
spec/r37_tilebench.json.
"""
import triton
import triton.language as tl

B = 16          # one vector register on this machine
T = 16          # one matrix tile


# ---- point-wise ----------------------------------------------------------

@triton.jit
def vector_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0)
             + tl.load(y_ptr + offs, mask=m, other=0.0), mask=m)


@triton.jit
def mul2(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0) * 2.0,
             mask=m)


@triton.jit
def relu(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, tl.maximum(x, 0.0), mask=m)


@triton.jit
def leaky_relu(x_ptr, o_ptr, n, slope, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 0, x, x * slope), mask=m)


@triton.jit
def sigmoid(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=m)


@triton.jit
def swiglu(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, (a * tl.sigmoid(a)) * b, mask=m)


@triton.jit
def quantize_global(x_ptr, o_ptr, scale, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, (x / scale).to(tl.int8), mask=m)


# ---- reduction / normalisation -------------------------------------------

@triton.jit
def mean_reduction(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + tl.program_id(0), tl.sum(x, axis=0) / n)


@triton.jit
def l2_norm(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + tl.program_id(0), tl.sqrt(tl.sum(x * x, axis=0)))


@triton.jit
def softmax(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + row * n + offs, mask=m, other=-float('inf'))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(o_ptr + row * n + offs, e / tl.sum(e, axis=0), mask=m)


@triton.jit
def rmsnorm(x_ptr, w_ptr, o_ptr, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + row * n + offs, mask=m, other=0.0)
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    r = tl.sqrt(tl.sum(x * x, axis=0) / n + eps)
    tl.store(o_ptr + row * n + offs, (x / r) * w, mask=m)


@triton.jit
def argmax(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=-float('inf'))
    tl.store(o_ptr + tl.program_id(0), tl.argmax(x, axis=0))


# ---- matmul / attention ---------------------------------------------------

@triton.jit
def matmul_tile(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
                M: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
    b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
    tl.store(c_ptr + om[:, None] * N + on[None, :], tl.dot(a, b))


@triton.jit
def matmul_descriptor(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr):
    da = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                   block_shape=[16, 16])
    db = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1],
                                   block_shape=[16, 16])
    dc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[16, 16])
    dc.store([0, 0], tl.dot(da.load([0, 0]), db.load([0, 0])))


# ---- data layout ----------------------------------------------------------

@triton.jit
def matrix_copy(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0), mask=m)


@triton.jit
def matrix_transpose(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    x = tl.load(x_ptr + om[:, None] * N + on[None, :])
    tl.store(o_ptr + on[:, None] * M + om[None, :], tl.trans(x))


@triton.jit
def reverse_array(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + (n - 1 - offs), x, mask=m)


@triton.jit
def destindex(x_ptr, idx_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    i = tl.load(idx_ptr + offs, mask=m, other=0)
    tl.store(o_ptr + i, tl.load(x_ptr + offs, mask=m, other=0.0), mask=m)


# ---- stencil --------------------------------------------------------------

@triton.jit
def conv1d(x_ptr, w_ptr, o_ptr, n, BLOCK: tl.constexpr, R: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in tl.static_range(R):
        acc += tl.load(x_ptr + offs + r, mask=offs + r < n, other=0.0) \
            * tl.load(w_ptr + r)
    tl.store(o_ptr + offs, acc, mask=offs < n)


@triton.jit
def jacobi_stencil_2d(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    c = tl.load(x_ptr + row * N + offs)
    up = tl.load(x_ptr + (row - 1) * N + offs)
    dn = tl.load(x_ptr + (row + 1) * N + offs)
    tl.store(o_ptr + row * N + offs, (c + up + dn) / 3.0)


# ---- point-wise, continued ------------------------------------------------

@triton.jit
def fused_activation(x_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Bias, then GELU's tanh approximation, then a scale."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0) + tl.load(b_ptr + offs,
                                                           mask=m, other=0.0)
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    # tanh(z) = 1 - 2/(exp(2z) + 1)
    t = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + t), mask=m)


@triton.jit
def dropout(x_ptr, keep_ptr, o_ptr, n, p, BLOCK: tl.constexpr):
    """Inference-shaped dropout: a precomputed keep mask and a rescale."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    keep = tl.load(keep_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, tl.where(keep > 0.0, x / (1.0 - p), 0.0), mask=m)


@triton.jit
def dequantize_rowwise(q_ptr, s_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    q = tl.load(q_ptr + offs, mask=m, other=0)
    scale = tl.load(s_ptr + tl.program_id(0))
    tl.store(o_ptr + offs, q.to(tl.float32) * scale, mask=m)


@triton.jit
def weight_dequant(q_ptr, s_ptr, z_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Affine dequantisation: (q - zero) * scale, both per group."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    q = tl.load(q_ptr + offs, mask=m, other=0).to(tl.float32)
    zero = tl.load(z_ptr + tl.program_id(0)).to(tl.float32)
    tl.store(o_ptr + offs, (q - zero) * tl.load(s_ptr + tl.program_id(0)),
             mask=m)


@triton.jit
def rope(x_ptr, cos_ptr, sin_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Rotary embedding on an already-split pair of halves."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    xr = tl.load(x_ptr + offs, mask=m, other=0.0)
    xi = tl.load(x_ptr + offs + n, mask=m, other=0.0)
    c = tl.load(cos_ptr + offs, mask=m, other=1.0)
    s = tl.load(sin_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, xr * c - xi * s, mask=m)
    tl.store(o_ptr + offs + n, xr * s + xi * c, mask=m)


# ---- reduction and normalisation, continued -------------------------------

@triton.jit
def layernorm(x_ptr, w_ptr, b_ptr, o_ptr, n, eps, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    row = tl.program_id(0) * n
    x = tl.load(x_ptr + row + offs, mask=m, other=0.0)
    mean = tl.sum(x, axis=0) / n
    d = tl.where(m, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    y = d / tl.sqrt(var + eps)
    y = y * tl.load(w_ptr + offs, mask=m, other=1.0) + tl.load(
        b_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + row + offs, y, mask=m)


@triton.jit
def cross_entropy(logit_ptr, o_ptr, target, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    row = tl.program_id(0) * n
    z = tl.load(logit_ptr + row + offs, mask=m, other=-float('inf'))
    peak = tl.max(z, axis=0)
    lse = peak + tl.log(tl.sum(tl.exp(z - peak), axis=0))
    picked = tl.sum(tl.where(offs == target, z, 0.0), axis=0)
    tl.store(o_ptr + tl.program_id(0), lse - picked)


@triton.jit
def kl_divergence(p_ptr, q_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    row = tl.program_id(0) * n
    p = tl.load(p_ptr + row + offs, mask=m, other=0.0)
    q = tl.load(q_ptr + row + offs, mask=m, other=1.0)
    term = tl.where(p > 0.0, p * tl.log(p / q), 0.0)
    tl.store(o_ptr + tl.program_id(0), tl.sum(term, axis=0))


@triton.jit
def batch_normalization(x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, o_ptr, n,
                        eps, BLOCK: tl.constexpr):
    """Inference batch norm: the statistics are inputs, not reductions."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    mean = tl.load(mean_ptr + offs, mask=m, other=0.0)
    var = tl.load(var_ptr + offs, mask=m, other=1.0)
    y = (x - mean) / tl.sqrt(var + eps)
    tl.store(o_ptr + offs, y * tl.load(w_ptr + offs, mask=m, other=1.0)
             + tl.load(b_ptr + offs, mask=m, other=0.0), mask=m)


@triton.jit
def moe_topk_gating(logit_ptr, o_ptr, idx_ptr, n, BLOCK: tl.constexpr):
    """Top-1 gating: the winning expert's softmax weight, and which it is."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    row = tl.program_id(0) * n
    z = tl.load(logit_ptr + row + offs, mask=m, other=-float('inf'))
    peak = tl.max(z, axis=0)
    e = tl.exp(z - peak)
    weight = tl.max(e, axis=0) / tl.sum(e, axis=0)
    # The index of the maximum, without a two-input reduction.
    pick = tl.max(tl.where(z == peak, offs, 0), axis=0)
    tl.store(o_ptr + tl.program_id(0), weight)
    tl.store(idx_ptr + tl.program_id(0), pick)


@triton.jit
def histogramming(x_ptr, o_ptr, n, NBINS: tl.constexpr, BLOCK: tl.constexpr):
    """One bin per program: count the inputs that land in it."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=-1)
    tl.store(o_ptr + b, tl.sum(tl.where(m & (v == b), 1, 0), axis=0))


# ---- matmul and attention, continued --------------------------------------

@triton.jit
def matmul_int8(a_ptr, b_ptr, c_ptr, scale, K: tl.constexpr,
                N: tl.constexpr, M: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
    b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc * scale)


@triton.jit
def streamk_matmul(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
                   M: tl.constexpr, SPLIT: tl.constexpr):
    """One k-slice per program, accumulated into C atomically."""
    s = tl.program_id(0)
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(a_ptr + om[:, None] * (K * SPLIT) + s * K + ok[None, :])
    b = tl.load(b_ptr + (s * K + ok[:, None]) * N + on[None, :])
    tl.atomic_add(c_ptr + om[:, None] * N + on[None, :], tl.dot(a, b))


@triton.jit
def flash_attention(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, D: tl.constexpr,
                    L: tl.constexpr):
    """One key block, online-softmax shaped: the inner step of flash."""
    od = tl.arange(0, D)
    ol = tl.arange(0, L)
    q = tl.load(q_ptr + ol[:, None] * D + od[None, :])
    k = tl.load(k_ptr + ol[:, None] * D + od[None, :])
    v = tl.load(v_ptr + ol[:, None] * D + od[None, :])
    s = tl.dot(q, tl.trans(k)) * sm_scale
    peak = tl.max(s, axis=1)
    p = tl.exp(s - peak[:, None])
    acc = tl.dot(p.to(tl.float16), v)
    tl.store(o_ptr + ol[:, None] * D + od[None, :],
             acc / tl.sum(p, axis=1)[:, None])


@triton.jit
def flash_decode(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, D: tl.constexpr,
                 L: tl.constexpr):
    """Decode: one query row against a block of keys."""
    od = tl.arange(0, D)
    ol = tl.arange(0, L)
    q = tl.load(q_ptr + od)
    k = tl.load(k_ptr + ol[:, None] * D + od[None, :])
    v = tl.load(v_ptr + ol[:, None] * D + od[None, :])
    s = tl.sum(k * q[None, :], axis=1) * sm_scale
    p = tl.exp(s - tl.max(s, axis=0))
    acc = tl.sum(v * p[:, None], axis=0) / tl.sum(p, axis=0)
    tl.store(o_ptr + od, acc)


@triton.jit
def block_sparse_attention(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, sm_scale,
                           D: tl.constexpr, L: tl.constexpr):
    """Flash's inner step with a per-block skip mask."""
    od = tl.arange(0, D)
    ol = tl.arange(0, L)
    q = tl.load(q_ptr + ol[:, None] * D + od[None, :])
    k = tl.load(k_ptr + ol[:, None] * D + od[None, :])
    v = tl.load(v_ptr + ol[:, None] * D + od[None, :])
    keep = tl.load(mask_ptr + ol[:, None] * L + ol[None, :])
    s = tl.dot(q, tl.trans(k)) * sm_scale
    s = tl.where(keep > 0, s, -float('inf'))
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    acc = tl.dot(p.to(tl.float16), v)
    tl.store(o_ptr + ol[:, None] * D + od[None, :],
             acc / tl.sum(p, axis=1)[:, None])


@triton.jit
def linear_self_attention(q_ptr, k_ptr, v_ptr, o_ptr, D: tl.constexpr,
                          L: tl.constexpr):
    """Linear attention: (K^T V) first, so there is no L x L matrix."""
    od = tl.arange(0, D)
    ol = tl.arange(0, L)
    q = tl.load(q_ptr + ol[:, None] * D + od[None, :])
    k = tl.load(k_ptr + ol[:, None] * D + od[None, :])
    v = tl.load(v_ptr + ol[:, None] * D + od[None, :])
    kv = tl.dot(tl.trans(k), v).to(tl.float16)
    tl.store(o_ptr + ol[:, None] * D + od[None, :], tl.dot(q, kv))


# ---- stencil and convolution, continued -----------------------------------

@triton.jit
def conv2d(x_ptr, w_ptr, o_ptr, N: tl.constexpr, R: tl.constexpr,
           BLOCK: tl.constexpr):
    """A 3x3 direct convolution over one output row."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dy in tl.static_range(R):
        for dx in tl.static_range(R):
            acc += tl.load(x_ptr + (row + dy) * N + col + dx) * tl.load(
                w_ptr + dy * R + dx)
    tl.store(o_ptr + row * N + col, acc)


@triton.jit
def conv3d(x_ptr, w_ptr, o_ptr, N: tl.constexpr, R: tl.constexpr,
           BLOCK: tl.constexpr):
    """A 3x3x3 direct convolution over one output row of one plane."""
    plane = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dz in tl.static_range(R):
        for dy in tl.static_range(R):
            for dx in tl.static_range(R):
                base = ((plane + dz) * N + row + dy) * N + col + dx
                acc += tl.load(x_ptr + base) * tl.load(
                    w_ptr + (dz * R + dy) * R + dx)
    tl.store(o_ptr + (plane * N + row) * N + col, acc)


@triton.jit
def max_pool2d(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
               BLOCK: tl.constexpr):
    """2x2 max pooling over one output row."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
    for dy in tl.static_range(K):
        for dx in tl.static_range(K):
            acc = tl.maximum(acc, tl.load(
                x_ptr + (row * K + dy) * N + col * K + dx))
    tl.store(o_ptr + row * (N // K) + col, acc)


@triton.jit
def gaussian_blur(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """A separable 1-2-1 blur, applied along the row."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    base = x_ptr + row * N + col
    acc = tl.load(base) * 0.5 + tl.load(base + 1) * 0.25
    tl.store(o_ptr + row * N + col, acc + tl.load(base + 2) * 0.25)


# ---- data layout, continued -----------------------------------------------

@triton.jit
def interleave(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Two arrays into one, alternating."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs * 2, tl.load(a_ptr + offs, mask=m, other=0.0),
             mask=m)
    tl.store(o_ptr + offs * 2 + 1, tl.load(b_ptr + offs, mask=m, other=0.0),
             mask=m)


@triton.jit
def bitonic_sort(x_ptr, o_ptr, BLOCK: tl.constexpr, STAGES: tl.constexpr):
    """One bitonic block sort.

    The compare-exchange partner is lane `i ^ span`, which is a
    cross-lane read. A GPU does it with a warp shuffle; a machine
    without one does it through memory, and the barrier between the
    store and the gather is what says so.
    """
    i = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i)
    for k in tl.static_range(STAGES):
        for j in tl.static_range(k + 1):
            span = 1 << (k - j)
            tl.store(o_ptr + i, v)
            tl.debug_barrier()
            partner = tl.load(o_ptr + (i ^ span))
            up = ((i >> (k + 1)) & 1) == 0
            take_low = ((i & span) == 0) == up
            v = tl.where(take_low, tl.minimum(v, partner),
                         tl.maximum(v, partner))
    tl.store(o_ptr + i, v)


@triton.jit
def radix_sort(key_ptr, o_ptr, n, BIT: tl.constexpr, BLOCK: tl.constexpr):
    """One radix pass: stable scatter by a single bit."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    k = tl.load(key_ptr + offs, mask=m, other=0)
    bit = (k >> BIT) & 1
    ones = tl.sum(tl.where(m & (bit == 1), 1, 0), axis=0)
    zeros_before = tl.cumsum(tl.where(m & (bit == 0), 1, 0), axis=0) - 1
    ones_before = tl.cumsum(tl.where(m & (bit == 1), 1, 0), axis=0) - 1
    dest = tl.where(bit == 0, zeros_before, n - ones + ones_before)
    tl.store(o_ptr + dest, k, mask=m)


@triton.jit
def top_k_selection(x_ptr, o_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):
    """Rank-based top-k: an element is kept when fewer than K beat it."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=-float('inf'))
    rank = tl.sum(tl.where(v[None, :] > v[:, None], 1, 0), axis=1)
    tl.store(o_ptr + offs, tl.where(rank < K, v, -float('inf')), mask=m)


F32 = '*fp32'
I32 = 'i32'
CE = 'constexpr'

KERNELS = {
    'vector_add': (vector_add, {'x_ptr': F32, 'y_ptr': F32, 'o_ptr': F32,
                                'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'mul2': (mul2, {'x_ptr': F32, 'o_ptr': F32, 'n': I32, 'BLOCK': CE},
             {'BLOCK': B}),
    'relu': (relu, {'x_ptr': F32, 'o_ptr': F32, 'n': I32, 'BLOCK': CE},
             {'BLOCK': B}),
    'leaky_relu': (leaky_relu, {'x_ptr': F32, 'o_ptr': F32, 'n': I32,
                                'slope': 'fp32', 'BLOCK': CE}, {'BLOCK': B}),
    'sigmoid': (sigmoid, {'x_ptr': F32, 'o_ptr': F32, 'n': I32, 'BLOCK': CE},
                {'BLOCK': B}),
    'swiglu': (swiglu, {'a_ptr': F32, 'b_ptr': F32, 'o_ptr': F32, 'n': I32,
                        'BLOCK': CE}, {'BLOCK': B}),
    'quantize_global': (quantize_global, {'x_ptr': F32, 'o_ptr': '*i8',
                                          'scale': 'fp32', 'n': I32,
                                          'BLOCK': CE}, {'BLOCK': B}),
    'mean_reduction': (mean_reduction, {'x_ptr': F32, 'o_ptr': F32,
                                        'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'l2_norm': (l2_norm, {'x_ptr': F32, 'o_ptr': F32, 'n': I32, 'BLOCK': CE},
                {'BLOCK': B}),
    'softmax': (softmax, {'x_ptr': F32, 'o_ptr': F32, 'n': I32, 'BLOCK': CE},
                {'BLOCK': B}),
    'rmsnorm': (rmsnorm, {'x_ptr': F32, 'w_ptr': F32, 'o_ptr': F32,
                          'n': I32, 'eps': 'fp32', 'BLOCK': CE},
                {'BLOCK': B}),
    'argmax': (argmax, {'x_ptr': F32, 'o_ptr': '*i32', 'n': I32,
                        'BLOCK': CE}, {'BLOCK': B}),
    'matmul_fp32_fp16_fp8': (matmul_tile, {'a_ptr': '*fp16', 'b_ptr': '*fp16',
                                           'c_ptr': F32, 'K': CE, 'N': CE,
                                           'M': CE},
                             {'K': T, 'N': T, 'M': T}),
    'batched_matmul': (matmul_descriptor,
                       {'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': '*fp16',
                        'M': CE, 'N': CE, 'K': CE},
                       {'M': T, 'N': T, 'K': T}),
    'matrix_copy': (matrix_copy, {'x_ptr': F32, 'o_ptr': F32, 'n': I32,
                                  'BLOCK': CE}, {'BLOCK': B}),
    'matrix_transpose': (matrix_transpose, {'x_ptr': F32, 'o_ptr': F32,
                                            'M': CE, 'N': CE},
                         {'M': T, 'N': T}),
    'reverse_array': (reverse_array, {'x_ptr': F32, 'o_ptr': F32, 'n': I32,
                                      'BLOCK': CE}, {'BLOCK': B}),
    'destindex': (destindex, {'x_ptr': F32, 'idx_ptr': '*i32', 'o_ptr': F32,
                              'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    '1d_conv': (conv1d, {'x_ptr': F32, 'w_ptr': F32, 'o_ptr': F32, 'n': I32,
                         'BLOCK': CE, 'R': CE}, {'BLOCK': B, 'R': 3}),
    'jacobi_stencil_2d': (jacobi_stencil_2d, {'x_ptr': F32, 'o_ptr': F32,
                                              'N': CE, 'BLOCK': CE},
                          {'N': B, 'BLOCK': B}),
    'fused_activation': (fused_activation,
                         {'x_ptr': F32, 'b_ptr': F32, 'o_ptr': F32,
                          'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'dropout': (dropout, {'x_ptr': F32, 'keep_ptr': F32, 'o_ptr': F32,
                          'n': I32, 'p': 'fp32', 'BLOCK': CE}, {'BLOCK': B}),
    'dequantize_rowwise': (dequantize_rowwise,
                           {'q_ptr': '*i8', 's_ptr': F32, 'o_ptr': F32,
                            'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'weight_dequant': (weight_dequant,
                       {'q_ptr': '*i8', 's_ptr': F32, 'z_ptr': '*i8',
                        'o_ptr': F32, 'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'rope': (rope, {'x_ptr': F32, 'cos_ptr': F32, 'sin_ptr': F32,
                    'o_ptr': F32, 'n': I32, 'BLOCK': CE}, {'BLOCK': B}),

    'layernorm': (layernorm, {'x_ptr': F32, 'w_ptr': F32, 'b_ptr': F32,
                              'o_ptr': F32, 'n': I32, 'eps': 'fp32',
                              'BLOCK': CE}, {'BLOCK': B}),
    'cross_entropy': (cross_entropy, {'logit_ptr': F32, 'o_ptr': F32,
                                      'target': I32, 'n': I32, 'BLOCK': CE},
                      {'BLOCK': B}),
    'kl_divergence': (kl_divergence, {'p_ptr': F32, 'q_ptr': F32,
                                      'o_ptr': F32, 'n': I32, 'BLOCK': CE},
                      {'BLOCK': B}),
    'batch_normalization': (batch_normalization,
                            {'x_ptr': F32, 'mean_ptr': F32, 'var_ptr': F32,
                             'w_ptr': F32, 'b_ptr': F32, 'o_ptr': F32,
                             'n': I32, 'eps': 'fp32', 'BLOCK': CE},
                            {'BLOCK': B}),
    'moe_topk_gating': (moe_topk_gating,
                        {'logit_ptr': F32, 'o_ptr': F32, 'idx_ptr': '*i32',
                         'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'histogramming': (histogramming, {'x_ptr': '*i32', 'o_ptr': '*i32',
                                      'n': I32, 'NBINS': CE, 'BLOCK': CE},
                      {'NBINS': B, 'BLOCK': B}),

    'matmul_int8': (matmul_int8, {'a_ptr': '*i8', 'b_ptr': '*i8',
                                  'c_ptr': F32, 'scale': 'fp32', 'K': CE,
                                  'N': CE, 'M': CE}, {'K': T, 'N': T, 'M': T}),
    'streamk_matmul': (streamk_matmul,
                       {'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': F32,
                        'K': CE, 'N': CE, 'M': CE, 'SPLIT': CE},
                       {'K': T, 'N': T, 'M': T, 'SPLIT': 2}),
    'flash_attention': (flash_attention,
                        {'q_ptr': '*fp16', 'k_ptr': '*fp16', 'v_ptr': '*fp16',
                         'o_ptr': F32, 'sm_scale': 'fp32', 'D': CE, 'L': CE},
                        {'D': T, 'L': T}),
    'flash_decode': (flash_decode,
                     {'q_ptr': F32, 'k_ptr': F32, 'v_ptr': F32, 'o_ptr': F32,
                      'sm_scale': 'fp32', 'D': CE, 'L': CE}, {'D': T, 'L': T}),
    'block_sparse_attention': (block_sparse_attention,
                               {'q_ptr': '*fp16', 'k_ptr': '*fp16',
                                'v_ptr': '*fp16', 'o_ptr': F32,
                                'mask_ptr': '*i32', 'sm_scale': 'fp32',
                                'D': CE, 'L': CE}, {'D': T, 'L': T}),
    'linear_self_attention': (linear_self_attention,
                              {'q_ptr': '*fp16', 'k_ptr': '*fp16',
                               'v_ptr': '*fp16', 'o_ptr': F32, 'D': CE,
                               'L': CE}, {'D': T, 'L': T}),

    '2d_conv': (conv2d, {'x_ptr': F32, 'w_ptr': F32, 'o_ptr': F32, 'N': CE,
                         'R': CE, 'BLOCK': CE}, {'N': B, 'R': 3, 'BLOCK': B}),
    '3d_conv': (conv3d, {'x_ptr': F32, 'w_ptr': F32, 'o_ptr': F32, 'N': CE,
                         'R': CE, 'BLOCK': CE}, {'N': B, 'R': 3, 'BLOCK': B}),
    '2d_max_pooling': (max_pool2d, {'x_ptr': F32, 'o_ptr': F32, 'N': CE,
                                    'K': CE, 'BLOCK': CE},
                       {'N': B * 2, 'K': 2, 'BLOCK': B}),
    'gaussian_blur': (gaussian_blur, {'x_ptr': F32, 'o_ptr': F32, 'N': CE,
                                      'BLOCK': CE}, {'N': B, 'BLOCK': B}),

    'interleave': (interleave, {'a_ptr': F32, 'b_ptr': F32, 'o_ptr': F32,
                                'n': I32, 'BLOCK': CE}, {'BLOCK': B}),
    'bitonic_sort': (bitonic_sort, {'x_ptr': F32, 'o_ptr': F32, 'BLOCK': CE,
                                    'STAGES': CE}, {'BLOCK': B, 'STAGES': 4}),
    'radix_sort': (radix_sort, {'key_ptr': '*i32', 'o_ptr': '*i32', 'n': I32,
                                'BIT': CE, 'BLOCK': CE},
                   {'BIT': 0, 'BLOCK': B}),
    'top_k_selection': (top_k_selection, {'x_ptr': F32, 'o_ptr': F32,
                                          'n': I32, 'K': CE, 'BLOCK': CE},
                        {'K': 4, 'BLOCK': B}),
}
