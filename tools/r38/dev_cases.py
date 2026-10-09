"""S1-04 dev suite: parameterised points over unchanged TileBench kernels.

    python tools/r38/dev_cases.py --write     regenerate spec/r38/suites/dev.json

Each point is an ORIGINAL TileBench JIT function (loaded exactly as R3.7
loads it) at a new shape, with inputs from a seeded generator and an
independent Python reference. Points are grouped into shape families
(operator x size range) and whole families are assigned to dev /
validation / heldout, so a held-out family's neighbours never appear in
training (roadmap section 5.3).

The split is a deterministic function of the family name, fixed before
any point is measured.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'r37'))
sys.path.insert(0, str(HERE))
import cases as C                                          # noqa: E402
import execute as E                                        # noqa: E402
import upstream as U                                       # noqa: E402
from archbench import bounds as B                          # noqa: E402

SUITE = ROOT / 'spec' / 'r38' / 'suites' / 'dev.json'
SPLIT = (('dev', 60), ('validation', 20), ('heldout', 20))

ELEMENTWISE_N = [5, 16, 17, 31, 48, 64, 100, 128, 160, 200, 256]
ROW_N = [3, 16, 17, 33, 64, 100, 128, 200]
ROWS = [1, 3]
TRANSPOSE = [(16, 16), (16, 48), (32, 32), (48, 16), (48, 48), (17, 33), (64, 32), (33, 64), (64, 64)]
GEMM = [(16, 16, 16), (16, 16, 32), (16, 16, 64), (16, 16, 128), (32, 16, 32), (16, 32, 32),
        (32, 32, 32), (32, 32, 64), (48, 16, 48), (64, 16, 16), (32, 32, 128), (64, 64, 32)]


def bucket(size):
    return 'xs' if size <= 16 else 's' if size <= 64 else 'm' if size <= 160 else 'l'


def split_of(family):
    h = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 100
    acc = 0
    for name, share in SPLIT:
        acc += share
        if h < acc:
            return name
    return SPLIT[-1][0]


def values(seed, n, lo=-2.0, hi=2.0, step=.125):
    r = random.Random(seed)
    k = int((hi - lo) / step)
    return [lo + step * r.randrange(k + 1) for _ in range(n)]


def _root(source):
    root = Path(source) / 'benchmarks/operators'
    return root if root.exists() else Path(source) / 'tilebench/benchmarks/operators'


def build(source):
    root = _root(source)
    out = {}
    fns = {}

    def add(pid, oid, fn_name, buffers, scalars, output, expect, workload, family_size,
            *, grid=(1, 1), ce=None, types=None, rtol=0.0, atol=0.0, descriptors=None, shape=None):
        key = (oid, fn_name)
        if key not in fns:
            fns[key] = U.load(root / oid / 'impl_triton.py')[fn_name]
        fn = fns[key]
        sig, constexprs, _ = U.configuration(oid, fn)
        constexprs.update(ce or {})
        for name, (dtype, _) in buffers.items():
            if name in sig:
                sig[name] = '*' + dtype
        sig.update(types or {})
        case = E.Case(oid, kernel_spec=(fn, sig, constexprs), buffers=buffers, scalars=scalars,
                      out=output, expect=expect, rtol=rtol, atol=atol,
                      program_id=(grid[0] - 1, grid[1] - 1, 0), descriptors=descriptors or {})
        family = f'{oid}/{bucket(family_size)}'
        workload.id = pid
        inputs = json.dumps({k: v for k, v in buffers.items()}, sort_keys=True)
        out[pid] = dict(case=case, workload=workload, family=family, split=split_of(family), shape=shape,
                        operator=oid, input_sha256=hashlib.sha256(inputs.encode()).hexdigest())

    unary = (
        ('mul2', 'mul2_kernel', 'output_ptr', lambda v: v * 2, 1, 0, 0.0),
        ('relu', 'relu_kernel', 'output_ptr', lambda v: max(v, 0), 1, 0, 0.0),
        ('leaky_relu', 'leaky_relu_kernel', 'y_ptr', lambda v: v if v > 0 else C.f32(v * C.f32(.01)), 1, 0, 0.0),
        ('sigmoid', 'sigmoid_kernel', 'y_ptr', lambda v: 1 / (1 + math.exp(-v)), 2, 1, C.TOL_EXP),
    )
    for n in ELEMENTWISE_N:
        g = (math.ceil(n / 16), 1)
        for oid, fn, dst, ref, fp, sfu, tol in unary:
            x = values(f'{oid}/{n}', n)
            add(f'{oid}/n{n}', oid, fn, {'x_ptr': ('fp32', x), dst: ('fp32', [0] * n)},
                {'n_elements': n}, dst, [ref(v) for v in x],
                B.elementwise('', oid, n, inputs=1, outputs=1, fp_per_elem=fp, sfu_per_elem=sfu), n,
                grid=g, rtol=tol, shape={'n': n})
        x, y = values(f'add-x/{n}', n), values(f'add-y/{n}', n)
        add(f'vector_add/n{n}', 'vector_add', 'add_kernel',
            {'x_ptr': ('fp32', x), 'y_ptr': ('fp32', y), 'output_ptr': ('fp32', [0] * n)},
            {'n_elements': n}, 'output_ptr', [a + b for a, b in zip(x, y)],
            B.elementwise('', 'vector_add', n, inputs=2, outputs=1, fp_per_elem=1), n, grid=g, shape={'n': n})
        add(f'swiglu/n{n}', 'swiglu', 'swiglu_kernel',
            {'x_ptr': ('fp32', x), 'y_ptr': ('fp32', y), 'out_ptr': ('fp32', [0] * n)},
            {'n_elements': n}, 'out_ptr', [a / (1 + math.exp(-a)) * b for a, b in zip(x, y)],
            B.elementwise('', 'swiglu', n, inputs=2, outputs=1, fp_per_elem=3, sfu_per_elem=1), n,
            grid=g, rtol=C.TOL_EXP, shape={'n': n})
        b = values(f'bias/{n}', n, -1, 1)
        z = [a * c + d for a, c, d in zip(x, y, b)]
        add(f'fused_activation/n{n}', 'fused_activation', 'fused_activation_kernel',
            {'x_ptr': ('fp32', x), 'gate_ptr': ('fp32', y), 'bias_ptr': ('fp32', b), 'out_ptr': ('fp32', [0] * n)},
            {'n_elements': n}, 'out_ptr', [v / (1 + math.exp(-v)) for v in z],
            B.elementwise('', 'fused_activation', n, inputs=3, outputs=1, fp_per_elem=3, sfu_per_elem=1), n,
            grid=g, rtol=C.TOL_EXP, shape={'n': n})
        add(f'reverse_array/n{n}', 'reverse_array', 'reverse_kernel',
            {'input_ptr': ('fp32', x), 'output_ptr': ('fp32', [0] * n)}, {'N': n}, 'output_ptr',
            list(reversed(x)), B.copy('', 'reverse_array', n), n, grid=g, shape={'n': n})

    for rows in ROWS:
        for n in ROW_N:
            size = rows * n
            x = values(f'row/{rows}/{n}', size, -1.5, 1.5)
            w = values(f'w/{n}', n, .5, 1.5)
            bias = values(f'b/{n}', n, -.5, .5)
            sm, ln, rn = [], [], []
            for r in range(rows):
                row = x[r * n:(r + 1) * n]
                p = [math.exp(v - max(row)) for v in row]
                sm += [v / sum(p) for v in p]
                mu = sum(row) / n
                sc = math.sqrt(sum(v * v for v in row) / n - mu * mu + .125)
                ln += [(v - mu) / sc * ww + bb for v, ww, bb in zip(row, w, bias)]
                rs = math.sqrt(sum(v * v for v in row) / n + .125)
                rn += [v / rs * ww for v, ww in zip(row, w)]
            add(f'softmax/r{rows}n{n}', 'softmax', 'softmax_online_kernel',
                {'input_ptr': ('fp32', x), 'output_ptr': ('fp32', [0] * size)},
                {'input_row_stride': n, 'output_row_stride': n, 'n_cols': n}, 'output_ptr', sm,
                B.row_reduction('', 'softmax', rows, n, writes_per_row=0, writes_per_elem=1,
                                fp_per_elem=3, sfu_per_elem=1), size, grid=(rows, 1), rtol=C.TOL_CHAIN, shape={'rows': rows, 'n': n})
            lw = B.row_reduction('', 'layernorm', rows, n, writes_per_row=0, writes_per_elem=1, fp_per_elem=5)
            lw.read_bytes = (size + 2 * n) * 4
            add(f'layernorm/r{rows}n{n}', 'layernorm', 'layernorm_kernel',
                {'x_ptr': ('fp32', x), 'weight_ptr': ('fp32', w), 'bias_ptr': ('fp32', bias),
                 'out_ptr': ('fp32', [0] * size)},
                {'stride_row': n, 'N_SIZE': n, 'eps': .125}, 'out_ptr', ln, lw, size,
                grid=(rows, 1), rtol=C.TOL_CHAIN, atol=2 ** -16, shape={'rows': rows, 'n': n})
            rw = B.row_reduction('', 'rmsnorm', rows, n, writes_per_row=0, writes_per_elem=1, fp_per_elem=3)
            rw.read_bytes = (size + n) * 4
            add(f'rmsnorm/r{rows}n{n}', 'rmsnorm', 'rmsnorm_kernel',
                {'x_ptr': ('fp32', x), 'rms_w_ptr': ('fp32', w), 'out_ptr': ('fp32', [0] * size)},
                {'stride_row': n, 'N_SIZE': n, 'eps': .125}, 'out_ptr', rn, rw, size,
                grid=(rows, 1), rtol=C.TOL_CHAIN, atol=2 ** -18, shape={'rows': rows, 'n': n})

    for m, n in TRANSPOSE:
        a = values(f'tr/{m}/{n}', m * n)
        add(f'matrix_transpose/{m}x{n}', 'matrix_transpose', 'transpose_kernel',
            {'x_ptr': ('fp32', a), 'output_ptr': ('fp32', [0] * (m * n))},
            {'m': m, 'n': n, 'stride_xm': n, 'stride_xn': 1, 'stride_om': m, 'stride_on': 1},
            'output_ptr', [a[j * n + i] for i in range(n) for j in range(m)],
            B.copy('', 'matrix_transpose', m * n), m * n,
            grid=(math.ceil(m / 16), math.ceil(n / 16)), shape={'m': m, 'n': n})

    for M, N, K in GEMM:
        a = values(f'ga/{M}/{N}/{K}', M * K, -1, 1, .5)
        bt = values(f'gb/{M}/{N}/{K}', N * K, -1, 1, .5)
        expected = [sum(a[i * K + k] * bt[j * K + k] for k in range(K)) for i in range(M) for j in range(N)]
        desc = {'a_desc': {'buffer': 'a', 'shape': [M, K], 'strides': [K, 1], 'block': [16, 16]},
                'b_desc': {'buffer': 'bt', 'shape': [N, K], 'strides': [K, 1], 'block': [16, 16]},
                'c_desc': {'buffer': 'c', 'shape': [M, N], 'strides': [N, 1], 'block': [16, 16]}}
        add(f'matmul_fp16/{M}x{N}x{K}', 'matmul_fp32_fp16_fp8', 'matmul_kernel',
            {'a': ('fp16', a), 'bt': ('fp16', bt), 'c': ('fp16', [0] * (M * N))},
            {'M': M, 'N': N, 'K': K}, 'c', expected,
            B.gemm('', 'matmul_fp32_fp16_fp8', M, N, K, in_dtype='fp16', out_dtype='fp16'), M * N * K // 16,
            grid=((M // 16) * (N // 16), 1), descriptors=desc,
            types={k: 'tensordesc<fp16[16,16]>' for k in desc}, shape={'M': M, 'N': N, 'K': K})
    return out


def suite(points):
    fams = sorted({p['family'] for p in points.values()})
    return {
        'schema': 'r38-suite/1', 'id': 'dev',
        'contract': 'doc/r38_stage1_tasks_and_acceptance.md S1-04: parameterised dev/validation/heldout suite',
        'generator': 'tools/r38/dev_cases.py',
        'note': 'Split by whole shape family (operator x size range), a deterministic function of the family '
                'name. Held-out points may be read once, by the S1-06 final evaluation.',
        'level': 2, 'repeats': 1, 'wall_budget_seconds': 7200,
        'split_rule': {'hash': 'sha256(family) % 100', 'shares': dict(SPLIT)},
        'families': {f: split_of(f) for f in fams},
        'points': [{'id': pid, 'operator': p['operator'], 'family': p['family'], 'split': p['split'],
                    'shape': p['shape'], 'input_sha256': p['input_sha256']}
                   for pid, p in sorted(points.items())],
    }


if __name__ == '__main__':
    import os
    src = Path(os.environ.get('TILEBENCH_SOURCE', ROOT.parent / 'Tilebench'))
    pts = build(src)
    s = suite(pts)
    counts = {k: sum(p['split'] == k for p in s['points']) for k, _ in SPLIT}
    ops = sorted({p['operator'] for p in s['points']})
    print(f'{len(pts)} points, {len(ops)} operators, {len(s["families"])} families; {counts}')
    if '--write' in sys.argv:
        SUITE.write_text(json.dumps(s, indent=1) + '\n')
        print(f'wrote {SUITE.relative_to(ROOT)}')
