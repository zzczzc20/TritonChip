#!/usr/bin/env python3
"""P1+ T3/T4: small PyTorch graphs through torch.compile -> Inductor -> TNCA.

Each graph is compiled with `tnca.inductor.configure()` in force, run on
fresh random inputs, and compared against eager PyTorch on the CPU. Inductor's
generated wrapper code is captured (torch._inductor.utils.run_and_get_code, the
upstream test helper) and inspected: every Triton kernel launched must be a
TNCA launch, and any computation left on the host -- an extern kernel, an ATen
call, a C++ kernel -- is a FALLBACK. A graph in the must-offload list fails
on any fallback; `fallback_detected` is the T4 control, a graph with an op
this path cannot offload, and it must be REPORTED as a fallback rather than
silently computed.

    ~/src/torchenv/bin/python tools/r38/torch_tests.py [--only NAME ...]
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import torch                                                # noqa: E402
import tnca.inductor                                        # noqa: E402

CONFIG = tnca.inductor.configure()
from torch._inductor.utils import run_and_get_code          # noqa: E402

# Host-side computation in Inductor's generated wrapper. Buffer allocation,
# reinterpretation and views are not computation and are not listed.
FALLBACK = [
    (re.compile(r'\bextern_kernels\.\w+'), 'extern kernel'),
    (re.compile(r'\btorch\.ops\.aten\.(?!empty|empty_strided|as_strided|reinterpret)\w+'), 'ATen op'),
    (re.compile(r'\baten\.(?!empty|empty_strided|as_strided)\w+\('), 'ATen op'),
    (re.compile(r'\bcpp_fused\w*'), 'C++ kernel'),
]


def _executed(src):
    """The wrapper as it runs: no comments (Inductor annotates each kernel
    launch with the ATen ops it replaced), and nothing from the benchmark
    harness Inductor appends after `def benchmark_compiled_module`."""
    src = src.split('def benchmark_compiled_module', 1)[0]
    return '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith('#'))


def fallbacks(code):
    found = []
    for src in map(_executed, code):
        for pat, what in FALLBACK:
            for m in pat.finditer(src):
                found.append(f'{what}: {m.group(0)}')
    return sorted(set(found))


def kernels(code):
    return sorted({m.group(1) for src in code for m in re.finditer(r'def (triton_\w+)\(', src)})


def g_elementwise(x, y):
    return x * 2 + y


def g_fused_act(x, y, b):
    return torch.nn.functional.silu(x) * y + b


def g_row_sum(x):
    return x.sum(dim=-1)


def g_softmax(x):
    return torch.softmax(x, dim=-1)


def g_rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * w


def g_matmul(a, b):
    return a @ b


def g_mlp_block(x, w, g):
    h = x @ w
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * g
    return torch.nn.functional.silu(h)


def g_softmax_nan(x):
    return torch.softmax(x, dim=-1)


def make_softmax_nan():
    x = torch.randn(3, 37)
    x[1, 5] = float('nan')                        # row 1 must come out all NaN
    return (x,)


def g_sliced_view(x):
    return x[:, 3:] * 2 + 1                      # storage offset, non-contiguous rows


def g_transposed_input(x, y):
    return x.t() + y                             # a strided (transposed) read


def g_inplace_add(x, y):
    x.add_(y)                                    # input and output are one storage
    return x


def g_fallback_detected(x):
    return torch.sort(x, dim=-1).values          # no Triton lowering on this path


GRAPHS = {
    'elementwise': (g_elementwise, lambda: (torch.randn(37), torch.randn(37)), 0.0, True),
    'fused_act': (g_fused_act, lambda: (torch.randn(45), torch.randn(45), torch.randn(45)), 2 ** -17, True),
    'row_sum': (g_row_sum, lambda: (torch.randn(3, 37),), 2 ** -18, True),
    'softmax': (g_softmax, lambda: (torch.randn(3, 37),), 2 ** -17, True),
    'rmsnorm': (g_rmsnorm, lambda: (torch.randn(3, 37), torch.randn(37)), 2 ** -16, True),
    'matmul': (g_matmul, lambda: (torch.randn(16, 32), torch.randn(32, 16)), 2 ** -16, True),
    'mlp_block': (g_mlp_block, lambda: (torch.randn(16, 32), torch.randn(32, 16), torch.randn(16)), 2 ** -14, True),
    'softmax_nan': (g_softmax_nan, make_softmax_nan, 2 ** -17, True),
    'sliced_view': (g_sliced_view, lambda: (torch.randn(3, 37),), 0.0, True),
    'transposed_input': (g_transposed_input, lambda: (torch.randn(20, 12), torch.randn(12, 20)), 0.0, True),
    'inplace_add': (g_inplace_add, lambda: (torch.randn(37), torch.randn(37)), 0.0, True),
    'fallback_detected': (g_fallback_detected, lambda: (torch.randn(3, 8),), 0.0, False),
}


def run(name):
    fn, make, tol, must_offload = GRAPHS[name]
    torch._dynamo.reset()
    torch.manual_seed(0)
    inputs = make()
    # Eager on its own copy: an in-place graph must not see the compiled
    # run's writes, and the compiled run must start from the same values.
    want = fn(*[t.clone() for t in inputs])
    compiled = torch.compile(fn)
    got, code = run_and_get_code(compiled, *inputs)
    nan_want, nan_got = torch.isnan(want), torch.isnan(got)
    nan_ok = bool((nan_want == nan_got).all())
    finite = ~nan_want
    err = (got[finite] - want[finite]).abs().max().item() if finite.any() else 0.0
    scale = max(want[finite].abs().max().item() if finite.any() else 1.0, 1.0)
    fb = fallbacks(code)
    ks = kernels(code)
    problems = []
    if not nan_ok:
        problems.append(f'NaN positions differ: eager {int(nan_want.sum())}, compiled {int(nan_got.sum())}')
    if torch.isinf(got).any() and not torch.isinf(want).any():
        problems.append('infinite output')
    if err > tol * scale:
        problems.append(f'max error {err:.3g} > {tol * scale:.3g}')
    if must_offload and fb:
        problems.append(f'host fallback: {fb}')
    if must_offload and not ks:
        problems.append('no Triton kernel was generated')
    if not must_offload and not fb:
        problems.append('an op this path cannot offload was not reported as a fallback')
    return dict(name=name, ok=not problems, problems=problems, max_error=err, kernels=ks, fallbacks=fb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--json', type=Path, default=ROOT / 'sim' / 'build' / 'r38' / 'torch' / 'torch_tests.json')
    args = ap.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in GRAPHS:
        if args.only and name not in args.only:
            continue
        t = time.monotonic()
        try:
            row = run(name)
        except Exception as e:                            # noqa: BLE001
            row = dict(name=name, ok=False, problems=[f'{type(e).__name__}: {str(e)[-1200:]}'])
        row['seconds'] = round(time.monotonic() - t, 1)
        rows.append(row)
        print(f'{"PASS" if row["ok"] else "FAIL"} {name}  kernels={row.get("kernels")} '
              f'fallbacks={row.get("fallbacks")} err={row.get("max_error")}'
              + ''.join(f'\n     {p}' for p in row['problems']), flush=True)
    args.json.write_text(json.dumps(dict(config=CONFIG, rows=rows), indent=1, default=str) + '\n')
    print(f'torch: {sum(r["ok"] for r in rows)}/{len(rows)} pass')
    return 0 if rows and all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
