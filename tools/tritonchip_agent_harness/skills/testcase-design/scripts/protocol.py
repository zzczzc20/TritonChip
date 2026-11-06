"""The evaluation's correctness protocol, for test cases of your own.

This is a copy of the code the evaluation judges a point with, so a case you
judge here is judged the way the hidden benchmark judges it. The evaluator's
copy is the original; a test in the harness keeps the two equal.

Reference   eager PyTorch in binary32 on the CPU, from the same module and
            inputs, every low-precision weight and input upcast exactly, TF32
            off. The tolerance therefore measures the device's arithmetic,
            not the reference's.
Criterion   max |got - ref| / max(max |ref|, 1) over every output <= tol.
            NaN positions must agree. Integer outputs must be exact.
Tolerance   by the dtype the case computes in (TOLERANCE); a whole model in
            fp16 gets MODEL_TOLERANCE.

`torch.testing.assert_close` is NOT this criterion. It is per element and,
for fp16, about twenty times stricter, so it rejects changes the evaluation
accepts (a different accumulation order, say), and it says nothing about
NaN positions.
"""
import copy

import torch

TOLERANCE = {'fp32': 2e-4, 'fp16': 2e-2, 'bf16': 6e-2}
MODEL_TOLERANCE = {'fp16': 5e-2}
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16',
               # The two OCP FP8 encodings, as the evaluation's registry names them: GEMM
               # operand formats only; a case in them computes a bf16 or fp16 result and is
               # judged at that dtype's tolerance (TOLERANCE has no FP8 entry on purpose).
               'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}
_NAME = {torch.float32: 'fp32', torch.float16: 'fp16', torch.bfloat16: 'bf16'}


def flatten(out):
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)):
        return tuple(t for o in out for t in flatten(o))
    raise TypeError(f'output of type {type(out).__name__}')


def to_fp32(module, inputs, device='cpu'):
    module = module.float().to(device)
    ins = tuple(t.float().to(device) if t.is_floating_point() else t.to(device) for t in inputs)
    return module, ins


class ieee_fp32:
    """IEEE binary32 only: no TF32, highest matmul precision, for the duration."""

    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                      torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')

    def __exit__(self, *exc):
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, p) = self.saved
        torch.set_float32_matmul_precision(p)


@torch.no_grad()
def reference(module, inputs, device='cpu'):
    """The reference outputs, as CPU tensors. The module you pass is left as it is."""
    m, ins = to_fp32(copy.deepcopy(module).eval(), inputs, device)
    with ieee_fp32():
        return tuple(t.detach().cpu() for t in flatten(m(*ins)))


def compare(got, ref, tol):
    """max |got - ref| / max(max |ref|, 1) over all outputs; NaN positions
    must agree and integer outputs must be exact."""
    got = flatten(got)
    if len(got) != len(ref):
        return dict(ok=False, detail=f'{len(got)} outputs, reference has {len(ref)}', max_rel_err=None)
    worst, problems = 0.0, []
    for i, (g, r) in enumerate(zip(got, ref)):
        g = g.detach().cpu()
        if tuple(g.shape) != tuple(r.shape):
            problems.append(f'output {i}: shape {tuple(g.shape)} != {tuple(r.shape)}')
            continue
        if not r.is_floating_point():
            bad = int((g.to(r.dtype) != r).sum())
            if bad:
                problems.append(f'output {i}: {bad} integer elements differ')
            continue
        g = g.float()
        gn, rn = torch.isnan(g), torch.isnan(r)
        if not bool((gn == rn).all()):
            problems.append(f'output {i}: NaN positions differ ({int(gn.sum())} vs {int(rn.sum())})')
        fin = ~rn
        if fin.any():
            err = (g[fin] - r[fin]).abs().max().item()
            scale = max(r[fin].abs().max().item(), 1.0)
            worst = max(worst, err / scale)
    if worst > tol:
        problems.append(f'max relative error {worst:.3g} > {tol:.3g}')
    return dict(ok=not problems, detail='; '.join(problems), max_rel_err=worst)


def dtype_of(module, inputs):
    """The dtype a case computes in: the first floating-point input, else the
    first floating-point parameter. None when there is neither."""
    for t in list(inputs) + list(module.parameters()) + list(module.buffers()):
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype in _NAME:
            return _NAME[t.dtype]
    return None


def tolerance(dtype, model=False):
    """The protocol's tolerance for a case in `dtype`; `model` for a whole fp16 model."""
    if model and dtype in MODEL_TOLERANCE:
        return MODEL_TOLERANCE[dtype]
    return TOLERANCE[dtype]


def classify(exc):
    """A failure status from an exception, by the codes this repository
    raises (python/tnca/errors.py) as they surface through Inductor."""
    text = f'{type(exc).__name__}: {exc}'
    for e in _chain(exc):
        text += f'\n{type(e).__name__}: {e}'
    t = text
    if any(k in t for k in ('BUFFER', 'overlaps the device heap', 'heap exhausted', 'does not fit',
                            'CAPACITY', 'out of device memory')):
        return 'capacity-exceeded'
    if any(k in t for k in ('UNSUPPORTED', 'Unsupported', 'COMPILE_FAILED', 'CompilationError',
                            'LoweringException', 'NotImplementedError')):
        return 'compile-failure'
    if any(k in t for k in ('TIMEOUT', 'Timeout', 'WAIT_TIMEOUT')):
        return 'timeout'
    return 'runtime-fault'


def _chain(e):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__
