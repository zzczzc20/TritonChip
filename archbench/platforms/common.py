"""What every platform runner shares: the reference, the comparison, the
failure taxonomy and the record."""
import datetime
import platform as _platform
import traceback

import torch

SCHEMA = 'archbench-torch-record/1'

# roadmap section 4.8: failures are kept apart, never merged into "slow".
STATUSES = ('pass', 'numeric-error', 'compile-failure', 'runtime-fault', 'capacity-exceeded',
            'timeout', 'unsupported', 'not-measured')


def flatten(out):
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)):
        return tuple(t for o in out for t in flatten(o))
    raise TypeError(f'output of type {type(out).__name__}')


def _to_fp32(module, inputs, device):
    module = module.float().to(device)
    ins = tuple(t.float().to(device) if t.is_floating_point() else t.to(device) for t in inputs)
    return module, ins


@torch.no_grad()
def reference(point, seed, device='cpu'):
    """Eager PyTorch in binary32 on `device`, from a fresh build of the point.

    Low-precision weights and inputs upcast exactly, so the reference
    computes the same mathematical function at higher precision, and the
    point's tolerance measures the platform's arithmetic, not the
    reference's. TF32 is off for the duration (protocol: IEEE fp32 only).
    """
    m, ins = point.make(seed)
    m, ins = _to_fp32(m.eval(), ins, device)
    with _ieee_fp32():
        return tuple(t.detach().cpu() for t in flatten(m(*ins)))


class _ieee_fp32:
    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                      torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')

    def __exit__(self, *exc):
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, p) = self.saved
        torch.set_float32_matmul_precision(p)


ieee_fp32 = _ieee_fp32


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


def error_text(exc, limit=3000):
    s = ''.join(traceback.format_exception_only(type(exc), exc))
    for e in list(_chain(exc))[1:4]:
        s += f'\n caused by {type(e).__name__}: {str(e)[:800]}'
    return s[-limit:]


def environment():
    import triton
    return dict(torch=torch.__version__, triton=triton.__version__,
                python=_platform.python_version(), machine=_platform.machine(),
                host=_platform.node())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def base_record(point, platform, evidence_level):
    return dict(schema=SCHEMA, point_id=point.id, point_sha256=point.definition_sha256(),
                kind=point.kind, family=point.family, tier=point.tier, dtype=point.dtype,
                track=list(point.track), platform=platform, evidence_level=evidence_level,
                status='not-measured', problems=[], fallbacks=[], kernels=[],
                correctness=None, time=None, demand=point.demand.as_dict(), bound=None,
                eta=None, host=None, counters=None, env=environment(), timestamp=now())
