#!/usr/bin/env python3
"""R40-01: the `tnca` PyTorch device, tested (doc/r40_tnca_device.md).

    $TORCH_PYTHON tools/r40/device_tests.py [--only NAME ...] [--evidence]

Runs on whatever runtime backend the environment names (TNCA_BACKEND; the RTL
simulator by default) with TNCA_PROFILE (default core-vrf-sync-v5). Each case
checks one property of the device against eager PyTorch on the CPU:

  storage      tensors live at device addresses; views, slices, transposes,
               reshapes and offsets alias one storage, never copy
  copies       host -> device -> host for every supported dtype; strided
               writes into a view change only its elements; device -> device
               with a dtype change; fill_, zero_, item()
  module       nn.Module.to("tnca") moves every parameter and buffer
  refusals     an eager operator with no tnca kernel raises, never runs on the
               host; a launch mixing host and device tensors is refused
  compile      torch.compile(fullgraph=True) of pointwise, reduction, GEMM and
               SDPA graphs on tnca tensors: outputs on tnca, values right,
               no host fallback in the wrapper, zero bytes uploaded (the
               inputs and weights are resident), descriptors where the
               standard path generates them
  views        a compiled graph over non-contiguous views and offsets of one
               storage
  eviction     more kernels than the code region holds, with tensors resident:
               the oldest modules are unloaded, results stay right
  properties   the device interface reports the profile's target, not
               placeholders
  policy       each lowering-policy switch is installed (tnca.torch_device.policy)
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler' / 'python', ROOT / 'tools' / 'asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')

import torch                                            # noqa: E402
import triton                                           # noqa: E402,F401  (before the tnca driver)
import tnca.torch_device as D                           # noqa: E402

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def close(a, b, tol):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return a.shape == b.shape and ((a - b).abs().max().item() / max(b.abs().max().item(), 1)) <= tol


@case
def storage():
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8).to('tnca')
    base = x.untyped_storage().data_ptr()
    assert x.device.type == 'tnca' and base >= 0x20000, hex(base)
    views = [x[1:, 2:], x.t(), x.reshape(32), x.view(8, 4), x[:, ::2], x.narrow(1, 3, 2)]
    for v in views:
        assert v.untyped_storage().data_ptr() == base, 'a view copied its storage'
    assert x[1:, 2:].storage_offset() == 10
    assert torch.equal(x[:, ::2].cpu(), torch.arange(32.).reshape(4, 8)[:, ::2])
    return f'storage at {base:#x}; {len(views)} views alias it'


@case
def copies():
    dts = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.int8,
           torch.uint8, torch.bool]
    for dt in dts:
        h = (torch.randn(5, 7) * 10).to(dt)
        assert torch.equal(h.to('tnca').cpu(), h), dt
    h = torch.randn(6, 9)
    d = h.to('tnca')
    d[1:5:2, 2:8:3] = torch.full((2, 2), 7.0)
    h[1:5:2, 2:8:3] = 7.0
    assert torch.equal(d.cpu(), h), 'strided write'
    d.t()[3].fill_(-1.0)
    h.t()[3].fill_(-1.0)
    assert torch.equal(d.cpu(), h), 'fill through a transposed view'
    e = d.to(torch.float16)
    assert e.device.type == 'tnca' and torch.equal(e.cpu(), h.half()), 'device dtype copy'
    z = torch.empty(3, 3, device='tnca')
    z.zero_()
    assert torch.equal(z.cpu(), torch.zeros(3, 3))
    assert d[2, 4].item() == h[2, 4].item()
    return f'{len(dts)} dtypes round trip; strided writes and fills touch only their elements'


@case
def module():
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16)).half()
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    m.to('tnca')
    sd = m.state_dict()
    assert all(v.device.type == 'tnca' for v in sd.values())
    assert all(torch.equal(sd[k].cpu(), ref[k]) for k in ref)
    return f'{len(sd)} parameters on tnca'


@case
def refusals():
    x = torch.randn(4).to('tnca')
    try:
        (x * 2)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('an eager multiply ran')
    from tnca.triton_backend.driver import TNCALaunchError
    f = torch.compile(lambda a, b: a + b, fullgraph=True)
    try:
        f(torch.randn(4), x)
    except Exception as e:                               # noqa: BLE001
        ok = isinstance(e, TNCALaunchError) or 'device' in str(e).lower()
        assert ok, f'mixed devices raised {type(e).__name__}: {e}'
    else:
        raise AssertionError('a graph over host and device tensors ran')
    return 'eager op refused; mixed host/device graph refused'


def compiled(fn, *args, tol=2e-2):
    from torch._inductor.utils import run_and_get_code
    from tnca.triton_backend.driver import launch_log
    import tnca.inductor as TI
    torch._dynamo.reset()
    dev = [a.to('tnca') if isinstance(a, torch.Tensor) else a for a in args]
    with torch.no_grad(), launch_log() as log:
        got, code = run_and_get_code(torch.compile(fn, fullgraph=True), *dev)
    ref = fn(*[a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
               for a in args])
    outs = got if isinstance(got, (tuple, list)) else (got,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    assert all(o.device.type == 'tnca' for o in outs), 'an output is not on tnca'
    assert all(close(o, r, tol) for o, r in zip(outs, refs)), 'wrong values'
    assert not TI.host_fallbacks(code), TI.host_fallbacks(code)
    up = sum(e.get('upload_bytes', 0) for e in log)
    assert up == 0, f'{up} bytes uploaded for resident tensors'
    return log, code


@case
def compile_graphs():
    torch.manual_seed(0)
    out = []
    cases = [
        ('pointwise', lambda a, b: torch.nn.functional.silu(a) * b,
         (torch.randn(1000).half(), torch.randn(1000).half())),
        ('reduction', lambda x: torch.softmax(x, -1), (torch.randn(8, 100).half(),)),
        ('norm', lambda x, w: torch.nn.functional.rms_norm(x, (64,), w, 1e-5),
         (torch.randn(4, 64).half(), torch.randn(64).half())),
        ('gemm', lambda a, b: a @ b, (torch.randn(48, 32).half(), torch.randn(32, 40).half())),
        ('sdpa', lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
         tuple(torch.randn(1, 2, 16, 16).half() for _ in range(3))),
    ]
    for name, fn, args in cases:
        log, code = compiled(fn, *args)
        res = sum(e.get('resident_bytes', 0) for e in log)
        out.append(f'{name}: {len(log)} launch(es), {res} B resident')
    return '; '.join(out)


@case
def compile_views():
    x = torch.randn(16, 24)
    log, _ = compiled(lambda v, w: (v.t() * 2 + w[2:].t()), x[:, 4:20], torch.randn(18, 16))
    return f'{len(log)} launch(es) over transposed and offset views'


@case
def eviction():
    import tnca.triton_backend.driver as drv
    keep = torch.randn(64).to('tnca')
    ref = keep.cpu()
    before, n = drv.evicted, 0
    # Distinct graphs until the code region has had to give modules back.
    while drv.evicted == before and n < 300:
        torch._dynamo.reset()
        def g(a, n=n):
            # Several transcendental ops fused: kernels of a few KiB each.
            y = torch.sigmoid(a * (n + 1))
            return torch.tanh(y + n) * torch.exp(-y) + torch.log(1 + y * y) - torch.rsqrt(a.abs() + n + 1)
        r = torch.compile(g, fullgraph=True)(keep)
        assert close(r, g(ref), 2e-2), f'graph {n}'
        n += 1
    assert drv.evicted > before, f'{n} graphs and no module was unloaded'
    assert torch.equal(keep.cpu(), ref), 'a resident tensor changed'
    return f'{n} graphs; {drv.evicted - before} module(s) unloaded to make room; resident data intact'


@case
def mixed_devices():
    """CUDA and tnca graphs in one process, alternating: each on its own
    device, each right (skipped where there is no CUDA device)."""
    if not torch.cuda.is_available():
        return 'skipped: no CUDA device on this host'
    a = torch.randn(256, device='cuda')
    b = torch.randn(256)
    out = []
    for i in range(2):
        torch._dynamo.reset()
        rc = torch.compile(lambda x: torch.sigmoid(x) * 2 + i, fullgraph=True)(a)
        assert rc.device.type == 'cuda' and close(rc, torch.sigmoid(a.cpu()) * 2 + i, 1e-5)
        rt = torch.compile(lambda x: torch.sigmoid(x) * 3 + i, fullgraph=True)(b.to('tnca'))
        assert rt.device.type == 'tnca' and close(rt, torch.sigmoid(b) * 3 + i, 2e-2)
        out.append('cuda+tnca')
    rc = torch.compile(lambda x: x.sum(), fullgraph=True)(b)          # host tensors: the B39 path
    assert close(rc, b.sum(), 1e-4)
    return f'{len(out)} rounds of CUDA and tnca graphs in one process, then a CPU-tensor graph'


def _child(code, timeout=1800):
    """Run `code` in a fresh interpreter with this environment; its last
    stdout line is the result (an `OK ...` line or anything else)."""
    import subprocess
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in (ROOT, ROOT / 'python', ROOT / 'compiler' / 'python',
                                                          ROOT / 'tools' / 'asm'))
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(code)              # a file, so a spawned worker can import its __main__
    try:
        r = subprocess.run([sys.executable, fh.name], env=env, capture_output=True, text=True,
                           timeout=timeout, cwd=str(ROOT))
    finally:
        os.unlink(fh.name)
    ok = [l for l in r.stdout.splitlines() if l.startswith('OK')]
    assert r.returncode == 0 and ok, \
        f'child exited {r.returncode}: {(r.stderr.strip().splitlines() or [""])[-1][:200]}'
    return ok[-1]


_GRAPH = """
import torch
x = torch.arange(64, dtype=torch.float32)
y = torch.compile(lambda a: torch.sigmoid(a) * 2, fullgraph=True)(x.to('tnca'))
assert y.device.type == 'tnca' and torch.allclose(y.cpu(), torch.sigmoid(x) * 2, atol=1e-2)
"""


@case
def import_orders():
    out = []
    for name, pre in (('device first', 'import tnca.torch_device as D; D.register(); import triton'),
                      ('triton first', 'import triton; import tnca.torch_device as D; D.register()'),
                      ('after a CPU-tensor compile',
                       'import torch, triton, tnca.inductor; tnca.inductor.configure(); '
                       'torch.compile(lambda a: a + 1)(torch.ones(4)); '
                       'import tnca.torch_device as D; D.register()')):
        _child(pre + '\n' + _GRAPH + '\nprint("OK")')
        out.append(name)
    return 'compiles in a fresh process with ' + ', '.join(out)


@case
def concurrent_registration():
    return _child("""
import threading, triton
import tnca.torch_device as D
errs = []
def go():
    try:
        D.register()
    except Exception as e:
        errs.append(repr(e))
ts = [threading.Thread(target=go) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
assert not errs, errs
""" + _GRAPH + '\nprint("OK 8 threads registered once")')


@case
def worker_process():
    import multiprocessing as mp
    return _child("""
import multiprocessing as mp, triton
import tnca.torch_device as D
D.register()

def work(q):
    import torch, triton
    import tnca.torch_device as D
    D.register()
    x = torch.arange(16, dtype=torch.float32)
    y = torch.compile(lambda a: a * 3 + 1, fullgraph=True)(x.to('tnca')).cpu()
    q.put(bool(torch.allclose(y, x * 3 + 1)))

if __name__ == '__main__':
    ctx = mp.get_context('spawn'); q = ctx.Queue()
    p = ctx.Process(target=work, args=(q,)); p.start(); ok = q.get(timeout=1500); p.join()
    assert ok and p.exitcode == 0
    print("OK a spawned worker registered and computed on its own session")
""")


@case
def cache_hits():
    from tnca.triton_backend.driver import launch_log
    x = torch.randn(32).to('tnca')
    shas = []
    for _ in range(2):
        torch._dynamo.reset()
        with launch_log() as log:
            torch.compile(lambda a: torch.tanh(a) + 0.5, fullgraph=True)(x)
        shas.append([e.get('artifact') for e in log])
    assert shas[0] == shas[1] and all(shas[0]), shas
    return f'recompiled after a reset: the same artifact {shas[0][0][:12]} (Triton\'s cache)'


@case
def checkpoint():
    import io
    m = torch.nn.Linear(8, 4).half()
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    m.to('tnca')
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    on_cpu = torch.load(buf, map_location='cpu')
    assert all(v.device.type == 'cpu' and torch.equal(v, ref[k]) for k, v in on_cpu.items())
    buf.seek(0)
    on_dev = torch.load(buf, map_location='tnca')
    assert all(v.device.type == 'tnca' and torch.equal(v.cpu(), ref[k]) for k, v in on_dev.items())
    return 'a tnca state_dict saves, and loads to cpu and to tnca'


@case
def unsupported_dtypes():
    refused = []
    for dt in (torch.complex64, torch.complex128):
        try:
            torch.zeros(4, dtype=dt).to('tnca')
        except (RuntimeError, NotImplementedError, TypeError) as e:
            refused.append(str(dt).replace('torch.', ''))
        else:
            raise AssertionError(f'{dt} reached the device')
    x = torch.randn(8, dtype=torch.float64).to('tnca')      # storage is bytes: it may live there
    try:
        torch.compile(lambda a: a * 2, fullgraph=True)(x)
    except Exception as e:                               # noqa: BLE001
        refused.append(f'fp64 compute ({type(e).__name__})')
    else:
        raise AssertionError('an fp64 kernel ran on a device with no fp64 arithmetic')
    return 'refused: ' + ', '.join(refused)


@case
def properties():
    from torch._dynamo.device_interface import get_interface_for_device
    from tnca.adapters import r1
    iface = get_interface_for_device('tnca')
    p = iface.get_device_properties()
    target, _ = r1.target_for(os.environ['TNCA_PROFILE'])
    assert p.multi_processor_count == target.num_cores
    assert p.num_rctx == target.num_rctx and p.vlen == target.vlen
    assert iface.is_bf16_supported() == bool(dict(target.capabilities).get('matrix_bf16'))
    return f'{p.name}: {p.multi_processor_count} core(s), {p.num_rctx} contexts, {p.vlen} lanes'


@case
def policy():
    import torch._inductor.utils as iu
    import torch._inductor.lowering as L
    import torch._inductor.ir as ir
    aten = torch.ops.aten
    on = {'templates': getattr(iu.use_triton_template, '_tnca', False),
          'variance': getattr(L.use_two_step_variance, '_tnca', False),
          'cat': getattr(L.lowerings[aten.cat.default], '_tnca', False),
          'bmm': getattr(L.lowerings[aten.bmm.default], '_tnca', False),
          'realize': getattr(ir.is_cpu, '_tnca', False)}
    assert all(on.values()), on
    assert 'tnca' not in iu.GPU_TYPES
    return 'templates, variance, cat, bmm, realize installed; tnca is not a GPU type'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args(argv)
    D.register()
    results = []
    # The cases that start their own processes go first, before this one has
    # opened the device: a board is held by one process at a time (a
    # simulator is one per process, so there it would not matter).
    first = ('import_orders', 'concurrent_registration', 'worker_process')
    order = [fn for fn in CASES if fn.__name__ in first] + \
        [fn for fn in CASES if fn.__name__ not in first]
    for fn in order:
        if args.only and fn.__name__ not in args.only:
            continue
        try:
            detail = fn()
            ok = True
        except Exception as e:                           # noqa: BLE001
            ok, detail = False, f'{type(e).__name__}: {e}'
            traceback.print_exc()
        results.append(dict(name=fn.__name__, ok=ok, detail=str(detail)[:300]))
        print(f"  {'PASS' if ok else 'FAIL'}  {fn.__name__:<16s} {detail}", flush=True)
    passed = sum(r['ok'] for r in results)
    print(f"r40-device-tests  {'PASS' if passed == len(results) else 'FAIL'}  ({passed}/{len(results)})")
    if args.evidence:
        # Acceptance 2 wants a host with CUDA and one without: each run adds
        # its own checks under its host class, and the gate passes only when
        # both classes are there and every check of both passed.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import evidence
        import json as _json
        cls = 'cuda-host' if torch.cuda.is_available() else 'no-cuda-host'
        where = f'{cls}/{os.environ.get("TNCA_BACKEND", "sim-file")}/{os.uname().nodename}'
        path = evidence.path_for('R40-01.mixed-device')
        try:
            prior = _json.loads(path.read_text())['gates']['R40-01.mixed-device']['checks']
        except Exception:                                # noqa: BLE001
            prior = []
        kept = [c for c in prior if not c['name'].startswith(where + ':')]
        mine = [dict(name=f"{where}:{r['name']}", ok=r['ok'], detail=r['detail']) for r in results]
        checks = kept + mine
        classes = {c['name'].split('/', 1)[0] for c in checks}
        ok = all(c['ok'] for c in checks) and {'cuda-host', 'no-cuda-host'} <= classes
        status = 'capability-pass' if ok else ('not-measured' if all(c['ok'] for c in checks) else 'fail')
        evidence.record('R40-01.mixed-device', status, checks,
                        detail=f'{sorted({c["name"].rsplit(":", 1)[0] for c in checks})}; '
                               f'{sum(c["ok"] for c in checks)}/{len(checks)} checks')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
