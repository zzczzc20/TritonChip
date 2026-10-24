#!/usr/bin/env python3
"""FP8 as a generic Torch dtype on `tnca`: casts, tensor plumbing, _scaled_mm, linears.

    $TORCH_PYTHON tools/performance_harness/fp8_cases.py --out DIR \
        [--group casts|plumbing|scaled_mm|linear ...] [--quick]

Every graph is `torch.compile(backend="inductor", fullgraph=True)` of plain
Torch code on `tnca` tensors: Inductor, Triton, the TNCA backend, a sealed
artifact and the Session, executing on TNCA_BACKEND (default `systemc-tlm-grid`,
the native SystemC/TLM model with physical-grid launches; `systemc-tlm` is its
per-program form, and `sim-file` and `fpga-multicore-sim` are the RTL
simulators). The CPU computes every reference. The graphs are the generic
patterns quantisation code is written in; nothing here names a model, a
layer or a checkpoint.

groups
  casts      fp32/bf16/fp16 <-> float8_e4m3fn/float8_e5m2, bit for bit
             against torch CPU's own conversion: every 8-bit and every 16-bit
             pattern, and a structured binary32 set (every FP8 value, every
             midpoint between neighbours and the binary32 values either side
             of it, the overflow and subnormal boundaries, signed zeros,
             infinities, NaNs with payloads, random bit patterns over every
             exponent); 1-D and 2-D (transposed-read) kernels; eager device
             copies with a dtype change
  plumbing   host<->tnca copies, device views and slices, compiled slicing
             and cat, clamp-then-cast quantisation with static, rowwise and
             amax-derived scales, scale-multiply dequantisation
  scaled_mm  torch._scaled_mm through Inductor's own lowering
             (kernel/mm.py tuned_scaled_mm, the mm template with its
             scaling epilogue): tensorwise and rowwise scales, bf16/fp16/fp32
             out, with and without bias, e4m3, e5m2 and e4m3 x e5m2, M/N/K
             tails, under any TNCA_GEMM_TILE. Checked
             bit for bit against an independent per-K binary32 FMA reference
             in the MMA's order (ISA sec 9C.1) and compared with torch CPU
             _scaled_mm; the FP8 MMA is read from the compiled assembly and
             its dynamic count from the model's counters
  linear     a static W8A8 linear (quantise with a static input scale,
             _scaled_mm with weight and input scales, bias, bf16 out) and a
             weight-only FP8 linear (upcast and scale the weight, bf16 GEMM)

A comparison reports raw bits. Where both sides are NaN with different
encodings it is counted apart from value mismatches: TNCA arithmetic
returns the canonical NaN (ISA sec 8.2), so a NaN that went through an
add, multiply, divide, min or max has the device's encoding, not the
host's propagated payload (evidence/r41/fp8/README.md).

Development evidence: model results are model predictions and RTL
simulation results are simulations; neither is a board record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
                str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]

GROUPS = ('casts', 'plumbing', 'scaled_mm', 'linear')


# ------------------------------------------------------------ comparison ----
def _torch():
    import torch
    return torch


def bits(t):
    torch = _torch()
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(width)


def _hex(v, size):
    return f'0x{int(v) & ((1 << (8 * size)) - 1):0{2 * size}x}'


def compare(got, ref, inputs=None):
    """Raw-bit comparison; a NaN on both sides with different bits is a NaN
    encoding difference, anything else a value mismatch."""
    torch = _torch()
    got = got.detach().cpu()
    if got.dtype != ref.dtype or tuple(got.shape) != tuple(ref.shape):
        return dict(equal=False, error=f'{got.dtype}{tuple(got.shape)} vs {ref.dtype}{tuple(ref.shape)}')
    g, r = bits(got).flatten(), bits(ref).flatten()
    differ = g != r
    both_nan = got.float().isnan().flatten() & ref.float().isnan().flatten()
    value = differ & ~both_nan
    size = got.element_size()
    examples = []
    for i in value.nonzero().flatten()[:8].tolist():
        row = dict(index=i, got=_hex(g[i], size), reference=_hex(r[i], size))
        if inputs is not None:
            x = inputs.flatten()
            row['input'] = _hex(bits(x).flatten()[i], x.element_size())
        examples.append(row)
    nan_examples = [dict(index=i, got=_hex(g[i], size), reference=_hex(r[i], size))
                    for i in (differ & both_nan).nonzero().flatten()[:4].tolist()]
    out = dict(elements=ref.numel(), bit_mismatches=int(differ.sum()),
               value_mismatches=int(value.sum()), nan_encoding_mismatches=int((differ & both_nan).sum()),
               equal=not bool(differ.any()), examples=examples, nan_examples=nan_examples)
    if got.element_size() == 1 and got.is_floating_point() and bool(value.any()):
        out['max_ulp_diff'] = ulp_diff(got, ref)
    if got.dtype in (torch.float32, torch.float16, torch.bfloat16) and bool(value.any()):
        finite = ~(got.float().isnan() | ref.float().isnan())
        out['max_abs_diff'] = float((got.float() - ref.float()).abs()[finite].max())
        out['max_abs_reference'] = float(ref.float().abs()[finite].max())
        out['max_ulp_diff'] = ulp_diff(got, ref)
    return out


def ulp_diff(got, ref):
    """Largest distance in units in the last place, over non-NaN pairs
    (+0 and -0 are the same point)."""
    torch = _torch()
    def ordered(t):
        b = bits(t).to(torch.int64)
        top = 1 << (8 * t.element_size() - 1)
        b = torch.where(b < 0, b + 2 * top, b)
        magnitude = b & (top - 1)
        return torch.where(b >= top, -magnitude, magnitude)
    ok = ~(got.float().isnan() | ref.float().isnan())
    if not bool(ok.any()):
        return 0
    return int((ordered(got) - ordered(ref)).abs()[ok].max())


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(_torch().uint8).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------ inputs --------
def f32_structured(quick=False, seed=0):
    """A deterministic binary32 set aimed at every conversion corner."""
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    pats = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
            0x7F800001, 0xFF800001, 0x7FBFFFFF, 0xFFBFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
            0x7FA00000, 0xFFE00001, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
            0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF]
    parts = [torch.tensor(pats, dtype=torch.int64).to(torch.int32).view(torch.float32)]
    inf = torch.tensor(float('inf'))
    for fmt in (torch.float8_e4m3fn, torch.float8_e5m2):
        v = torch.arange(256, dtype=torch.uint8).view(fmt).float()
        p = torch.unique(v[(~v.isnan()) & (~v.isinf()) & (v >= 0)])     # sorted, finite, with 0
        top = p[-1]
        step = top - p[-2]
        # every value, every midpoint, and the region past the largest
        pts = torch.cat([p, (p[1:] + p[:-1]) / 2,
                         torch.stack([top + step / 2, top + step, 2 * top, p[1] / 2, p[1] / 4, p[1] * 0.75])])
        pts = torch.cat([pts, torch.nextafter(pts, inf), torch.nextafter(pts, -inf)])
        parts += [pts, -pts]
    e = torch.arange(-149, 128, dtype=torch.float64)
    for m in (1.0, 1.0 + 2.0 ** -23, 1.125, 1.5, 1.875, 2.0 - 2.0 ** -23):
        x = (m * torch.pow(2.0, e)).clamp(max=float(torch.finfo(torch.float32).max)).float()
        parts += [x, -x]
    n = 1 << (12 if quick else 17)
    parts.append(torch.randint(-2 ** 31, 2 ** 31, (n,), generator=g, dtype=torch.int64)
                 .to(torch.int32).view(torch.float32))
    scale = torch.pow(10.0, torch.empty(n // 4).uniform_(-8, 6, generator=g))
    parts.append((torch.randn(n // 4, generator=g) * scale).float())
    return torch.cat(parts)


def all16(dtype, quick=False):
    torch = _torch()
    x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    if quick:
        keep = torch.zeros(65536, dtype=torch.bool)
        keep[::61] = True
        b = torch.arange(65536)
        e = (b >> (7 if dtype == torch.bfloat16 else 10)) & (0xFF if dtype == torch.bfloat16 else 0x1F)
        keep |= e >= (0xFF if dtype == torch.bfloat16 else 0x1F)      # every inf/NaN
        keep |= e == 0                                                  # every subnormal
        x = x[keep]
    return x


def all8(dtype, repeat=4):
    torch = _torch()
    codes = torch.arange(256, dtype=torch.uint8)
    # repeated with an odd tail, so a masked last block is exercised
    return torch.cat([codes.repeat(repeat), codes[:3]]).view(dtype)


# ------------------------------------------------------------ the run -------
class Run:
    def __init__(self, out, quick, backend):
        self.out, self.quick, self.backend = out, quick, backend
        self.rows = []
        self.report = None
        self.models = None

    def counters(self):
        if not self.models:
            return {}
        snap = {}
        for i, model in enumerate(self.models.instances):
            if model.proc is not None:
                snap[str(i)] = model.counters_snapshot()
        return snap

    def compile_run(self, name, fn, args, *, dump=False, keep_on_host=()):
        """Compile `fn` on tnca copies of `args`, run it twice (cold, then
        steady under the launch log), and return the steady outputs on the
        host with the graph's record."""
        torch = _torch()
        from torch._inductor.utils import run_and_get_code
        from tnca.inductor import host_fallbacks, triton_kernels
        from tnca.triton_backend.driver import launch_log
        torch._dynamo.reset()
        dump_dir = self.out/'dump'/re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
        if dump:
            os.environ['TNCA_KERNEL_DUMP'] = '1'
            os.environ['TNCA_DUMP_DIR'] = str(dump_dir)
        try:
            # keep_on_host: argument positions the caller places on the CPU
            # (host metadata), passed as they are.
            dargs = [a if i in keep_on_host else to_device(a) for i, a in enumerate(args)]
            compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
            t0 = time.monotonic()
            cold, code = run_and_get_code(compiled, *dargs)
            compile_s = time.monotonic() - t0
        finally:
            os.environ.pop('TNCA_KERNEL_DUMP', None)
            os.environ.pop('TNCA_DUMP_DIR', None)
        before = self.counters()
        t0 = time.monotonic()
        with launch_log(isolated=True) as launches:
            steady = compiled(*dargs)
        run_s = time.monotonic() - t0
        after = self.counters()
        outs = steady if isinstance(steady, tuple) else (steady,)
        colds = cold if isinstance(cold, tuple) else (cold,)
        host = tuple(o.cpu() for o in outs)
        dyn = {}
        for e in launches:
            for k, v in (e.get('counters') or {}).items():
                if 'mma' in k and isinstance(v, (int, float)):
                    dyn[k] = dyn.get(k, 0) + v
        record = dict(kernels=triton_kernels(code), host_fallbacks=host_fallbacks(code),
                      launch_mma_counters=dyn,
                      launches=[dict(kernel=e['kernel'], grid=e.get('grid'),
                                     device_cycles=e.get('device_cycles'),
                                     upload_bytes=e.get('upload_bytes'),
                                     download_bytes=e.get('download_bytes'))
                                for e in launches],
                      outputs_on_device=all(o.device.type == 'tnca' for o in outs),
                      cold_equals_steady=all(torch.equal(bits(c.cpu()), bits(h)) for c, h in zip(colds, host)),
                      compile_seconds=round(compile_s, 3), steady_seconds=round(run_s, 3))
        if before or after:
            record['mma_commands'] = _sum_key(after, 'mma_commands') - _sum_key(before, 'mma_commands')
        if dump:
            record['mma'] = mma_census(dump_dir)
        (self.out/'code').mkdir(exist_ok=True)
        (self.out/'code'/(re.sub(r'[^A-Za-z0-9_.-]+', '_', name) + '.py')).write_text('\n\n'.join(code))
        return host, record

    def add(self, group, name, status, **row):
        row = dict(group=group, case=name, status=status, **row)
        self.rows.append(row)
        flag = {'pass': 'PASS', 'fail': 'FAIL', 'error': 'ERROR', 'refused': 'REFUSED'}.get(status, status)
        detail = row.get('summary', '')
        print(f'{flag:7s} {group:9s} {name:58s} {detail}', flush=True)
        self.save()

    def save(self):
        if self.report is not None:
            self.report['rows'] = self.rows
            self.report['summary'] = {s: sum(r['status'] == s for r in self.rows)
                                      for s in ('pass', 'fail', 'error', 'refused')}
            (self.out/'report.json').write_text(json.dumps(self.report, indent=1, default=str) + '\n')


def to_device(a):
    torch = _torch()
    if not isinstance(a, torch.Tensor):
        return a
    if a.dim() == 2 and not a.is_contiguous() and a.t().is_contiguous():
        # keep a column-major operand column-major on the device
        return a.t().contiguous().to('tnca').t()
    return a.to('tnca')


def _sum_key(tree, key):
    """Sum every counter named `key` or `core_<i>_<key>` (the model's
    per-core counters)."""
    if isinstance(tree, dict):
        return sum((v if (k == key or re.fullmatch(rf'core_\d+_{key}', k)) and isinstance(v, (int, float))
                    else _sum_key(v, key)) for k, v in tree.items())
    if isinstance(tree, list):
        return sum(_sum_key(v, key) for v in tree)
    return 0


_MMA_RE = re.compile(r'^\s*(?:@\S+\s+)?(m\.mma\.async(?:\.\w+)*)\b', re.M)


def mma_census(dump_dir):
    """Every m.mma.async in the compiled assembly, by mnemonic."""
    census = {}
    for s in sorted(Path(dump_dir).glob('*/*.s')):
        for m in _MMA_RE.finditer(s.read_text()):
            key = f'{s.parent.name}: {m.group(1)}'
            census[key] = census.get(key, 0) + 1
    return census


def judge(cmp, *, nan_ok=False):
    """pass when bit-equal (or, where noted, equal but for NaN encodings)."""
    if cmp.get('error'):
        return 'fail'
    if cmp['value_mismatches']:
        return 'fail'
    if cmp['nan_encoding_mismatches'] and not nan_ok:
        return 'fail'
    return 'pass'


def summary_of(cmp):
    if cmp.get('error'):
        return cmp['error']
    s = f"{cmp['elements']} elements, {cmp['bit_mismatches']} bit mismatches"
    if cmp['nan_encoding_mismatches']:
        s += f" ({cmp['nan_encoding_mismatches']} NaN-encoding only)"
    return s


# ------------------------------------------------------------ casts ---------
def group_casts(run):
    torch = _torch()
    E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
    F8 = (E4, E5)
    wide = (torch.float32, torch.bfloat16, torch.float16)
    tag = {torch.float32: 'f32', torch.bfloat16: 'bf16', torch.float16: 'f16', E4: 'e4m3', E5: 'e5m2'}
    s32 = f32_structured(run.quick)
    sources = {torch.float32: s32, torch.bfloat16: all16(torch.bfloat16, run.quick),
               torch.float16: all16(torch.float16, run.quick)}

    chunk = 1 << 16          # 4096 programs of 16 lanes: a per-program backend's launch limit

    def cast_case(name, fn, x, ref, extra=None):
        try:
            if x.numel() > chunk and x.dim() == 1:
                # One compiled graph over equal chunks (zero padded), one launch each.
                n = -(-x.numel() // chunk)
                padded = torch.cat([x, torch.zeros(n * chunk - x.numel(), dtype=x.dtype)]).reshape(n, chunk)
                (y0,), rec = run.compile_run(f'casts/{name}', fn, (padded[0],))
                device_fn = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
                y = torch.cat([y0] + [device_fn(padded[i].to('tnca')).cpu() for i in range(1, n)])[:x.numel()]
                rec['chunks'] = n
            else:
                (y,), rec = run.compile_run(f'casts/{name}', fn, (x,))
            cmp = compare(y, ref, x)
            run.add('casts', name, judge(cmp), summary=summary_of(cmp), comparison=cmp,
                    graph=rec, input_sha256=sha(x), output_sha256=sha(y), **(extra or {}))
        except Exception as e:                               # noqa: BLE001
            run.add('casts', name, 'error', summary=f'{type(e).__name__}: {str(e)[:300]}',
                    traceback=traceback.format_exc()[-4000:])

    for src in wide:
        for dst in F8:
            x = sources[src]
            cast_case(f'{tag[src]}->{tag[dst]} 1-D ({"exhaustive" if src != torch.float32 and not run.quick else "structured"})',
                      lambda x, dst=dst: x.to(dst), x, x.to(dst))
    for src in F8:
        for dst in wide:
            x = all8(src)
            cast_case(f'{tag[src]}->{tag[dst]} 1-D (all 256 patterns)', lambda x, dst=dst: x.to(dst), x, x.to(dst))
    # 2-D kernels: a transposed read makes Inductor tile the pointwise loop.
    cols = 96
    for src, dst in ((torch.float32, E4), (torch.float32, E5), (torch.float16, E4), (torch.bfloat16, E5)):
        rows = 16 if run.quick else 64
        x = sources[src][:rows * cols].reshape(rows, cols)
        cast_case(f'{tag[src]}->{tag[dst]} 2-D transposed read {tuple(x.shape)}',
                  lambda x, dst=dst: x.t().contiguous().to(dst), x, x.t().contiguous().to(dst))
    for src, dst in ((E4, torch.float32), (E5, torch.bfloat16), (E4, torch.float16)):
        x = torch.arange(256, dtype=torch.uint8).repeat(12).reshape(32, 96).view(src)
        cast_case(f'{tag[src]}->{tag[dst]} 2-D transposed read {tuple(x.shape)}',
                  lambda x, dst=dst: x.t().contiguous().to(dst), x, x.t().contiguous().to(dst))
    # Eager casts on the device: Tensor.to on a tnca tensor is a device copy
    # kernel (tnca.torch_device.copy), compiled by the same compiler.
    # (A strided device copy computes each element's offsets with integer
    # division in the kernel; inputs stay small enough for the launch deadline,
    # and for an RTL simulator's host wait with --quick.)
    n = 1024 if run.quick else 16384
    for src, dst, x in ((torch.float32, E4, s32[:n]), (torch.float32, E5, s32[:n]),
                        (torch.bfloat16, E4, sources[torch.bfloat16][::5][:n]), (E4, torch.float32, all8(E4)),
                        (E5, torch.float16, all8(E5)), (E4, torch.bfloat16, all8(E4))):
        name = f'eager device copy {tag[src]}->{tag[dst]}'
        try:
            d = x.to('tnca')
            y = d.to(dst).cpu()
            m = x[:(x.numel() // 8) * 8].reshape(8, -1)
            view = m.to('tnca').t()[1:, 2:]
            yv = view.to(dst).cpu()
            c1, c2 = compare(y, x.to(dst), x), compare(yv, m.t()[1:, 2:].to(dst))
            ok = judge(c1) == judge(c2) == 'pass'
            run.add('casts', name, 'pass' if ok else 'fail',
                    summary=f'contiguous: {summary_of(c1)}; strided view: {summary_of(c2)}',
                    comparison=c1, strided_comparison=c2)
        except Exception as e:                               # noqa: BLE001
            run.add('casts', name, 'error', summary=f'{type(e).__name__}: {str(e)[:300]}',
                    traceback=traceback.format_exc()[-4000:])
    # e4m3 <-> e5m2: Inductor's own Triton codegen refuses a direct FP8->FP8
    # cast on every device; tnca lowers it through binary32 (exact).
    for src, dst in ((E4, E5), (E5, E4)):
        x = all8(src)
        cast_case(f'{tag[src]}->{tag[dst]} 1-D (all 256 patterns)', lambda x, dst=dst: x.to(dst), x, x.to(dst))


# ------------------------------------------------------------ plumbing ------
def group_plumbing(run):
    torch = _torch()
    E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
    g = torch.Generator().manual_seed(7)

    def record(name, fn, args, ref, *, nan_ok=False, refs_extra=None):
        try:
            outs, rec = run.compile_run(f'plumbing/{name}', fn, args)
            refs = ref if isinstance(ref, tuple) else (ref,)
            cmps = [compare(o, r) for o, r in zip(outs, refs)]
            status = 'pass' if all(judge(c, nan_ok=nan_ok) == 'pass' for c in cmps) else 'fail'
            row = dict(summary='; '.join(summary_of(c) for c in cmps), comparisons=cmps, graph=rec)
            if refs_extra:
                for label, eref in refs_extra.items():
                    erefs = eref if isinstance(eref, tuple) else (eref,)
                    row[f'versus_{label}'] = [compare(o, r) for o, r in zip(outs, erefs)]
                    row['summary'] += f' | vs {label}: ' + '; '.join(summary_of(c) for c in row[f'versus_{label}'])
            run.add('plumbing', name, status, **row)
        except Exception as e:                               # noqa: BLE001
            run.add('plumbing', name, 'error', summary=f'{type(e).__name__}: {str(e)[:300]}',
                    traceback=traceback.format_exc()[-4000:])

    # host <-> device copies and device views: bytes, not values
    for fmt in (E4, E5):
        tag = 'e4m3' if fmt == E4 else 'e5m2'
        codes = torch.randint(0, 256, (37, 41), generator=g, dtype=torch.uint8)
        codes.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
        x = codes.view(fmt)
        try:
            d = x.to('tnca')
            checks = {
                'round trip': (d.cpu(), x),
                'transpose view': (d.t().cpu(), x.t()),
                'slice view': (d[3:30, 5:17].cpu(), x[3:30, 5:17]),
                'strided view': (d.reshape(-1)[17:900:3].cpu(), x.reshape(-1)[17:900:3]),
                'as_strided offset': (d.as_strided((5, 6), (41, 2), 50).cpu(), x.as_strided((5, 6), (41, 2), 50)),
                'uint8 reinterpret': (d.view(torch.uint8).cpu(), codes),
                'device copy_ into strided': (torch.empty(41, 37, dtype=fmt, device='tnca').t().copy_(d).cpu(), x),
                'empty().copy_ from host': (torch.empty(37, 41, dtype=fmt, device='tnca').copy_(x).cpu(), x),
                'clone on device': (d.t().clone().cpu(), x.t()),
            }
            cmps = {k: compare(a, b) for k, (a, b) in checks.items()}
            bad = [k for k, c in cmps.items() if not c['equal']]
            run.add('plumbing', f'host/device copies and views ({tag})', 'fail' if bad else 'pass',
                    summary=f'{len(cmps)} checks, mismatching: {bad or "none"}', comparisons=cmps)
        except Exception as e:                               # noqa: BLE001
            run.add('plumbing', f'host/device copies and views ({tag})', 'error',
                    summary=f'{type(e).__name__}: {str(e)[:300]}', traceback=traceback.format_exc()[-4000:])

    w = torch.randint(0, 256, (48, 80), generator=g, dtype=torch.uint8)
    w.view(-1)[:4] = torch.tensor([0x7F, 0xFF, 0x00, 0x80], dtype=torch.uint8)
    w4 = w.view(E4)
    w5 = torch.where(w & 0x7C == 0x7C, w & 0x83, w).view(E5)            # finite e5m2, sign kept
    record('compiled slice + cat dim 0 (e4m3)', lambda a, b: torch.cat([a[:, :8], b[3:20, 5:13]], dim=0),
           (w4, w4), torch.cat([w4[:, :8], w4[3:20, 5:13]], dim=0))
    record('compiled cat dim 1 (e5m2)', lambda a, b: torch.cat([a, b], dim=1), (w5, w5[:, :7]),
           torch.cat([w5, w5[:, :7]], dim=1))
    record('compiled narrow + transpose + contiguous (e4m3)', lambda a: a[5:40:2, 3:].t().contiguous(),
           (w4,), w4[5:40:2, 3:].t().contiguous())

    x = torch.randn(64, 96, generator=g) * 40
    x.view(-1)[:6] = torch.tensor([float('nan'), -float('nan'), float('inf'), -float('inf'), 1e6, -0.0])
    s = torch.tensor(0.037)
    q4 = lambda x, s: (x / s).clamp(-448.0, 448.0).to(E4)
    q5 = lambda x, s: (x / s).clamp(-57344.0, 57344.0).to(E5)
    # NaN through x / s and clamp is an arithmetic NaN: canonical on TNCA.
    record('clamp-then-cast quantise f32, static scale (e4m3)', q4, (x, s), q4(x, s), nan_ok=True)
    record('clamp-then-cast quantise f32, static scale (e5m2)', q5, (x, s), q5(x, s), nan_ok=True)
    xb = x.to(torch.bfloat16)
    record('clamp-then-cast quantise bf16, static scale (e4m3)', q4, (xb, s),
           (xb.float() / s).clamp(-448.0, 448.0).to(E4), nan_ok=True,
           refs_extra={'eager_bf16_intermediates': q4(xb, s)})
    rs = torch.rand(64, 1, generator=g) + 0.01
    record('clamp-then-cast quantise f32, rowwise scale (e4m3)', q4, (x, rs), q4(x, rs), nan_ok=True)
    xd = torch.randn(64, 96, generator=g) * 3
    dq = lambda x: (x / (x.abs().amax() / 448.0)).clamp(-448.0, 448.0).to(E4)
    record('amax-derived scale, then quantise (e4m3)', dq, (xd,), dq(xd))
    ws = torch.tensor(0.0123)
    record('dequantise w.to(bf16) * s (e4m3)', lambda w, s: w.to(torch.bfloat16) * s, (w4, ws),
           w4.to(torch.bfloat16) * ws, nan_ok=True)
    record('dequantise w.float() * s (e4m3)', lambda w, s: w.float() * s, (w4, ws), w4.float() * ws, nan_ok=True)
    record('dequantise w.to(fp16) * s (e5m2)', lambda w, s: w.to(torch.float16) * s, (w5, ws),
           w5.to(torch.float16) * ws, nan_ok=True)
    cs = torch.rand(48, 1, generator=g) + 0.5
    record('dequantise per-channel w.to(bf16) * s[:, None] (e4m3)', lambda w, s: w.to(torch.bfloat16) * s,
           (w4, cs), w4.to(torch.bfloat16) * cs, nan_ok=True)


# ------------------------------------------------------------ scaled_mm -----
def per_k_fma(A, B):
    """binary32 accumulator from +0.0, one FMA per K index in ascending
    order (ISA sec 9C.1). FP8 x FP8 and BF16 x BF16 products are exact in
    binary32 (checked), so each add of a product rounds as an FMA does."""
    torch = _torch()
    A, B = A.float(), B.float()
    acc = torch.zeros(A.shape[0], B.shape[1], dtype=torch.float32)
    A64, B64 = A.double(), B.double()
    for k in range(A.shape[1]):
        p = A[:, k:k+1] * B[k:k+1, :]
        if not torch.equal(p.double(), A64[:, k:k+1] * B64[k:k+1, :]):
            raise ValueError(f'product at k={k} is not exact in binary32')
        acc = acc + p
    return acc


def scaled_reference(a, b, sa, sb, bias, out_dtype, acc=None, scale_result=None):
    """Inductor's _scaled_mm semantics in the MMA's order: per-K FMA of the
    exactly dequantised operands, then the scaling epilogue
    (kernel/mm_common.scale_mm_epilogue: acc * (scale_a * scale_b), + bias)
    in binary32, rounded once to out_dtype. An FP8 output with a
    scale_result is first multiplied by its binary32 reciprocal, as torch
    does (ATen _scaled_mm_out_cpu_emulated)."""
    torch = _torch()
    acc = per_k_fma(a, b) if acc is None else acc
    scale = sa.float().reshape(-1, 1) * sb.float().reshape(1, -1) if sa.numel() > 1 or sb.numel() > 1 \
        else (sa.float() * sb.float())
    out = acc * scale
    if bias is not None:
        out = out + bias.float()
    if scale_result is not None and out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out = out * torch.reciprocal(scale_result.float())
    return out.to(out_dtype)


def order_sensitivity(a, b, exact_acc):
    torch = _torch()
    A, B = a.float(), b.float()
    K = A.shape[1]
    acc = torch.zeros_like(exact_acc)
    for start in range(0, K, 16):
        blk = torch.zeros_like(acc)
        for k in range(start, min(K, start + 16)):
            blk = blk + A[:, k:k+1] * B[k:k+1, :]
        acc = acc + blk
    rev = torch.zeros_like(exact_acc)
    for k in reversed(range(K)):
        rev = rev + A[:, k:k+1] * B[k:k+1, :]
    return dict(blocked16_acc_mismatches=int((bits(acc) != bits(exact_acc)).sum()),
                reversed_acc_mismatches=int((bits(rev) != bits(exact_acc)).sum()))


def fp8_operands(M, N, K, fmt, g):
    """FP8 operands over several binades, so per-K partial sums round."""
    torch = _torch()
    mag = lambda *s: torch.pow(2.0, torch.randint(-3, 4, s, generator=g).float())
    a = (torch.randn(M, K, generator=g) * mag(M, K)).to(fmt)
    bt = (torch.randn(N, K, generator=g) * mag(N, K)).to(fmt)
    return a, bt.t()                       # a row-major, b column-major (torch._scaled_mm's layout)


def group_scaled_mm(run):
    torch = _torch()
    E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
    g = torch.Generator().manual_seed(11)
    tag = {torch.bfloat16: 'bf16', torch.float16: 'f16', torch.float32: 'f32', E4: 'e4m3', E5: 'e5m2'}
    if run.quick:
        matrix = [((16, 16, 32), E4, 'tensorwise', torch.bfloat16, True),
                  ((17, 24, 40), E5, 'rowwise', torch.float32, False),
                  ((5, 16, 16), E4, 'tensorwise', torch.float16, False)]
    else:
        matrix = []
        for fmt in (E4, E5):
            for scaling in ('tensorwise', 'rowwise'):
                for out_dtype in (torch.bfloat16, torch.float16, torch.float32):
                    for bias in (False, True):
                        matrix.append(((32, 48, 64), fmt, scaling, out_dtype, bias))
        for shape in ((16, 16, 16), (5, 16, 16), (16, 32, 8), (1, 64, 64), (17, 24, 40), (33, 48, 80),
                      (64, 96, 128), (40, 72, 200), (128, 192, 384)):
            for scaling in ('tensorwise', 'rowwise'):
                matrix.append((shape, E4, scaling, torch.bfloat16, True))
        matrix.append(((33, 40, 48), E5, 'rowwise', torch.float16, True))
        # use_fast_accum (the template's `acc = tl.dot(a, b, acc)` form), and
        # FP8 outputs with and without scale_result.
        matrix += [((32, 48, 64), E4, 'tensorwise', torch.bfloat16, True, dict(use_fast_accum=True)),
                   ((32, 48, 64), E4, 'tensorwise', E4, False, {}),
                   ((32, 48, 64), E4, 'tensorwise', E4, False, dict(scale_result=0.37)),
                   ((33, 40, 48), E4, 'rowwise', E5, False, dict(scale_result=4.0)),
                   ((17, 24, 40), E5, 'tensorwise', E5, False, dict(scale_result=0.125))]
    for (M, N, K), fmt, scaling, out_dtype, with_bias, *opt in matrix:
        opt = opt[0] if opt else {}
        name = (f'{tag[fmt]} {scaling} {M}x{N}x{K} -> {tag[out_dtype]}{" + bias" if with_bias else ""}'
                + ''.join(f' {k}={v}' for k, v in opt.items()))
        try:
            a, b = fp8_operands(M, N, K, fmt, g)
            if scaling == 'tensorwise':
                sa, sb = torch.tensor(0.0371), torch.tensor(1.83)
            else:
                sa = torch.rand(M, 1, generator=g) * 0.1 + 0.01
                sb = torch.rand(1, N, generator=g) * 2 + 0.5
            bias = (torch.randn(N, generator=g) * 4).to(out_dtype) if with_bias else None
            sr = torch.tensor(opt['scale_result']) if 'scale_result' in opt else None
            fast = opt.get('use_fast_accum', False)
            args = (a, b, sa, sb) + ((bias,) if with_bias else ()) + ((sr,) if sr is not None else ())
            if with_bias:
                fn = lambda a, b, sa, sb, c, o=out_dtype, f=fast: torch._scaled_mm(
                    a, b, sa, sb, bias=c, out_dtype=o, use_fast_accum=f)
            elif sr is not None:
                fn = lambda a, b, sa, sb, r, o=out_dtype: torch._scaled_mm(a, b, sa, sb, scale_result=r, out_dtype=o)
            else:
                fn = lambda a, b, sa, sb, o=out_dtype: torch._scaled_mm(a, b, sa, sb, out_dtype=o)
            (y,), rec = run.compile_run(f'scaled_mm/{name}', fn, args, dump=True)
            acc = per_k_fma(a, b)
            ref = scaled_reference(a, b, sa, sb, bias, out_dtype, acc, scale_result=sr)
            cmp = compare(y, ref)
            row = dict(comparison=cmp, graph=rec, order_sensitivity=order_sensitivity(a, b, acc),
                       reference='independent: exact FP8 dequantisation, per-K binary32 FMA in ascending K, '
                                 'acc * (scale_a * scale_b) + bias in binary32, one rounding to out_dtype')
            try:
                cpu = torch._scaled_mm(*args[:4], bias=bias, scale_result=sr, out_dtype=out_dtype,
                                       use_fast_accum=fast)
                row['versus_torch_cpu_scaled_mm'] = compare(y, cpu)
                # What torch CPU computes instead (ATen _scaled_mm_out_cpu_emulated):
                # each operand scaled in binary32 first, then a binary32 matmul.
                scaled_first = torch.matmul(a.float() * sa, b.float() * sb)
                if bias is not None:
                    scaled_first = scaled_first + bias.float()
                if sr is not None and out_dtype in (E4, E5):
                    scaled_first = scaled_first * (1 / sr.float())
                row['torch_cpu_versus_scale_first_emulation'] = compare(cpu, scaled_first.to(out_dtype))
            except Exception as e:                           # noqa: BLE001
                row['versus_torch_cpu_scaled_mm'] = dict(error=f'torch CPU _scaled_mm: {type(e).__name__}: {str(e)[:200]}')
            # One m.mma.async per 16-column output fragment per K step of
            # the configured tile (TNCA_GEMM_TILE = BLOCK_M, BLOCK_N, BLOCK_K).
            _bm, bn, bk = (int(v) for v in os.environ.get('TNCA_GEMM_TILE', '16,16,16').split(','))
            programs = -(-M // _bm) * -(-N // bn)
            expect_mma = programs * -(-K // bk) * (bn // 16)
            census = rec.get('mma', {})
            fp8_mma = sum(n for k, n in census.items() if k.endswith(f'.{tag[fmt]}'))
            other_mma = sum(n for k, n in census.items() if not k.endswith(f'.{tag[fmt]}'))
            row.update(expected_dynamic_mma=expect_mma, static_fp8_mma=fp8_mma, static_other_mma=other_mma)
            template = any('tem_fused' in k for k in rec['kernels'])
            mma_ok = fp8_mma > 0 and other_mma == 0 and (rec.get('mma_commands') in (None, expect_mma))
            ok = (judge(cmp) == 'pass' and template and mma_ok and not rec['host_fallbacks'])
            cpu_cmp = row['versus_torch_cpu_scaled_mm']
            cpu_note = (summary_of(cpu_cmp) + (f" (max ulp {cpu_cmp.get('max_ulp_diff')})" if cpu_cmp.get('value_mismatches') else '')
                        if 'elements' in cpu_cmp else cpu_cmp.get('error'))
            if 'torch_cpu_versus_scale_first_emulation' in row:
                emu = row['torch_cpu_versus_scale_first_emulation']
                cpu_note += f"; torch CPU = scale-first emulation: {emu['bit_mismatches']} mismatches"

            row['summary'] = (f'vs per-K reference: {summary_of(cmp)}; vs torch CPU: {cpu_note}; '
                              f'FP8 MMA static {fp8_mma}, dynamic {rec.get("mma_commands")} (expected {expect_mma})')
            run.add('scaled_mm', name, 'pass' if ok else 'fail', **row)
        except Exception as e:                               # noqa: BLE001
            run.add('scaled_mm', name, 'error', summary=f'{type(e).__name__}: {str(e)[:400]}',
                    traceback=traceback.format_exc()[-4000:])
    # Mixed encodings: torch accepts e4m3 x e5m2; ISA sec 9C's mod[3:2]
    # names one input format for both operands, so the compiler converts
    # both exactly to FP16 for one 16-deep K block and refuses deeper ones.
    try:
        a, _ = fp8_operands(16, 16, 32, E4, g)
        _, b = fp8_operands(16, 16, 32, E5, g)
        sa, sb = torch.tensor(0.5), torch.tensor(0.25)
        fn = lambda a, b, sa, sb: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
        (y,), rec = run.compile_run('scaled_mm/mixed', fn, (a, b, sa, sb), dump=True)
        ref = scaled_reference(a, b, sa, sb, None, torch.bfloat16)
        cmp = compare(y, ref)
        run.add('scaled_mm', 'e4m3 x e5m2 tensorwise 16x16x32 -> bf16', judge(cmp),
                summary=f'vs per-K reference: {summary_of(cmp)}; MMA {rec.get("mma")}', comparison=cmp, graph=rec)
    except Exception as e:                                   # noqa: BLE001
        run.add('scaled_mm', 'e4m3 x e5m2 tensorwise 16x16x32 -> bf16', 'refused',
                summary=f'{type(e).__name__}: {str(e)[:300]}', traceback=traceback.format_exc()[-3000:])


# ------------------------------------------------------------ linear --------
def w8a8_linear(x, weight, weight_scale, input_scale, bias):
    """Static W8A8: the input quantised with a static scale, FP8 GEMM with
    both scales, bias, bf16 out. Plain torch."""
    import torch
    q = (x / input_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    y = torch._scaled_mm(q.reshape(-1, q.shape[-1]), weight.t(), scale_a=input_scale,
                         scale_b=weight_scale, bias=bias, out_dtype=torch.bfloat16)
    return q, y.reshape(*x.shape[:-1], weight.shape[0])


def weight_only_linear(x, weight, weight_scale, bias):
    """Weight-only FP8: the weight upcast and scaled, then a bf16 GEMM."""
    import torch
    return torch.nn.functional.linear(x, weight.to(x.dtype) * weight_scale, bias)


def group_linear(run):
    torch = _torch()
    E4 = torch.float8_e4m3fn
    g = torch.Generator().manual_seed(23)
    shapes = ([(16,), 64, 32], [(7,), 40, 24]) if run.quick else \
        ([(16,), 64, 32], [(7,), 40, 24], [(33,), 72, 40], [(64,), 256, 192], [(2, 13), 96, 80])
    for lead, K, N in shapes:
        x = (torch.randn(*lead, K, generator=g) * 2).to(torch.bfloat16)
        w = (torch.randn(N, K, generator=g) * torch.pow(2.0, torch.randint(-2, 3, (N, K), generator=g).float())).to(E4)
        ws, xs = torch.tensor(0.0213), torch.tensor(0.0157)
        bias = (torch.randn(N, generator=g)).to(torch.bfloat16)
        shape = f'x{tuple(x.shape)} w{tuple(w.shape)}'
        name = f'static W8A8 {shape}'
        try:
            (q, y), rec = run.compile_run(f'linear/{name}', w8a8_linear, (x, w, ws, xs, bias), dump=True)
            x2 = x.reshape(-1, K)
            q_ref = (x2.float() / xs).clamp(-448.0, 448.0).to(E4)
            y_ref = scaled_reference(q_ref, w.t(), xs, ws, bias, torch.bfloat16).reshape(*lead, N)
            cq, cy = compare(q, q_ref.reshape(q.shape)), compare(y, y_ref)
            row = dict(comparisons=dict(quantised_input=cq, output=cy), graph=rec,
                       reference='quantisation in binary32 (Inductor computes x / scale in binary32), '
                                 'then the independent _scaled_mm reference')
            # torch CPU _scaled_mm on the device's own quantised input: the
            # GEMM's difference alone, without the quantisation's.
            row['versus_torch_cpu_scaled_mm_same_q'] = compare(y, torch._scaled_mm(
                q.reshape(-1, K), w.t(), xs, ws, bias=bias, out_dtype=torch.bfloat16).reshape(*lead, N))
            try:
                qe, ye = w8a8_linear(x, w, ws, xs, bias)
                row['versus_eager_cpu'] = dict(quantised_input=compare(q, qe), output=compare(y, ye))
                # Eager rounds x / scale to bf16 (the tensor's dtype) before
                # clamping and casting; Inductor's Triton kernels keep the
                # binary32 quotient. Both statements, checked:
                double = (x2.float() / xs).to(torch.bfloat16).clamp(-448.0, 448.0).to(E4)
                row['eager_q_is_bf16_rounded_quotient'] = compare(qe.reshape(double.shape), double)
            except Exception as e:                           # noqa: BLE001
                row['versus_eager_cpu'] = dict(error=f'{type(e).__name__}: {str(e)[:200]}')
            census = rec.get('mma', {})
            fp8 = sum(n for k, n in census.items() if k.endswith('.e4m3'))
            other = sum(n for k, n in census.items() if not k.endswith('.e4m3'))
            ok = judge(cq) == judge(cy) == 'pass' and fp8 > 0 and other == 0 and not rec['host_fallbacks']
            ev = row['versus_eager_cpu']
            evs = (f"q {summary_of(ev['quantised_input'])}"
                   + (f" (max {ev['quantised_input'].get('max_ulp_diff')} FP8 ulp; eager q = bf16-rounded quotient: "
                      f"{row['eager_q_is_bf16_rounded_quotient']['bit_mismatches']} mismatches)"
                      if ev['quantised_input'].get('value_mismatches') else '')
                   + f", y {summary_of(ev['output'])}"
                   + (f" (max abs {ev['output'].get('max_abs_diff', 0):.3g} of {ev['output'].get('max_abs_reference', 0):.3g})"
                      if ev['output'].get('value_mismatches') else '')
                   if 'quantised_input' in ev else ev.get('error'))
            same = row['versus_torch_cpu_scaled_mm_same_q']
            row['summary'] = (f'q: {summary_of(cq)}; y: {summary_of(cy)}; FP8 MMA static {fp8}, '
                              f'dynamic {rec.get("mma_commands")}; vs torch CPU _scaled_mm on the same q: '
                              f'{summary_of(same)}' + (f" (max ulp {same.get('max_ulp_diff')})" if same.get('value_mismatches') else '')
                              + f'; vs eager CPU: {evs}')
            run.add('linear', name, 'pass' if ok else 'fail', **row)
        except Exception as e:                               # noqa: BLE001
            run.add('linear', name, 'error', summary=f'{type(e).__name__}: {str(e)[:400]}',
                    traceback=traceback.format_exc()[-4000:])
        name = f'weight-only FP8 {shape}'
        try:
            (y,), rec = run.compile_run(f'linear/{name}', weight_only_linear, (x, w, ws, bias), dump=True)
            wd = (w.float() * ws).to(torch.bfloat16)
            x2 = x.reshape(-1, K)
            acc = per_k_fma(x2, wd.t())
            y_ref = (acc + bias.float()).to(torch.bfloat16).reshape(*lead, N)
            cy = compare(y, y_ref)
            row = dict(comparison=cy, graph=rec,
                       reference='dequantised weight rounded once to bf16; per-K binary32 FMA of the bf16 '
                                 'operands; bias added once in binary32; one rounding to bf16')
            ye = weight_only_linear(x, w, ws, bias)
            row['versus_eager_cpu'] = compare(y, ye)
            census = rec.get('mma', {})
            bf16 = sum(n for k, n in census.items() if k.endswith('.bf16'))
            ok = judge(cy) == 'pass' and bf16 > 0 and not rec['host_fallbacks']
            ev = row['versus_eager_cpu']
            row['summary'] = (f'y: {summary_of(cy)}; BF16 MMA static {bf16}, dynamic {rec.get("mma_commands")}; '
                              f'vs eager CPU: {summary_of(ev)}' + (f" (max ulp {ev.get('max_ulp_diff', 0)})"
                                                                   if ev.get('value_mismatches') else ''))
            run.add('linear', name, 'pass' if ok else 'fail', **row)
        except Exception as e:                               # noqa: BLE001
            run.add('linear', name, 'error', summary=f'{type(e).__name__}: {str(e)[:400]}',
                    traceback=traceback.format_exc()[-4000:])


# ------------------------------------------------------------ main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--group', action='append', choices=GROUPS)
    ap.add_argument('--quick', action='store_true',
                    help='reduced inputs and matrix (for the RTL simulators)')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native', TNCA_INDEX_OPT='1',
                      TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    backend = os.environ.setdefault('TNCA_BACKEND', 'systemc-tlm-grid')
    run = Run(out, args.quick, backend)
    import torch
    import triton
    import tnca
    if not Path(tnca.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit('tnca imported outside the requested source tree')
    if backend.startswith('systemc-tlm'):
        import tlm_backend
        run.models = tlm_backend.register()
    import tnca.torch_device as device
    from archbench.provenance import software_tree
    device.register()
    torch._dynamo.config.recompile_limit = 1 << 20
    torch._dynamo.config.accumulated_recompile_limit = 1 << 20
    run.report = dict(schema='tnca-fp8-cases/1', status='running', quick=args.quick, backend=backend,
                      execution='torch.compile(backend=inductor, fullgraph=True) on tnca tensors, Session on '
                                + backend, evidence=('model-prediction-uncalibrated' if backend.startswith('systemc')
                                                     else 'rtl-simulation'),
                      software_tree=software_tree(ROOT, refresh=True),
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__),
                                     cpu_capability=torch.backends.cpu.get_cpu_capability()),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    run.save()
    started = time.monotonic()
    for group in args.group or GROUPS:
        globals()[f'group_{group}'](run)
    run.report.update(status='complete', wall_seconds=round(time.monotonic() - started, 1))
    run.save()
    print(json.dumps(run.report['summary']))
    return 0 if not (run.report['summary']['fail'] or run.report['summary']['error']) else 1


if __name__ == '__main__':
    sys.exit(main())
