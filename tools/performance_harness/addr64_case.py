#!/usr/bin/env python3
"""R41 S2-04 D: standard Torch on device addresses above 4 GiB and storages over 2 GiB.

    addr64_case.py --out DIR CASE [CASE ...]

Each CASE runs in its own process on the tnca PyTorch device (TNCA_BACKEND,
default systemc-tlm-grid with TNCA_TLM_MEMORY_BYTES, e.g. 0x200000000 for
the 8 GiB window) and writes DIR/<case>/result.json, its Inductor wrappers
and the assembly of every kernel it launched. Every output is compared with
an independent CPU reference: the same Torch function evaluated eagerly by
PyTorch on CPU tensors holding the same values.

Cases:

  place:<addr>   every tensor, argument block and scratch at or above <addr>
                 (filler allocations take what is below): copy, add, a
                 reduction, softmax, mm, linear, embedding with int64
                 indices, cat of slice views
  large:<dtype>  one storage over 2 GiB (bf16, i8, u8: 2.25 GiB; bf16-4g:
                 4.25 GiB) with three filled 64-row bands (head, middle,
                 tail); slices, reductions, transposes, cat, embedding,
                 matmul through views whose byte offsets pass 2 GiB, a
                 graph input that is itself such a view, an in-place update
                 and eager copies
  clone          an eager device clone of a whole 2.25 GiB storage
  stream         a full 2.25 GiB bf16 storage: uploaded, read entirely by a
                 compiled reduction, written entirely by a compiled
                 pointwise op, copied entirely by an eager clone; every
                 byte compared against the CPU in chunks
  neg:<what>     refusals: truncation (a 64-bit index whose low 32 bits are
                 a valid row), window (an address past the device window),
                 hole (an access into the TMU page at 0x7fff0000), alias
                 (overlapping argument buffers), abi (argument values that do
                 not fit their slot), alloc (the allocator around the hole
                 and at the end of the window)

A case that should fail is PASS when the refusal happens and names its
reason, and FAIL when the program runs to completion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm'),
                str(ROOT/'tools/performance_harness/systemc')]

GIB = 1 << 30
HOLE = (0x7fff0000, 0x7fff1000)


# ------------------------------------------------------------------ setup ----
def _device():
    import torch
    import triton  # noqa: F401
    backend = os.environ.setdefault('TNCA_BACKEND', 'systemc-tlm-grid')
    if backend.startswith('systemc-tlm'):
        import tlm_backend
        tlm_backend.register()
    import tnca.torch_device as D
    D.register()
    torch.manual_seed(41)
    return torch, D


def _addr(D, t):
    return D.storage_address(t) + t.storage_offset() * t.element_size()


class Case:
    def __init__(self, out, name):
        self.out, self.name = out, name
        self.rows = []
        self.t0 = time.monotonic()

    def record(self, row):
        row['seconds'] = round(time.monotonic() - self.t0, 2)
        self.t0 = time.monotonic()
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        (self.out/'rows.json').write_text(json.dumps(self.rows, indent=1) + '\n')


def _compare(got, ref, tol):
    import torch
    if tuple(got.shape) != tuple(ref.shape) or got.dtype != ref.dtype:
        return dict(match=False, why=f'{tuple(got.shape)} {got.dtype} != {tuple(ref.shape)} {ref.dtype}')
    if tol is None:
        bad = (got.view(torch.uint8) != ref.view(torch.uint8)).sum().item() if got.dtype.itemsize == 1 \
            else (got != ref).sum().item()
        return dict(match=bad == 0, mismatches=int(bad), rule='bitwise equal')
    err = (got.double() - ref.double()).abs()
    bound = tol[0] + tol[1] * ref.double().abs()
    bad = int((err > bound).sum().item())
    return dict(match=bad == 0, mismatches=bad, max_abs_error=float(err.max().item()) if err.numel() else 0.0,
                rule=f'|got-ref| <= {tol[0]} + {tol[1]}*|ref|')


def _compiled(case, D, label, fn, dev_args, ref, tol=None, *, expect_int64=None):
    """torch.compile(fn) on tnca tensors against the CPU reference `ref`."""
    import torch
    from torch._inductor.utils import run_and_get_code
    from tnca.triton_backend.driver import launch_log
    row = dict(test=label)
    try:
        torch._dynamo.reset()
        with launch_log() as log:
            out, codes = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), *dev_args)
        got = out.cpu()
        row.update(_compare(got, ref, tol))
        row['inputs'] = [hex(_addr(D, a)) for a in dev_args if isinstance(a, torch.Tensor)]
        row['output'] = hex(_addr(D, out))
        row['kernels'] = sorted({e['kernel'] for e in log})
        row['launches'] = [dict(kernel=e['kernel'], grid=list(e['grid'])) for e in log]
        src = '\n'.join(codes)
        row['int64_index'] = bool(re.search(r'\.to\(tl\.int64\)|INDEX_DTYPE : tl\.constexpr = tl\.int64', src))
        if expect_int64 is not None and row['int64_index'] != expect_int64:
            row['match'] = False
            row['why'] = f'expected Inductor int64 indexing {expect_int64}'
        for i, c in enumerate(codes):
            (case.out/f'{label}.wrapper{i}.py').write_text(c)
        row['status'] = 'PASS' if row['match'] else 'WRONG'
    except Exception as e:                                   # noqa: BLE001
        row.update(status='FAIL', error=f'{type(e).__name__}: {" ".join(str(e).split())[:800]}')
    case.record(row)
    return row


def _place(torch, D, target):
    """Fill the heap below `target` so that every later allocation (tensors,
    argument blocks, scratch) lands at or above it."""
    s = D.MEMORY.session()
    heap = s._heap
    fillers = []
    for lo, hi in heap.reserved:
        if heap.next < lo < target:
            fillers.append(torch.empty(lo - heap.next, dtype=torch.uint8, device='tnca'))
    if target > heap.next:
        fillers.append(torch.empty(target - heap.next, dtype=torch.uint8, device='tnca'))
    holes = [(a, n) for a, n in heap.free_list]
    return fillers, dict(heap_next=hex(heap.next), free_below=[(hex(a), n) for a, n in holes])


# ------------------------------------------------------------- the graphs ----
def _standard_graphs(torch):
    """(label, fn, cpu inputs, tolerance) for the representative graphs."""
    F = torch.nn.functional
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g)               # noqa: E731
    bf = torch.bfloat16
    return [
        ('copy_transpose', lambda a: a.t().contiguous(), [r(48, 40)], None),
        ('add_f32', lambda a, b: a + b, [r(4100), r(4100)], None),
        ('add_bf16_rows', lambda a, b: a + b[None, :], [r(40, 64).to(bf), r(64).to(bf)], None),
        ('sum_rows', lambda a: a.sum(-1), [r(32, 300)], (1e-4, 1e-5)),
        ('softmax', lambda a: torch.softmax(a, -1), [r(8, 100)], (1e-6, 1e-5)),
        ('mm_bf16', lambda a, b: a @ b, [r(32, 48).to(bf), r(48, 40).to(bf)], (2e-2, 2e-2)),
        ('linear_bf16_m1', lambda x, w: F.linear(x, w), [r(1, 64).to(bf), r(96, 64).to(bf)], (2e-2, 2e-2)),
        ('embedding_i64', lambda w, i: F.embedding(i, w),
         [r(1000, 64).to(bf), torch.randint(0, 1000, (37,), generator=g)], None),
        ('cat_slices', lambda a, b: torch.cat([a[:, 3:], b[1:]], 0) * 2, [r(20, 35), r(9, 32)], None),
    ]


def case_place(case, target):
    torch, D = _device()
    fillers, info = _place(torch, D, target)
    case.record(dict(test='placement', target=hex(target), **info, status='PASS'))
    for label, fn, args, tol in _standard_graphs(torch):
        ref = fn(*args)
        dev = [a.to('tnca') for a in args]
        row = _compiled(case, D, label, fn, dev, ref, tol)
        low = [a for a in row.get('inputs', []) + [row.get('output', '0x0')] if int(a, 16) < target]
        if low:
            row['status'] = 'FAIL'
            row['why'] = f'tensors below the target address: {low}'
    s = D.MEMORY.session()
    ev = s.latest_event
    arg = ev._task['arg_addr'] if ev is not None and getattr(ev, '_task', None) else None
    case.record(dict(test='argument_block', last_arg_block=hex(arg) if arg else None,
                     status='PASS' if arg and arg >= target else 'FAIL'))


# ---------------------------------------------------------- large storage ----
LARGE = {'bf16': ('bfloat16', 9 << 28), 'i8': ('int8', 9 << 28), 'u8': ('uint8', 9 << 28),
         'bf16-4g': ('bfloat16', 17 << 28)}


def case_large(case, which):
    torch, D = _device()
    dtype = getattr(torch, LARGE[which][0])
    nbytes, C = LARGE[which][1], 2048
    esize = torch.empty(0, dtype=dtype).element_size()
    R = nbytes // (C * esize)
    g = torch.Generator().manual_seed(11)
    big = torch.empty((R, C), dtype=dtype, device='tnca')
    bands = {}
    for lo in (0, R // 2, R - 64):
        if dtype.is_floating_point:
            x = (torch.randn(64, C, generator=g) * 4).to(dtype)
        else:
            x = torch.randint(-120 if dtype == torch.int8 else 0, 120, (64, C), generator=g, dtype=dtype)
        big[lo:lo + 64] = x.to('tnca')
        bands[lo] = x
    head, mid, tail = bands[0], bands[R // 2], bands[R - 64]
    base = D.storage_address(big)
    case.record(dict(test='storage', dtype=str(dtype), rows=R, cols=C, bytes=big.untyped_storage().nbytes(),
                     base=hex(base), end=hex(base + big.untyped_storage().nbytes()),
                     tail_band_byte_offset=(R - 64) * C * esize, status='PASS'))
    # int8 widens through sitofp; uint8 stays integer (u8 -> float is a
    # separate, address-independent defect: see the evidence README).
    up = (lambda t: t.float()) if dtype == torch.int8 else (lambda t: t)
    tol = (2e-2, 2e-2) if dtype == torch.bfloat16 else None
    int64 = R * C > (1 << 31) - 1

    def run(label, fn, ref, *extra, tol=tol):
        return _compiled(case, D, f'{which}.{label}', fn, [big, *extra], ref, tol, expect_int64=int64)

    run('tail_pointwise', lambda b: up(b[-16:]) * 2 + 1, up(tail[-16:]) * 2 + 1)
    run('tail_rows_sum', lambda b: up(b[-8:]).float().sum(-1), up(tail[-8:]).float().sum(-1), tol=(1e-3, 1e-4))
    run('far_cat', lambda b: torch.cat([b[:8, :64], b[-8:, :64], b[R // 2:R // 2 + 8, :64]]),
        torch.cat([head[:8, :64], tail[-8:, :64], mid[:8, :64]]), tol=None)
    run('tail_transpose', lambda b: up(b[-16:, -48:]).t().contiguous(), up(tail[-16:, -48:]).t().contiguous())
    idx = torch.tensor([0, 1, R - 1, R - 64, R // 2 + 3, 5, R - 2, R // 2, R - 1, 2], dtype=torch.int64)
    rows = torch.stack([next(x[i - lo] for lo, x in bands.items() if lo <= i < lo + 64) for i in idx.tolist()])
    run('embedding_far_rows', lambda b, i: torch.nn.functional.embedding(i, b), rows, idx.to('tnca'), tol=None)
    if dtype == torch.bfloat16:
        x = torch.randn(4, 64, generator=g).to(dtype)
        run('mm_tail_weight_slice', lambda b, x: x @ b[-64:, :48],
            (x.float() @ tail[-64:, :48].float()).to(dtype), x.to('tnca'), tol=(5e-2, 2e-2))
        run('linear_tail_weight_slice', lambda b, x: torch.nn.functional.linear(x, b[-48:, :64]),
            torch.nn.functional.linear(x.float(), tail[-48:, :64].float()).to(dtype), x.to('tnca'),
            tol=(5e-2, 2e-2))
    # A graph input that is itself a view whose storage offset passes 2 GiB.
    view = big[-16:]
    _compiled(case, D, f'{which}.view_input', lambda a: up(a) + 3, [view], up(tail[-16:]) + 3, tol)
    # In-place through a view past 2 GiB: the neighbours must not change.
    sl = big[R - 40:R - 24]
    try:
        torch._dynamo.reset()
        torch.compile(lambda a: a.mul_(2) if dtype.is_floating_point else a.add_(1), fullgraph=True,
                      dynamic=False)(sl)
        expect = tail.clone()
        if dtype.is_floating_point:
            expect[24:40] = (tail[24:40].float() * 2).to(dtype)
        else:
            expect[24:40] = tail[24:40] + 1
        res = _compare(big[R - 64:].cpu(), expect, None)
        case.record(dict(test=f'{which}.inplace_view', view=hex(_addr(D, sl)), **res,
                         status='PASS' if res['match'] else 'WRONG'))
        bands[R - 64] = tail = expect
    except Exception as e:                                   # noqa: BLE001
        case.record(dict(test=f'{which}.inplace_view', status='FAIL', error=f'{type(e).__name__}: {e}'[:800]))
    # Eager device-to-device copies of views past 2 GiB.
    for label, fn, ref in (('eager_tail_clone', lambda: big[-16:].clone(), tail[-16:]),
                           ('eager_tail_to_host', lambda: big[-64:], tail)):
        try:
            res = _compare(fn().cpu(), ref, None)
            case.record(dict(test=f'{which}.{label}', **res, status='PASS' if res['match'] else 'WRONG'))
        except Exception as e:                               # noqa: BLE001
            case.record(dict(test=f'{which}.{label}', status='FAIL', error=f'{type(e).__name__}: {e}'[:800]))


# ---------------------------------------------------------------- stream ----
def case_stream(case):
    torch, D = _device()
    R, C = 589824, 2048                    # 2.25 GiB of bf16
    CH = 32768                             # rows per host chunk (128 MiB)

    def chunk(i):
        g = torch.Generator().manual_seed(1000 + i)
        return (torch.randn(min(CH, R - i * CH), C, generator=g) * 3).bfloat16()
    nch = -(-R // CH)
    x = torch.empty((R, C), dtype=torch.bfloat16, device='tnca')
    t = time.monotonic()
    for i in range(nch):
        x[i * CH:(i + 1) * CH] = chunk(i).to('tnca')
    case.record(dict(test='upload', bytes=x.untyped_storage().nbytes(), base=hex(D.storage_address(x)),
                     end=hex(D.storage_address(x) + x.untyped_storage().nbytes()),
                     upload_seconds=round(time.monotonic() - t, 1), status='PASS'))

    def check(label, dev, ref_of, tol):
        bad, worst = 0, 0.0
        for i in range(nch):
            got = dev[i * CH:(i + 1) * CH].cpu()
            res = _compare(got, ref_of(i), tol)
            bad += res.get('mismatches', 0)
            worst = max(worst, res.get('max_abs_error', 0.0))
        return dict(test=label, mismatches=bad, max_abs_error=worst, status='PASS' if bad == 0 else 'WRONG')

    from tnca.triton_backend.driver import launch_log
    torch._dynamo.reset()
    with launch_log() as log:
        s = torch.compile(lambda a: a.float().sum(-1), fullgraph=True, dynamic=False)(x)
    got = s.cpu()
    ref = torch.cat([chunk(i).float().sum(-1) for i in range(nch)])
    res = _compare(got, ref, (1e-2, 1e-4))
    case.record(dict(test='read_all_row_sums', kernels=[(e['kernel'], list(e['grid'])) for e in log],
                     **res, status='PASS' if res['match'] else 'WRONG'))
    del s
    torch._dynamo.reset()
    with launch_log() as log:
        y = torch.compile(lambda a: a * 2 - 1, fullgraph=True, dynamic=False)(x)
    row = check('write_all_pointwise', y, lambda i: chunk(i) * 2 - 1, None)
    row.update(output=hex(_addr(D, y)), output_end=hex(_addr(D, y) + y.untyped_storage().nbytes()),
               kernels=[(e['kernel'], list(e['grid'])) for e in log])
    case.record(row)
    del y
    t = time.monotonic()
    z = x.clone()
    D.MEMORY.synchronize()
    row = check('eager_clone_all', z, chunk, None)
    row.update(clone_seconds=round(time.monotonic() - t, 1), output=hex(_addr(D, z)))
    case.record(row)


def case_clone(case):
    """An eager device-to-device clone of a whole 2.25 GiB storage (one copy
    launch indexes in i32; past 2**31 - 2**20 bytes it is several)."""
    torch, D = _device()
    R, C = 589824, 2048
    g = torch.Generator().manual_seed(5)
    x = torch.empty((R, C), dtype=torch.bfloat16, device='tnca')
    bands = {}
    for lo in (0, R // 2 - 32, R - 64):
        bands[lo] = (torch.randn(64, C, generator=g) * 2).bfloat16()
        x[lo:lo + 64] = bands[lo].to('tnca')
    t = time.monotonic()
    try:
        z = x.clone()
        D.MEMORY.synchronize()
        bad = sum(_compare(z[lo:lo + 64].cpu(), b, None)['mismatches'] for lo, b in bands.items())
        case.record(dict(test='eager_clone_2_25_gib', bytes=x.untyped_storage().nbytes(),
                         source=hex(_addr(D, x)), copy=hex(_addr(D, z)), band_mismatches=bad,
                         clone_seconds=round(time.monotonic() - t, 1), status='PASS' if bad == 0 else 'WRONG'))
    except Exception as e:                                   # noqa: BLE001
        case.record(dict(test='eager_clone_2_25_gib', status='FAIL',
                         error=f'{type(e).__name__}: {" ".join(str(e).split())}'[:600]))


# ------------------------------------------------------------- negatives ----
def _expect_refusal(case, label, fn, pattern):
    try:
        out = fn()
        case.record(dict(test=label, status='FAIL', why='ran to completion', result=repr(out)[:200]))
    except Exception as e:                                   # noqa: BLE001
        text = f'{type(e).__name__}: {" ".join(str(e).split())}'
        hit = re.search(pattern, text) is not None
        case.record(dict(test=label, status='PASS' if hit else 'FAIL', refused=text[:600], expected=pattern))


def _tail_of_model_stderr(D):
    dev = D.MEMORY.session().dev
    try:
        return (Path(dev.dir)/'stderr.log').read_text()[-400:]
    except Exception:                                        # noqa: BLE001
        return ''


def case_neg(case, what):
    torch, D = _device()
    import torch._inductor.config as icfg
    F = torch.nn.functional
    if what == 'truncation':
        # Without Inductor's own index assert, an index of 2**32 + 5 reaches
        # the address computation; its low 32 bits are the valid row 5.
        icfg.assert_indirect_indexing = False
        w = torch.randn(64, 16).bfloat16()
        idx = torch.tensor([1, 2, (1 << 32) + 5, 3])
        dw, di = w.to('tnca'), idx.to('tnca')
        _expect_refusal(case, 'int64_index_high_bits', lambda: torch.compile(
            lambda w, i: F.embedding(i, w), fullgraph=True, dynamic=False)(dw, di).cpu(),
            r'TRAP_USER|KERNEL_FAULT|refused|faulted')
        case.rows[-1]['model'] = _tail_of_model_stderr(D)
    elif what == 'window':
        icfg.assert_indirect_indexing = False
        # A 2.25 GiB int8 table (Inductor indexes it in 64 bits) and an index
        # whose row lies past the device window: the exactly formed 64-bit
        # address must reach the memory system and be refused there.
        R = 9 << 28 >> 11
        big = torch.empty((R, 2048), dtype=torch.int8, device='tnca')
        far = (D.MEMORY.session().dev.memory_bytes // 2048) + 7
        idx = torch.tensor([0, far, 1])
        _expect_refusal(case, 'address_past_window', lambda: torch.compile(
            lambda w, i: F.embedding(i, w).float(), fullgraph=True, dynamic=False)(big, idx.to('tnca')).cpu(),
            r'outside model memory window|KERNEL_FAULT|DeviceLost|refused')
        case.rows[-1]['model'] = _tail_of_model_stderr(D)
    elif what == 'window-alloc':
        s = D.MEMORY.session()
        free = s._heap.capacity - s._heap.used
        _expect_refusal(case, 'allocation_past_window',
                        lambda: torch.empty(free + (1 << 20), dtype=torch.uint8, device='tnca'),
                        r'out of device memory|OutOfMemory|capacity')
        from tnca.errors import TncaError  # noqa: F401
        buf = s.allocate(4096)
        _expect_refusal(case, 'upload_past_buffer', lambda: s.upload(buf, bytes(64), offset=4096 - 32),
                        r'runs past')
        _expect_refusal(case, 'transfer_past_window',
                        lambda: s.dev.download(s.dev.memory_bytes - 64, 128),
                        r'outside model memory window|refused|DeviceLost')
    elif what == 'hole':
        icfg.assert_indirect_indexing = False
        s = D.MEMORY.session()
        heap = s._heap
        # Leave 4 KiB + 64 B below the TMU page. An 8 KiB storage cannot fit
        # there and must start after the page; a 4 KiB one then reuses the
        # gap and must end at or before the page.
        pad = torch.empty(HOLE[0] - heap.next - 4160, dtype=torch.uint8, device='tnca')
        straddle = torch.empty(8192, dtype=torch.uint8, device='tnca')
        below = torch.empty(4096, dtype=torch.int8, device='tnca')
        sa, ba = D.storage_address(straddle), D.storage_address(below)
        case.record(dict(test='storage_that_would_straddle_moves_past_page', storage=hex(sa),
                         status='PASS' if sa >= HOLE[1] else 'FAIL'))
        case.record(dict(test='gap_below_page_reused_without_overlap', storage=hex(ba), end=hex(ba + 4096),
                         status='PASS' if HOLE[0] - 4160 <= ba and ba + 4096 <= HOLE[0] else 'FAIL'))
        stats = s.memory_stats
        case.record(dict(test='heap_reserved_ranges', reserved=[[hex(a), hex(b)] for a, b in
                                                                stats['heap_reserved_ranges']],
                         status='PASS' if [tuple(x) for x in stats['heap_reserved_ranges']] == [HOLE] else 'FAIL'))
        # An out-of-bounds row of `below` that lands inside the TMU page.
        row = (HOLE[0] - ba) // 16 + 3
        w = below.view(256, 16)
        _expect_refusal(case, 'kernel_access_into_tmu_page', lambda: torch.compile(
            lambda w, i: F.embedding(i, w).float(), fullgraph=True, dynamic=False)(
                w, torch.tensor([0, row]).to('tnca')).cpu(),
            r'TMU MMIO is not ordinary model memory|KERNEL_FAULT|DeviceLost|refused')
        case.rows[-1]['row_address'] = hex(ba + 16 * row)
        case.rows[-1]['model'] = _tail_of_model_stderr(D)
        del pad
    elif what == 'alias':
        a = torch.randn(128, 64).bfloat16()
        da = a.to('tnca')
        # Two overlapping views of one storage into a kernel that uses the
        # transfer engine: the runtime cannot prove the engine and the LSU
        # keep program order on shared bytes, so it refuses the overlap.
        _expect_refusal(case, 'dma_kernel_overlapping_arguments', lambda: torch.compile(
            lambda x, y: x @ y, fullgraph=True, dynamic=False)(da[:32], da[:64, :48]).cpu(),
            r'ALIAS_UNDECLARED|overlap')
        # Disjoint views of one storage: the runtime's span for a pointer
        # runs to the end of its buffer (it is given no extent), so this is
        # refused too -- recorded as the conservative rule it is.
        torch._dynamo.reset()
        try:
            got = torch.compile(lambda x, y: x @ y, fullgraph=True, dynamic=False)(da[:32], da[64:, :48]).cpu()
            res = _compare(got, (a[:32].float() @ a[64:, :48].float()).bfloat16(), (5e-2, 2e-2))
            case.record(dict(test='dma_kernel_disjoint_views', outcome='ran', **res,
                             status='PASS' if res['match'] else 'WRONG'))
        except Exception as e:                               # noqa: BLE001
            case.record(dict(test='dma_kernel_disjoint_views', outcome='refused', status='NOTE',
                             refused=f'{type(e).__name__}: {" ".join(str(e).split())}'[:400]))
        # LSU-only kernels declare their pointer pairs and run in program
        # order: two views of one storage, disjoint and far apart.
        b = torch.randn(4096)
        db = b.to('tnca')
        torch._dynamo.reset()
        got = torch.compile(lambda x, y: x * 2 + y, fullgraph=True, dynamic=False)(db[:2048], db[2048:]).cpu()
        res = _compare(got, b[:2048] * 2 + b[2048:], None)
        case.record(dict(test='lsu_kernel_two_views_of_one_storage', **res,
                         status='PASS' if res['match'] else 'WRONG'))
    elif what == 'abi':
        from tnca import abi
        _expect_refusal(case, 'i32_argument_too_wide', lambda: abi.check_value('n', 'i32', 1 << 31), r'does not fit')
        _expect_refusal(case, 'pointer_outside_64_bits',
                        lambda: abi.encode_block([('p', 'device_ptr64')], {'p': 1 << 64}), r'outside 64 bits')
        from tnca.tensor import TensorDescriptor
        _expect_refusal(case, 'descriptor_dimension_past_i32',
                        lambda: TensorDescriptor(None, (1 << 31,), (1,), (16,), 'u8'), r'exceed signed i32')
        data = TensorDescriptor(None, (3, 4), (8, 1), (2, 4), 'bf16').pack((1 << 33) + 0x40, 1 << 20)
        import struct
        base = struct.unpack_from('<Q', data, 0)[0]
        stride = struct.unpack_from('<q', data, 48)[0]
        case.record(dict(test='descriptor_base_above_4g_packed_64bit', base=hex(base), stride=stride,
                         status='PASS' if base == (1 << 33) + 0x40 and stride == 16 else 'FAIL'))
    else:
        raise SystemExit(f'unknown negative case {what}')


# ------------------------------------------------------------------ main ----
def child(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case = Case(out, args.child)
    dump = out/'dump'
    os.environ.update(TNCA_KERNEL_DUMP='1', TNCA_DUMP_DIR=str(dump))
    status = 'complete'
    try:
        kind, _, arg = args.child.partition(':')
        if kind == 'place':
            case_place(case, int(arg, 0))
        elif kind == 'large':
            case_large(case, arg)
        elif kind == 'stream':
            case_stream(case)
        elif kind == 'clone':
            case_clone(case)
        elif kind == 'neg':
            case_neg(case, arg)
        else:
            raise SystemExit(f'unknown case {args.child}')
    except Exception:                                        # noqa: BLE001
        status = 'crashed'
        case.record(dict(test='harness', status='FAIL', error=traceback.format_exc()[-2000:]))
    asm = {}
    if dump.is_dir():
        for s in sorted(dump.glob('*/*.s')):
            asm[s.parent.name] = hashlib.sha256(s.read_bytes()).hexdigest()
    result = dict(case=args.child, status=status, rows=case.rows,
                  passed=sum(r['status'] == 'PASS' for r in case.rows),
                  failed=[r['test'] for r in case.rows if r['status'] not in ('PASS', 'NOTE')],
                  asm_sha256=asm, env={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
    (out/'result.json').write_text(json.dumps(result, indent=1) + '\n')
    return 0 if status == 'complete' and not result['failed'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--child', help=argparse.SUPPRESS)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('cases', nargs='*')
    args = ap.parse_args()
    if args.child:
        return child(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.cases:
        d = out/re.sub(r'[^A-Za-z0-9_.-]', '_', name)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with (d/'log.txt').open('w') as log:
            try:
                rc = subprocess.run([sys.executable, __file__, '--out', str(d), '--child', name],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
        res = json.loads((d/'result.json').read_text()) if (d/'result.json').is_file() else {}
        summary[name] = dict(returncode=rc, passed=res.get('passed'), failed=res.get('failed'))
        print(name, json.dumps(summary[name]), flush=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return 0 if all(v['returncode'] == 0 for v in summary.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
