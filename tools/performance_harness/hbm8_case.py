#!/usr/bin/env python3
"""R41 hbm8: an 8 GiB HBM window on the three-core image, through standard Torch.

    hbm8_case.py --case CASE --out DIR            # board: TNCA_BACKEND(_ARGS) from the env
    hbm8_case.py --case CASE --out DIR --sim      # sparse 8 GiB RTL stand-in (hbm8_sim.py)
        [--rtl-root R] [--hbm-base B] [--small-geometry]

Every case opens its own session (one process per case), asks the backend for
an 8 GiB window and places real tnca tensors at chosen device addresses with
the production first-fit allocator (fillers are untouched torch.empty
storages; they never move a probe). Device work is torch.compile'd standard
Torch on all three cores; host transfers are the backend's DMA path with its
cache-ownership handoff. Cases:

  window     the BO, the window registers, the heap and the PC0-PC31 map
  samples    a probe at every allocated PC boundary, both BO edges and the
             virtual 4 GiB boundary: host DMA write, in-place device add on
             three cores, host DMA read; distinct values catch any alias
  crossing   device-only store-load across physical 4 GiB, virtual 4 GiB and
             the window end, then DMA out and DMA in across the same lines
  alias      pairs that differ only in address bit 30/31/32 and the TMU page
             + 2/4/6 GiB: a dropped bit makes two of them one line
  bigoffset  one 2.25 GiB storage; kernels whose element offsets times the
             element size exceed 2^31 (int32 offset scaling)
  dma        DMA in and out of large tensors at the top of the window
  stream     board: one 2.25 GiB storage written by DMA, read entirely by a
             device row reduction, read back by DMA; sim: same at 8 KiB
  negatives  host/allocator/MMIO refusals at the 8 GiB end and the reserved
             TMU page (board-safe: nothing here can latch a device fault)
  clamp      request 16 GiB: the runtime exposes exactly the 8 GiB group

DIR/report.json: status functional-pass | failed-preserved, with every
address, counter delta and check. Functional evidence only; no performance
gate is decided here. A device-side out-of-window access latches a sticky
AXI fault that only an image reset clears, so it is tested in RTL
(tb/fpga/tb_line_axi_hbm8.v), never on the shared card.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler/python'), str(ROOT / 'tools/asm'),
                str(ROOT / 'tools/r40'), str(ROOT / 'tools/performance_harness')]
from hbm_address import topology_probe_fill_sizes, topology_probe_offsets  # noqa: E402

GIB = 1 << 30
WINDOW = 8 * GIB
LOCAL = 0x20000                      # R_AXI_STAT LOCAL_LIMIT of the B40 image
TMU = (0x7fff0000, 0x7fff1000)       # python/tnca/runtime.py DEVICE_RESERVED_RANGES
PROBE = 512
FRAME = 64 << 10
TOP_FREE = 1 << 20                   # free top of the window left by non-sample cases
CASES = ('window', 'samples', 'crossing', 'alias', 'bigoffset', 'dma', 'stream', 'negatives', 'clamp')


# ---------------------------------------------------------------------------
# The plan: device-address placements per case. Needed before the session for
# the stand-in's sparse map, and by the case itself to place its tensors.

def sizes(mode):
    board = mode == 'board'
    return dict(cross=(64 << 10) if board else 2048,
                dma=(256 << 20) if board else (16 << 10),
                dma_chunk=64 << 20,
                big_rows=36, big_cols=1 << 24, big_touch=16,
                stream_rows=36864 if board else 32, stream_cols=16384 if board else 64,
                stream_chunk=64 << 20)


def window_like(base, size):
    from hbm8_sim import synthetic_allocation
    return dict(base=base, size=size, local_limit=LOCAL, allocation=synthetic_allocation(base, size))


def sample_layout(window):
    """R40 v2 topology probes plus the virtual 4 GiB boundary; the last one
    carries the 64 KiB launch-frame prefix (tools/r40/hbm_address.py)."""
    offsets = set(topology_probe_offsets(window))
    virtual4 = (1 << 32) - LOCAL - PROBE // 2
    if 0 <= virtual4 < window['size'] - PROBE:
        offsets.add(virtual4)
    offsets = sorted(offsets)
    if any(b - a < PROBE for a, b in zip(offsets, offsets[1:])):
        raise ValueError('probe offsets overlap')
    out = []
    for i, off in enumerate(offsets):
        prefix = FRAME if i == len(offsets) - 1 else 0
        out.append(dict(offset=off, storage_start=off - prefix, storage_bytes=prefix + PROBE, view_offset=prefix))
    return out


def physical_boundary_offsets(window):
    """BO offsets whose physical address is a multiple of 4 GiB (bit 32/33 carry)."""
    base, size = window['base'], window['size']
    first = -(-base // (4 * GIB)) * 4 * GIB
    return [p - base for p in range(first, base + size, 4 * GIB) if 0 < p - base < size]


def plan(case, mode, window):
    """[(name, device_address, nbytes)] ascending, plus extra sim backing."""
    z = sizes(mode)
    size = window['size']
    top = LOCAL + size
    placements, extra = [], []
    if case == 'samples':
        for i, e in enumerate(sample_layout(window)):
            placements.append((f'probe{i}', LOCAL + e['storage_start'], e['storage_bytes']))
    elif case == 'crossing':
        span = z['cross']
        points = [('physical-%dGiB' % ((window['base'] + o) // GIB), LOCAL + o) for o in physical_boundary_offsets(window)]
        points.append(('virtual-4GiB', 1 << 32))
        points = sorted(points, key=lambda p: p[1])
        for name, b in points:
            placements.append((name, b - span // 2, span))
        placements.append(('near-window-end', top - TOP_FREE - span, span))
    elif case == 'alias':
        targets = []
        # Lows at 256/512/768/1024 MiB, highs at +1/+2/+4/+6 GiB: all distinct.
        for name, a, delta in (('bit30', LOCAL + (256 << 20), 1 << 30), ('bit31', LOCAL + (512 << 20), 1 << 31),
                               ('bit32', LOCAL + (768 << 20), 1 << 32), ('bits32+31', LOCAL + (1024 << 20), 3 << 31)):
            targets += [(f'{name}-low', a), (f'{name}-high', a + delta)]
        for k in (2, 4, 6):
            targets.append((f'tmu+{k}GiB', TMU[0] + k * GIB))
        for name, a in sorted(targets, key=lambda t: t[1]):
            placements.append((name, a, PROBE))
    elif case == 'bigoffset':
        # One storage; first fit places it after the reserved TMU page.
        nbytes = z['big_rows'] * z['big_cols'] * 4
        base = TMU[1]
        for r in range(z['big_rows']):
            extra.append((base + r * z['big_cols'] * 4, 4096))
        placements.append(('big', base, nbytes))
    elif case == 'dma':
        n = z['dma']
        placements.append(('dst-6GiB', LOCAL + 6 * GIB, n))
        placements.append(('src-top', top - TOP_FREE - n, n))
    elif case == 'stream':
        nbytes = z['stream_rows'] * z['stream_cols'] * 4
        if mode == 'board':
            placements.append(('stream', TMU[1], nbytes))
        else:
            b = LOCAL + physical_boundary_offsets(window)[0]
            placements.append(('stream', b - nbytes // 2, nbytes))
    elif case == 'negatives':
        extra.append((top - 4096, 4096))
    for _, a, n in placements:
        if a < LOCAL or a + n > top or (a < TMU[1] and a + n > TMU[0]):
            raise ValueError(f'placement {a:#x}+{n:#x} outside ordinary window memory')
    if any(a1 < a0 + n0 for (_, a0, n0), (_, a1, _) in zip(placements, placements[1:])):
        raise ValueError('placements must be ascending and disjoint')
    return placements, extra


def sparse_ranges(case, mode, window):
    """BO-offset ranges the stand-in backs for a case: the heap bottom (low
    tensors, launch frames, compiler temporaries), each placement and a
    margin above it; everything else in the 8 GiB is deliberately absent."""
    from hbm8_sim import merge_ranges
    placements, extra = plan(case, mode, window)
    ranges = [(0, 8 << 20)]
    for _, a, n in placements:
        if case in ('bigoffset',) and n > (64 << 20):
            continue                         # only its touched rows (extra)
        ranges.append((a - LOCAL, n + (64 << 10)))
    ranges += [(a - LOCAL, n) for a, n in extra]
    return merge_ranges(ranges, window['size'])


# ---------------------------------------------------------------------------
# Session-side helpers.

class Ctx:
    def __init__(self, mode, out):
        import torch
        import tnca.torch_device as D
        from tnca.triton_backend.driver import launch_log
        self.torch, self.D, self.launch_log = torch, D, launch_log
        self.mode, self.out, self.z = mode, out, sizes(mode)
        self.held, self.checks, self.launches = [], [], []
        self.session = D.MEMORY.session()
        self.dev = self.session.dev
        w = self.dev.axi_window
        self.window = dict(base=w['base'], size=w['size'], local_limit=w['local_limit'],
                           allocation=w.get('allocation'))

    def check(self, name, ok, **detail):
        self.checks.append(dict(name=name, ok=bool(ok), **detail))
        if not ok:
            raise AssertionError(f'{name}: {json.dumps(detail, default=str)[:800]}')

    def heap(self):
        h = self.session._heap
        return dict(next=h.next, limit=h.limit, free=list(h.free_list), reserved=list(h.reserved))

    def place(self, name, addr, nbytes, dtype=None):
        """A tnca tensor whose storage starts exactly at device address addr."""
        torch = self.torch
        dtype = dtype or torch.int32
        with self.session._lock:
            top = self.session._heap.next
        if addr < top:
            raise AssertionError(f'{name}: allocator already passed {addr:#x} (next {top:#x})')
        for fill in topology_probe_fill_sizes(top, addr, self.session._heap.reserved):
            self.held.append(torch.empty(fill, dtype=torch.uint8, device='tnca'))
        item = torch.empty((), dtype=dtype).element_size()
        t = torch.empty(nbytes // item, dtype=dtype, device='tnca')
        got = t.untyped_storage().data_ptr()
        self.check(f'placed:{name}', got == addr, want=hex(addr), got=hex(got), bytes=nbytes)
        return t

    def place_all(self, placements, dtype=None, frame=1 << 20):
        """Place ascending storages above a launch-frame hole kept at the heap
        bottom and released afterwards, so first fit puts launch frames and
        compiler outputs there. Placements end at least TOP_FREE below the
        window top (the launch reserve stays free); only `samples` reaches the
        exact last byte, with the R40 frame-prefix layout."""
        torch = self.torch
        hole = torch.empty(frame, dtype=torch.uint8, device='tnca')
        out = [self.place(name, addr, nbytes, dtype) for name, addr, nbytes in placements]
        del hole
        return out

    def metrics(self):
        return self.dev.memory_port_metrics()

    def run(self, label, fn, *args):
        """One standard Torch call; per-launch core and outlet deltas."""
        before = self.metrics()
        with self.torch.no_grad(), self.launch_log() as log:
            out = fn(*args)
        self.session.drain()
        after = self.metrics()
        grid = self.dev.physical_grid_metrics()
        outlets = [dict(port=a['port'], read_lines=b['read_lines'] - a['read_lines'],
                        write_lines=b['write_lines'] - a['write_lines']) for a, b in zip(before, after)]
        row = dict(label=label, launches=len(log), kernels=[e.get('kernel') for e in log],
                   grids=[e.get('grid') for e in log],
                   device_cycles=sum(e.get('device_cycles') or 0 for e in log),
                   last_launch_cores=[dict(core=i, issued=c['issued'], completed=c['completed'])
                                      for i, c in enumerate(grid['cores'])],
                   peak_active=grid.get('peak_active'), outlets=outlets,
                   upload_bytes=sum(e.get('upload_bytes') or 0 for e in log),
                   download_bytes=sum(e.get('download_bytes') or 0 for e in log))
        self.launches.append(row)
        return out, row

    def all_cores(self, row):
        return all(c['issued'] > 0 and c['issued'] == c['completed'] for c in row['last_launch_cores'])


def pattern_i32(torch, n, salt, addr):
    """Distinct small int32 words: lane, the storage's own line address, a salt."""
    lane = torch.arange(n, dtype=torch.int64)
    v = (lane * 1000003 + (addr >> 6) * 7919 + salt * 104729) % (1 << 26) + (1 << 20)
    return v.to(torch.int32)


def pattern_f16(torch, n, i):
    """Probe i's fp16 integers: lane -> (a_i * lane + b_i) mod 1021, a_i = 2i + 3.
    Two probes' affine maps agree on at most one lane, so a probe that reads
    or overwrites another's line differs in >= 31 of its 32 elements; values
    stay below 1021 + 1000, exact in fp16 through the device's +1/+1000."""
    lane = torch.arange(n, dtype=torch.int64)
    return ((lane * (2 * i + 3) + 7 * i + 1) % 1021).to(torch.float16)


def small_f32(torch, start, n):
    """Exact small floats 0..15 from a 64-bit element index (sums exact in fp32)."""
    idx = torch.arange(start, start + n, dtype=torch.int64)
    return (((idx * 2654435761) >> 13) & 15).to(torch.float32)


# ---------------------------------------------------------------------------
# Cases

def case_window(c, report):
    from tnca.backends import u50
    w = c.dev.axi_window
    link = c.dev.link
    stat = link.read_reg(u50.R_AXI_STAT)
    regs = dict(base=link.read_reg(u50.R_AXI_BASE), size=link.read_reg(u50.R_AXI_SIZE),
                base_alias_hbm1=link.read_reg(0x100), base_alias_hbm2=link.read_reg(0x108), stat=stat)
    alloc = w.get('allocation') or {}
    banks = alloc.get('physical_banks') or []
    heap = c.session._heap
    report.update(window=dict(base=hex(w['base']), size=w['size'], local_limit=hex(w['local_limit']), dma=w['dma']),
                  registers={k: hex(v) for k, v in regs.items()}, allocation=alloc,
                  heap=dict(base=hex(heap.base), limit=hex(heap.limit), capacity=heap.capacity,
                            reserved=[[hex(a), hex(b)] for a, b in heap.reserved]),
                  memory_configuration=c.dev.memory_configuration(),
                  capabilities={k: v for k, v in c.dev.capabilities().items()
                                if k in ('memory_bytes', 'memory_ports', 'hbm_dma', 'cache_ownership_handoff', 'medium')})
    c.check('window-size-8GiB', w['size'] == WINDOW, size=w['size'])
    c.check('local-limit', w['local_limit'] == LOCAL and (stat >> 32) << 12 == LOCAL, stat=hex(stat))
    c.check('stat-present-no-fault', stat & 2 and not stat & 1 and not stat & 4, stat=hex(stat))
    c.check('registers-readback', regs['base'] == w['base'] and regs['size'] == WINDOW and
            regs['base_alias_hbm1'] == regs['base_alias_hbm2'] == w['base'], regs={k: hex(v) for k, v in regs.items()})
    c.check('requested-equals-exposed', alloc.get('requested_bytes') == alloc.get('exposed_bytes') == WINDOW,
            requested=alloc.get('requested_bytes'), exposed=alloc.get('exposed_bytes'))
    pcs = [b['pc'] for b in banks]
    c.check('pc0-pc31-contiguous', pcs == list(range(32)) and
            all(b['allocated_base'] == w['base'] + i * (256 << 20) and b['allocated_size'] == 256 << 20
                for i, b in enumerate(banks)), pcs=pcs)
    c.check('heap-ends-at-window', heap.limit == LOCAL + WINDOW and heap.base == LOCAL, limit=hex(heap.limit))
    c.check('tmu-page-reserved', list(heap.reserved) == [TMU] and heap.capacity == WINDOW - (TMU[1] - TMU[0]),
            reserved=list(heap.reserved), capacity=heap.capacity)
    if c.mode == 'board':
        c.check('xrt-group-bo', alloc.get('connected_memory', {}).get('tag') == 'MBG[0:31]' and
                alloc.get('connected_memory', {}).get('size') == WINDOW and alloc.get('topology_sha256') ==
                'e3cf166dc3fbef5859e5c38fb183525932eae24f2f99101e24a022213a694f10',
                connected=alloc.get('connected_memory'), topology=alloc.get('topology_sha256'))


def case_samples(c, report):
    torch = c.torch
    layout = sample_layout(c.window)
    placements, _ = plan('samples', c.mode, c.window)
    bump = torch.compile(lambda t: t.add_(1), fullgraph=True, dynamic=False)
    probes, control = [], None
    for i, (e, (name, addr, nbytes)) in enumerate(zip(layout, placements)):
        if i == len(layout) - 1 and control is not None:
            del control
            control = None
        storage = c.place(name, addr, nbytes, torch.float16)
        x = storage[e['view_offset'] // 2:]
        va = x.data_ptr()
        c.check(f'view:{name}', va == LOCAL + e['offset'] and x.numel() == PROBE // 2, va=hex(va))
        h = pattern_f16(torch, PROBE // 2, i)
        x.copy_(h)                                     # host DMA write
        probes.append(dict(name=name, storage=storage, x=x, h=h, va=va))
        if i == 0:
            control = torch.empty(FRAME, dtype=torch.uint8, device='tnca')
    del control
    rows = []
    for p in probes:
        _, row = c.run(p['name'], bump, p['x'])
        rows.append(row)
    bad = [p['name'] for p in probes if not torch.equal(p['x'].cpu(), p['h'] + 1)]
    base, size = c.window['base'], c.window['size']
    phys = [(p['va'] - LOCAL + base, p['va'] - LOCAL + base + PROBE) for p in probes]
    banks = c.window['allocation']['physical_banks']
    hits = [[b['pc'] for b in banks if lo < b['allocated_base'] + b['allocated_size'] and hi > b['allocated_base']]
            for lo, hi in phys]
    report.update(samples=[dict(name=p['name'], device_address=hex(p['va']), physical=hex(lo), pcs=pc)
                           for p, (lo, _), pc in zip(probes, phys, hits)])
    c.check('values', not bad, wrong=bad)
    c.check('pc0-pc31-covered', sorted({pc for g in hits for pc in g}) == list(range(32)))
    c.check('pc-boundaries-crossed', sum(len(g) == 2 for g in hits) == 31, crossings=sum(len(g) == 2 for g in hits))
    c.check('above-4GiB-and-near-8GiB', max(p['va'] for p in probes) + PROBE == LOCAL + size and
            sum(p['va'] >= 1 << 32 for p in probes) >= 16)
    c.check('three-cores-each-launch', all(c.all_cores(r) for r in rows), cores=[r['last_launch_cores'] for r in rows[:3]])
    total = [sum(r['outlets'][k]['read_lines'] + r['outlets'][k]['write_lines'] for r in rows) for k in range(3)]
    c.check('three-outlets', all(t > 0 for t in total), outlet_lines=total)


def case_crossing(c, report):
    torch = c.torch
    placements, _ = plan('crossing', c.mode, c.window)
    n = c.z['cross'] // 4
    src = torch.empty(n, dtype=torch.int32, device='tnca')
    low = torch.empty(n, dtype=torch.int32, device='tnca')
    ys = list(zip([p[0] for p in placements], c.place_all(placements)))
    store = torch.compile(lambda y, s: y.copy_(s * 3 + 7), fullgraph=True, dynamic=False)
    load = torch.compile(lambda d, y: d.copy_(y * 5 - 1), fullgraph=True, dynamic=False)
    out = []
    for i, (name, y) in enumerate(ys):
        a = y.data_ptr()
        p = pattern_i32(torch, n, 100 + i, a)
        src.copy_(p)
        _, r1 = c.run(f'{name}:device-store', store, y, src)
        _, r2 = c.run(f'{name}:device-load', load, low, y)
        got_low, got_y = low.cpu(), y.cpu()             # DMA out (handoff writes back)
        c.check(f'{name}:store-load', torch.equal(got_low, (p * 3 + 7) * 5 - 1))
        c.check(f'{name}:dma-out', torch.equal(got_y, p * 3 + 7))
        q = pattern_i32(torch, n, 200 + i, a)
        y.copy_(q)                                      # DMA in across the boundary
        _, r3 = c.run(f'{name}:device-load-after-dma', load, low, y)
        c.check(f'{name}:dma-in', torch.equal(low.cpu(), q * 5 - 1))
        c.check(f'{name}:three-cores', all(c.all_cores(r) for r in (r1, r2, r3)))
        out.append(dict(name=name, start=hex(a), end=hex(a + n * 4),
                        physical_start=hex(a - LOCAL + c.window['base']), launches=[r1, r2, r3]))
    report.update(crossings=out)


def case_alias(c, report):
    torch = c.torch
    placements, _ = plan('alias', c.mode, c.window)
    low = torch.empty(PROBE // 2, dtype=torch.float16, device='tnca')
    xs = list(zip([p[0] for p in placements], c.place_all(placements, torch.float16)))
    load = torch.compile(lambda d, y: d.copy_(y + 0), fullgraph=True, dynamic=False)
    bump = torch.compile(lambda t: t.add_(1000), fullgraph=True, dynamic=False)
    pats = {}
    for i, (name, x) in enumerate(xs):
        pats[name] = pattern_f16(torch, PROBE // 2, i)
        x.copy_(pats[name])                            # every probe written before any is read
    rows = []
    for name, x in xs:                                 # device LOAD first: an MMIO alias reads registers
        _, r = c.run(f'{name}:load', load, low, x)
        rows.append(r)
        c.check(f'{name}:device-load', torch.equal(low.cpu(), pats[name]), address=hex(x.data_ptr()))
    for name, x in xs:
        _, r = c.run(f'{name}:store', bump, x)
        rows.append(r)
    for name, x in xs:
        c.check(f'{name}:device-store', torch.equal(x.cpu(), pats[name] + 1000), address=hex(x.data_ptr()))
    c.check('three-cores', all(c.all_cores(r) for r in rows))
    report.update(alias=[dict(name=n, device_address=hex(x.data_ptr()),
                              physical=hex(x.data_ptr() - LOCAL + c.window['base'])) for n, x in xs])


def case_bigoffset(c, report):
    torch = c.torch
    z = c.z
    rows, cols, k = z['big_rows'], z['big_cols'], z['big_touch']
    placements, _ = plan('bigoffset', c.mode, c.window)
    name, addr, nbytes = placements[0]
    small = torch.empty(rows * k, dtype=torch.float32, device='tnca')
    big = torch.empty(rows * cols, dtype=torch.float32, device='tnca')   # first fit: after the TMU page
    c.check('placed:big', big.data_ptr() == addr, want=hex(addr), got=hex(big.data_ptr()))
    heads = torch.stack([small_f32(torch, r * cols, k) + r for r in range(rows)])
    for r in range(rows):
        big[r * cols:r * cols + k].copy_(heads[r])     # 64 B DMA writes, 64 MiB apart
    # Graph inputs are (rows, k) views with a 64 MiB row stride: the kernels'
    # element offsets reach 35 * 2^24 + 31, i.e. byte offsets above 2^31, and
    # only these 2 * rows * k elements of the 2.25 GiB storage are touched.
    heads_view = big.as_strided((rows, k), (cols, 1), 0)
    tails_view = big.as_strided((rows, k), (cols, 1), k)
    read = torch.compile(lambda v: v * 2 + 0.5, fullgraph=True, dynamic=False)
    out, r1 = c.run('strided-read', read, heads_view)
    c.check('read-beyond-2GiB-bytes', torch.equal(out.cpu(), heads * 2 + 0.5))
    small.copy_((heads.flatten() + 3).contiguous())
    write = torch.compile(lambda v, s: v.copy_(s.view(rows, k)), fullgraph=True, dynamic=False)
    _, r2 = c.run('strided-write', write, tails_view, small)
    got = torch.stack([big[r * cols + k:r * cols + 2 * k].cpu() for r in range(rows)])
    c.check('write-beyond-2GiB-bytes', torch.equal(got, heads + 3))
    last = (rows - 1) * cols * 4 + 2 * k * 4
    report.update(big=dict(start=hex(addr), end=hex(addr + nbytes), bytes=nbytes,
                           largest_touched_byte_offset=last, launches=[r1, r2]))
    c.check('offset-exceeds-int32', last > (1 << 31))


def _chunks(total, step):
    return [(o, min(step, total - o)) for o in range(0, total, step)]


def case_dma(c, report):
    torch = c.torch
    n = c.z['dma']
    placements, _ = plan('dma', c.mode, c.window)
    dst, src = c.place_all(placements, dtype=torch.float32)
    m = n // 4
    t0 = time.time()
    digests = []
    for off, ln in _chunks(m, c.z['dma_chunk'] // 4):
        v = small_f32(torch, off, ln)
        src[off:off + ln].copy_(v)
        digests.append(hashlib.sha256((v * 2 + 1).numpy().tobytes()).hexdigest())
    upload_s = time.time() - t0
    fn = torch.compile(lambda d, s: d.copy_(s * 2 + 1), fullgraph=True, dynamic=False)
    _, row = c.run('top-to-6GiB', fn, dst, src)
    t0 = time.time()
    bad = []
    for i, (off, ln) in enumerate(_chunks(m, c.z['dma_chunk'] // 4)):
        got = dst[off:off + ln].cpu()
        if hashlib.sha256(got.numpy().tobytes()).hexdigest() != digests[i]:
            bad.append(off)
        if not torch.equal(src[off:off + ln].cpu(), small_f32(torch, off, ln)):
            bad.append(('src', off))
    download_s = time.time() - t0
    c.check('dma-in-device-dma-out', not bad, bad=bad[:8])
    c.check('three-cores', c.all_cores(row))
    report.update(dma=dict(src=hex(src.data_ptr()), dst=hex(dst.data_ptr()), bytes_each=n,
                           host_upload_seconds=round(upload_s, 3), host_download_seconds=round(download_s, 3),
                           launch=row))


def case_stream(c, report):
    torch = c.torch
    rows, cols = c.z['stream_rows'], c.z['stream_cols']
    nbytes = rows * cols * 4
    placements, _ = plan('stream', c.mode, c.window)
    if c.mode == 'board':
        x = torch.empty(rows * cols, dtype=torch.float32, device='tnca')  # first fit: after the TMU page
        c.check('placed:stream', x.data_ptr() == placements[0][1], got=hex(x.data_ptr()))
    else:
        x, = c.place_all(placements, dtype=torch.float32)
    step = c.z['stream_chunk'] // 4
    want = torch.zeros(rows, dtype=torch.float64)
    digests = []
    t0 = time.time()
    for off, ln in _chunks(rows * cols, step):
        v = small_f32(torch, off, ln)
        x[off:off + ln].copy_(v)
        digests.append(hashlib.sha256(v.numpy().tobytes()).hexdigest())
        idx = torch.arange(off, off + ln) // cols
        want.index_add_(0, idx, v.double())
    upload_s = time.time() - t0
    fn = torch.compile(lambda t: t.view(rows, cols).sum(1), fullgraph=True, dynamic=False)
    out, row = c.run('row-sum', fn, x)
    got = out.cpu().double()
    c.check('streaming-read-exact', torch.equal(got, want), max_abs=float((got - want).abs().max()))
    t0 = time.time()
    bad = [off for i, (off, ln) in enumerate(_chunks(rows * cols, step))
           if hashlib.sha256(x[off:off + ln].cpu().numpy().tobytes()).hexdigest() != digests[i]]
    download_s = time.time() - t0
    c.check('dma-out-whole-storage', not bad, bad=bad[:8])
    c.check('three-cores', c.all_cores(row))
    read_lines = sum(o['read_lines'] for o in row['outlets'])
    c.check('outlets-carried-the-storage', read_lines * 64 >= nbytes, read_lines=read_lines)
    a = x.data_ptr()
    report.update(stream=dict(start=hex(a), end=hex(a + nbytes), bytes=nbytes,
                              physical_start=hex(a - LOCAL + c.window['base']),
                              physical_end=hex(a + nbytes - LOCAL + c.window['base']),
                              host_upload_seconds=round(upload_s, 3), host_download_seconds=round(download_s, 3),
                              device_read_bytes_per_cycle=(read_lines * 64 / row['device_cycles']
                                                           if row['device_cycles'] else None),
                              launch=row, performance_status='diagnostic-only'))
    if c.mode == 'board':
        c.check('single-storage-over-2GiB', nbytes > 2 * GIB and a + nbytes > 1 << 32)


def case_negatives(c, report):
    from tnca.backends import u50
    from tnca.errors import ArgumentError, BufferError
    from tnca.runtime import _Allocator, DEVICE_RESERVED_RANGES
    s, dev, link = c.session, c.dev, c.dev.link
    end = LOCAL + c.window['size']
    rows = []

    def expect(name, fn, exc):
        try:
            fn()
        except exc as e:
            rows.append(dict(name=name, refused=True, error=type(e).__name__))
            return
        except Exception as e:                          # noqa: BLE001
            rows.append(dict(name=name, refused=False, error=f'{type(e).__name__}: {e}'))
            raise AssertionError(f'{name}: wrong refusal {type(e).__name__}')
        rows.append(dict(name=name, refused=False))
        raise AssertionError(f'{name}: accepted')

    before = c.heap()
    expect('allocate-past-capacity', lambda: s.allocate(s._heap.capacity + 64), BufferError)
    c.check('allocator-unchanged-by-refusal', c.heap() == before)
    expect('upload-past-window-end', lambda: dev.upload(end - 8, b'\0' * 16), ArgumentError)
    # A download asks the device first; its 64-bit range check refuses it.
    reply, data = dev.download(end, 8)
    rows.append(dict(name='download-at-window-end', status=reply.status, bytes=len(data)))
    c.check('download-at-window-end-refused', reply.status == 3 and data == b'', status=reply.status)
    # Device-side 64-bit range checks (tnca_multicore_mmio range_ok/host_address_ok):
    # refusals are status words, never memory accesses.
    def status(op, a0, a1=0):
        return dev.request(op, a0, a1).status
    mmio = dict(download_last_word=status(2, end - 8, 1), download_past_end=status(2, end - 8, 2),
                download_at_end=status(2, end, 1), download_wrap=status(2, (1 << 64) - 8, 1),
                download_bit33=status(2, end + (8 << 30), 1),
                handoff_last_line=status(18, end - 64, 64), handoff_past_end=status(18, end - 64, 128),
                handoff_below_window=status(18, LOCAL - 64, 64))
    rows.append(dict(name='mmio-range-status', statuses=mmio))
    c.check('mmio-range', mmio['download_last_word'] == 0 and mmio['handoff_last_line'] == 0 and
            all(mmio[k] == 3 for k in ('download_past_end', 'download_at_end', 'download_wrap',
                                       'download_bit33', 'handoff_past_end', 'handoff_below_window')), statuses=mmio)
    # Coherent host window at the last word, then the DMA path for the same word.
    def window(addr, value=None):
        link.write_reg(u50.R_HOST_ADDR, addr)
        if value is not None:
            link.write_reg(u50.R_HOST_DATA, value)
        link.write_reg(u50.R_HOST_CTRL, u50.CTRL_START | (u50.CTRL_WE if value is not None else 0))
        for _ in range(100000):
            st = link.read_reg(u50.R_HOST_STATUS)
            if not st & u50.ST_BUSY:
                return st, link.read_reg(u50.R_HOST_DATA)
        raise AssertionError('host window stayed busy')
    marker = 0x5EED_0000_8000_0000 | (end - 8)
    st_w, _ = window(end - 8, marker)
    st_r, got = window(end - 8)
    st_x, _ = window(end)
    reply, data = dev.download(end - 8, 8)
    rows.append(dict(name='host-window', write_status=st_w, read_status=st_r, past_end_status=st_x,
                     read=hex(got), dma_read=data.hex()))
    c.check('host-window-last-word', not st_w & u50.ST_REFUSED and not st_r & u50.ST_REFUSED and got == marker and
            reply.ok and int.from_bytes(data, 'little') == marker, got=hex(got), dma=data.hex())
    c.check('host-window-past-end-refused', bool(st_x & u50.ST_REFUSED), status=st_x)
    # The reserved TMU page in an 8 GiB heap (pure allocator; no device access).
    heap = _Allocator(LOCAL, end, reserved=DEVICE_RESERVED_RANGES)
    a1, _ = heap.alloc(TMU[0] - LOCAL - 64)
    a2, _ = heap.alloc(128)
    a3, _ = heap.alloc(4 * GIB)
    big = _Allocator(LOCAL, end, reserved=DEVICE_RESERVED_RANGES)
    b1, _ = big.alloc(int(2.25 * GIB))
    rows.append(dict(name='reserved-page', straddle_moved_to=hex(a2), next_4GiB=hex(a3), first_2_25GiB=hex(b1)))
    c.check('reserved-page-skipped', a2 == TMU[1] and a3 >= TMU[1] and b1 == TMU[1] and a1 == LOCAL)
    c.check('largest-single-storage', (end - TMU[1]) > 5 * GIB and (TMU[0] - LOCAL) < 2 * GIB)
    status_reg = link.read_reg(u50.R_AXI_STAT)
    c.check('no-sticky-fault', not status_reg & 1 and not status_reg & 4, stat=hex(status_reg))
    report.update(negatives=rows)


def case_clamp(c, report):
    w = c.dev.axi_window
    alloc = w.get('allocation') or {}
    report.update(window=dict(base=hex(w['base']), size=w['size']), allocation=alloc)
    c.check('requested-16GiB', alloc.get('requested_bytes') == 2 * WINDOW)
    c.check('exposed-is-the-group', w['size'] == WINDOW == alloc.get('exposed_bytes') and
            c.session._heap.limit == LOCAL + WINDOW, size=w['size'])


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=CASES, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sim', action='store_true', help='sparse 8 GiB RTL stand-in instead of the board')
    ap.add_argument('--rtl-root', type=Path, help='stand-in RTL tree (default: this checkout)')
    ap.add_argument('--hbm-base', type=lambda v: int(v, 0), default=0, help='stand-in BO address')
    ap.add_argument('--small-geometry', action='store_true', help='stand-in with the 4x2 bench cache')
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mode = 'sim' if args.sim else 'board'
    request = 2 * WINDOW if args.case == 'clamp' else WINDOW
    os.environ.update(TNCA_ACTIVE_CORES='3', TNCA_RCTX_SPMD='1', TNCA_MAX_BLOCK='16', TNCA_TUNE='0',
                      TNCA_TENSOR_DESCRIPTORS='1', TNCA_TORCH_DEVICE='native')
    os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
    os.environ.setdefault('TRITON_CACHE_DIR', str(out / 'triton-cache'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(out / 'inductor-cache'))
    report = dict(schema='tnca-r41-hbm8-case/1', case=args.case, mode=mode, status='running',
                  requested_hbm_bytes=request, performance_gate='not-measured')
    if mode == 'sim':
        import hbm8_sim
        hbm8_sim.register()
        window = window_like(args.hbm_base, WINDOW)
        ranges = sparse_ranges(args.case, mode, window)
        lines = hbm8_sim.write_map(out / 'sparse.map', ranges)
        config = hbm8_sim.backend_args(sparse_map=out / 'sparse.map', window_bytes=WINDOW, hbm_base=args.hbm_base,
                                       rtl_root=args.rtl_root, image_geometry=not args.small_geometry)
        config['hbm_bytes'] = request
        os.environ.update(TNCA_BACKEND=hbm8_sim.NAME, TNCA_BACKEND_ARGS=json.dumps(config))
        os.environ.setdefault('TNCA_HOST_TIMEOUT', '7200')
        report.update(sparse=dict(ranges=[[hex(a), hex(n)] for a, n in ranges], backed_lines=lines),
                      stand_in=config)
    else:
        if os.environ.get('TNCA_BACKEND') != 'fpga-u280-multicore':
            ap.error('board mode needs TNCA_BACKEND=fpga-u280-multicore (or pass --sim)')
        config = json.loads(os.environ.get('TNCA_BACKEND_ARGS', '{}'))
        if int(config.get('hbm_bytes', 0)) != request:
            ap.error(f'case {args.case} needs backend hbm_bytes={request}')
        report['backend_args'] = config
    report['backend'] = os.environ['TNCA_BACKEND']

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    save()
    started = time.time()
    c = None
    try:
        import torch
        import triton
        import tnca.torch_device as D
        from archbench.provenance import software_tree
        D.register()
        tree = software_tree(ROOT, refresh=True)
        report.update(software_tree=tree, torch=torch.__version__, triton=triton.__version__,
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')})
        c = Ctx(mode, out)
        report['window'] = dict(base=hex(c.window['base']), size=c.window['size'],
                                local_limit=hex(c.window['local_limit']))
        if args.case != 'clamp':
            c.check('window-geometry', c.window['size'] == WINDOW and c.window['local_limit'] == LOCAL,
                    window=report['window'])
        globals()['case_' + args.case](c, report)
        if software_tree(ROOT, refresh=True) != tree:
            raise ValueError('software changed during the case')
        report['status'] = 'functional-pass'
    except BaseException:
        report.update(status='failed-preserved', error=traceback.format_exc())
        raise
    finally:
        if c is not None:
            report.update(checks=c.checks, launches=c.launches)
            try:
                st = c.dev.axi_status()
                report['axi_status_at_end'] = st
            except Exception as e:                      # noqa: BLE001
                report['axi_status_at_end'] = f'unreadable: {type(e).__name__}: {e}'
        report['wall_seconds'] = round(time.time() - started, 1)
        save()
    print(json.dumps(dict(case=args.case, mode=mode, status=report['status'],
                          checks=sum(ch['ok'] for ch in c.checks) if c else 0), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
