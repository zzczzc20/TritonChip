#!/usr/bin/env python3
"""R40-04 on the board: the HBM image's capacity, address map and traffic.

    TNCA_BACKEND=fpga-u280 TNCA_BACKEND_ARGS='{"xclbin": ..., "image_profile": "core-vrf-sync-v5"}' \\
    TNCA_PROFILE=core-vrf-sync-v5 TNCA_HBM_BYTES=... \\
    $TORCH_PYTHON tools/r40/hbm_board.py [--only CASE ...] [--json OUT] [--evidence]

Everything goes through the standard path: tnca tensors, torch.compile,
Inductor, Triton, the tnca device, the U280. Cases:

  window     the image reports an AXI window (R_AXI_STAT), the runtime gave it
             a buffer in the bank the link connected, and the heap ends where
             the window does
  samples    one tensor per step through the whole window -- with the window
             over HBM[0:31], one in each pseudo-channel -- each written by the
             host (DMA), incremented in place on the device, and read back
             only after every one has been, so an address that loses a high bit
             aliases two samples and is caught
  capacity   a standard Torch graph whose working set is >= 64 MiB (a GEMV over
             a 64 MiB fp16 weight), right against the CPU, with the bridge's
             counters showing the reads went to HBM
  stream     read (sum), write (fill) and copy graphs over a buffer larger than
             any on-chip store, for the bandwidth ledger: bytes per device
             cycle and the mean AXI read latency from the counters (sec 3.4's
             calibration input)

The gate R40-04.hbm passes when window, samples and capacity pass on the
board; stream is recorded for R40-04.bandwidth and does not decide it here.
The capacity and stream kernels run for tens of millions of cycles at one
8-byte beat per access: set TNCA_DEADLINE above that (e.g. 4000000000).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler' / 'python', ROOT / 'tools' / 'asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')

import torch                                            # noqa: E402
import triton                                           # noqa: E402,F401
import tnca.torch_device as D                           # noqa: E402

CASES = []
from hbm_address import PC_BYTES, probe_offsets, topology_probe_offsets
CLOCK_HZ = 100e6
# --small: the same cases at sizes the RTL simulator runs in minutes, against
# the simulated /hbm memory mode (TNCA_PROFILE=core-vrf-sync-v5/hbm) -- a dry
# run of this tool, not evidence for the gate.
SMALL = False


def case(fn):
    CASES.append(fn)
    return fn


def _session():
    from tnca.triton_backend.driver import _Sessions
    return _Sessions.get(os.environ['TNCA_PROFILE'])


def _counters():
    dev = _session().dev
    return dev.axi_counters() if hasattr(dev, 'axi_counters') else None


def _window(s):
    """The AXI window: the board's, or for the simulated memory mode the
    bench's (python/tnca/device.py)."""
    w = getattr(s.dev, 'axi_window', None)
    if w is None and SMALL:
        from tnca import device as _device
        local = _device.HBM_LOCAL_LIMIT
        w = dict(base=0, size=_device.memory_limit(os.environ['TNCA_PROFILE']) - local,
                 local_limit=local, dma=False)
    return w


def _delta(a, b):
    if a is None or b is None:
        return dict(reads=0, writes=0, read_cycles=0, read_beats=0)
    return {k: b[k] - a[k] for k in a}


def _launch_cycles(fn):
    from tnca.triton_backend.driver import launch_log
    with torch.no_grad(), launch_log() as log:
        out = fn()
        if isinstance(out, torch.Tensor):
            out.cpu()
    return out, sum(e['device_cycles'] or 0 for e in log), len(log)


@case
def window():
    s = _session()
    dev = s.dev
    w = _window(s)
    assert w, 'the image reports no AXI window (R_AXI_STAT bit 1); is this the HBM image?'
    bank = getattr(getattr(dev, 'link', None), 'hbm_bank', {}) or {}
    assert s._heap.limit == w['local_limit'] + w['size'], (s._heap.limit, w)
    st = dev.axi_status() if hasattr(dev, 'axi_status') else dict(error=False, stray=False)
    assert not st['error'] and not st['stray'], st
    if os.environ.get('R40_CONTRACT')=='v2':
        assert w['size']>=1<<30, 'v2 requires >=1 GiB actually exposed HBM, not the requested size'
        topology_probe_offsets(w)  # Refuse absent/ambiguous physical mapping.
        assert len(w['allocation']['physical_banks'])>=2, 'v2 requires at least two allocated physical PCs'
    return dict(base=hex(w['base']), size=w['size'], local_limit=hex(w['local_limit']),
                bank=bank.get('tag'), bank_bytes=bank.get('size'), bank_index=bank.get('index'),
                heap_end=hex(s._heap.limit), dma=w['dma'], allocation=w.get('allocation'))


@case
def samples():
    s = _session()
    w = _window(s)
    local, size = w['local_limit'], w['size']
    v2=os.environ.get('R40_CONTRACT')=='v2'
    offsets = topology_probe_offsets(w) if v2 else probe_offsets(w['base'], size)
    # The final v2 probe reaches the BO end. Preserve a separate small frame
    # region, then release it before launching the unchanged Torch operations.
    control=s.allocate(65536,name='hbm-boundary-control-reserve') if v2 else None
    n = len(offsets)
    bump = torch.compile(lambda t: t.add_(1), fullgraph=True)
    held, got = [], []
    for i, offset in enumerate(offsets):
        # A pad from wherever the heap's top is to this sample's place: the
        # probe's address is where the pad then starts (first fit).
        probe = torch.empty(8, dtype=torch.uint8, device='tnca')
        top = probe.untyped_storage().data_ptr()
        del probe
        gap = local + offset - top
        assert gap >= 0, f'allocator passed probe {i}: {top:#x} > {local + offset:#x}'
        if gap:
            held.append(torch.empty(gap, dtype=torch.uint8, device='tnca'))
        x = torch.empty(64, dtype=torch.int64, device='tnca')
        assert x.untyped_storage().data_ptr() == local + offset, 'probe allocation moved'
        h = torch.arange(64, dtype=torch.int64) * 1000003 + i * 7919 + (1 << 40) + (i << 50)
        x.copy_(h)
        got.append((h, x))
    addrs = [x.untyped_storage().data_ptr() for _, x in got]
    if control is not None:s.free(control)
    for _, x in got:
        bump(x)
    bad = [i for i, (h, x) in enumerate(got) if not torch.equal(x.cpu(), h + 1)]
    assert not bad, f'samples {bad[:8]} wrong (of {n}), at {[hex(addrs[i]) for i in bad[:4]]}'
    if v2:
        banks=w['allocation']['physical_banks']
        spans=[(a-local+w['base'],a-local+w['base']+512) for a in addrs]
        hits=[[b['pc'] for b in banks if lo<b['allocated_base']+b['allocated_size']
               and hi>b['allocated_base']] for lo,hi in spans]
        pcs=sorted({pc for group in hits for pc in group})
        wanted=sorted(b['pc'] for b in banks)
        assert pcs==wanted and any(len(group)>1 for group in hits), 'allocated-PC boundary coverage incomplete'
        return dict(samples=n,physical_pcs=pcs,configured_pc_coverage=True,
                    boundary_crossings=[dict(offset=offsets[i],pcs=group) for i,group in enumerate(hits) if len(group)>1],
                    topology_sha256=w['allocation']['topology_sha256'])
    pcs = sorted({(a - local + w['base']) // PC_BYTES for a in addrs})
    return dict(samples=n, first=hex(addrs[0]), last=hex(addrs[-1]),
                pseudo_channels=len(pcs), pc_first=pcs[0], pc_last=pcs[-1],
                all_pc_coverage=pcs == list(range(32)))


@case
def capacity():
    torch.manual_seed(0)
    n_out, k = (256, 128) if SMALL else (8192, 4096)    # 64 MiB of fp16 weight
    w = torch.randn(n_out, k, dtype=torch.float16) * 0.02
    x = torch.randn(1, k, dtype=torch.float16)
    dw, dx = w.to('tnca'), x.to('tnca')
    f = torch.compile(lambda a, b: a @ b.t(), fullgraph=True)
    c0 = _counters()
    y, cyc, launches = _launch_cycles(lambda: f(dx, dw))
    d = _delta(c0, _counters())
    ref = (x.float() @ w.float().t())
    err = ((y.cpu().float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    assert err < 2e-2, f'relative error {err}'
    nbytes = w.numel() * 2
    if c0 is not None:
        # (an image before AXI_RBEAT reads it as zero; its bridge is single-beat)
        beats = d['read_beats'] or d['reads']
        assert beats * 8 >= nbytes, f'{beats} AXI read beats cannot have carried {nbytes} bytes'
    return dict(working_set=nbytes + x.numel() * 2 + n_out * 2, rel_err=err, launches=launches,
                device_cycles=cyc, axi=d,
                read_bytes_per_cycle=round((d['read_beats'] or d['reads']) * 8 / max(cyc, 1), 4),
                mean_read_latency=round(d['read_cycles'] / max(d['reads'], 1), 2))


@case
def stream():
    n = (64 << 10) // 4 if SMALL else (96 << 20) // 4   # 96 MiB of fp32
    x = torch.randn(n).to('tnca')
    out = {}
    graphs = {
        'read': (torch.compile(lambda t: t.sum(), fullgraph=True), n * 4),
        'write': (torch.compile(lambda t: torch.full_like(t, 3.0), fullgraph=True), n * 4),
        'copy': (torch.compile(lambda t: t * 1.5, fullgraph=True), n * 8),
    }
    for name, (g, nbytes) in graphs.items():
        g(x)                                            # compile outside the measurement
        c0 = _counters()
        y, cyc, launches = _launch_cycles(lambda: g(x))
        d = _delta(c0, _counters())
        host_x = x.cpu()
        ref = {'read': lambda: host_x.sum(), 'write': lambda: torch.full_like(host_x, 3.0),
               'copy': lambda: host_x * 1.5}[name]()
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-2)
        out[name] = dict(bytes=nbytes, device_cycles=cyc, launches=launches, axi=d,
                         bytes_per_cycle=round(nbytes / max(cyc, 1), 4),
                         gb_per_s=round(nbytes / max(cyc, 1) * CLOCK_HZ / 1e9, 4),
                         mean_read_latency=round(d['read_cycles'] / max(d['reads'], 1), 2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    ap.add_argument('--evidence', action='store_true')
    ap.add_argument('--small', action='store_true', help='dry run at simulator sizes')
    args = ap.parse_args(argv)
    global SMALL
    SMALL = args.small
    if SMALL and args.evidence:
        ap.error('--small is a dry run; it records no evidence')
    D.register()
    results = []
    for fn in CASES:
        if args.only and fn.__name__ not in args.only:
            continue
        t0 = time.time()
        try:
            detail = fn()
            ok = True
        except Exception as e:                           # noqa: BLE001
            ok, detail = False, f'{type(e).__name__}: {e}'
            traceback.print_exc()
        results.append(dict(name=fn.__name__, ok=ok, detail=detail, wall_s=round(time.time() - t0, 1)))
        print(f"  {'PASS' if ok else 'FAIL'}  {fn.__name__:<10s} {json.dumps(detail, default=str)[:400]}",
              flush=True)
    passed = sum(r['ok'] for r in results)
    print(f"r40-hbm-board  {'PASS' if passed == len(results) else 'FAIL'}  ({passed}/{len(results)})")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(results=results, backend_args=os.environ.get(
            'TNCA_BACKEND_ARGS'), hbm_bytes=os.environ.get('TNCA_HBM_BYTES')), indent=1, default=str) + '\n')
    if args.evidence:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import evidence as EV
        need = [r for r in results if r['name'] in ('window', 'samples', 'capacity')]
        checks = [dict(name=f"board:{r['name']}", ok=r['ok'], detail=json.dumps(r['detail'], default=str)[:600])
                  for r in results]
        if len(need) == 3:
            if os.environ.get('R40_CONTRACT')=='v2':
                # Mapping/one graph alone cannot certify the inherited full
                # inventory, model peak +25% headroom or every active port.
                EV.record('R40-04.hbm','not-measured' if all(r['ok'] for r in need) else 'fail',checks,
                          detail='v2 allocation/boundary/64 MiB checkpoint; full-suite/model peak, all-port traffic and range/alias evidence still required')
                return 0 if passed==len(results) else 1
            sample = next(r for r in need if r['name'] == 'samples')
            full_map = sample['ok'] and sample['detail'].get('all_pc_coverage', False)
            checks.append(dict(name='board:all-32-PCs', ok=full_map,
                               detail='sampling must cover HBM[0:31], including the final PC'))
            EV.record('R40-04.hbm', 'capability-pass' if all(r['ok'] for r in need) and full_map else 'fail', checks,
                      detail='U280 HBM image: window, whole-window samples, >=64 MiB standard Torch graph')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
