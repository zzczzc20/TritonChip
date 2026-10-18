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
import hashlib
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

CASES = []
from hbm_address import PC_BYTES, probe_offsets, topology_probe_offsets, topology_probe_storage
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
    if dev.capabilities().get('physical_grid'):
        return dict(kind='outlet-lines', ports=dev.memory_port_metrics())
    counters = dev.axi_counters() if hasattr(dev, 'axi_counters') else None
    return dict(kind='legacy-axi', counters=counters) if counters is not None else None


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
        return None
    if a['kind'] != b['kind']:
        raise ValueError('traffic counter source changed within an experiment')
    if a['kind'] == 'outlet-lines':
        if not a['ports'] or [r['port'] for r in a['ports']] != [r['port'] for r in b['ports']]:
            raise ValueError('physical outlet identities changed')
        ports = [dict(port=x['port'], read_lines=y['read_lines']-x['read_lines'],
                      write_lines=y['write_lines']-x['write_lines'])
                 for x,y in zip(a['ports'],b['ports'])]
        if any(r[k]<0 for r in ports for k in ('read_lines','write_lines')):
            raise ValueError('outlet counters reset or decreased')
        # The public multicore line ABI carries 512-bit cache lines, not
        # eight-byte legacy bridge transactions. Latency is not exposed.
        return dict(kind=a['kind'], ports=ports, line_bytes=64,
                    read_bytes=sum(r['read_lines'] for r in ports)*64,
                    write_bytes=sum(r['write_lines'] for r in ports)*64,
                    mean_read_latency=None)
    delta = {k:b['counters'][k]-v for k,v in a['counters'].items()}
    if any(v<0 for v in delta.values()):
        raise ValueError('legacy AXI counters reset or decreased')
    return dict(kind=a['kind'], counters=delta,
                read_bytes=(delta['read_beats'] or delta['reads'])*8,
                write_bytes=None,
                mean_read_latency=delta['read_cycles']/delta['reads'] if delta['reads'] else None)


def _launch_cycles(fn):
    from tnca.triton_backend.driver import launch_log
    from tnca.device_timing import clock_span
    with torch.no_grad(), launch_log() as log:
        out = fn()
    # launch_log drains asynchronous events on exit. Do not perform a CPU
    # ownership handoff/read until the caller has captured device traffic.
    timing = clock_span([e.get('device_timing') for e in log])
    return out, dict(device_timing=timing,
                     device_cycles=timing['cycles'] if timing else None,
                     summed_kernel_cycles=sum(e['device_cycles'] or 0 for e in log),
                     launches=len(log), performance_status='diagnostic-only',
                     traffic_window='pre-call to drained completion; may include argument/control traffic outside launch timestamps')


def _rate(nbytes, cycles):
    return nbytes/cycles if nbytes is not None and cycles else None


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
    layout=(topology_probe_storage(w) if v2 else
            [dict(offset=o,storage_start=o,storage_bytes=512,view_offset=0) for o in offsets])
    control=None
    n = len(offsets)
    bump = torch.compile(lambda t: t.add_(1), fullgraph=True)
    held, got = [], []
    for i, entry in enumerate(layout):
        offset=entry['offset']
        # Do not query first-fit with a temporary after freeing the frame:
        # that would observe the reusable hole rather than the high watermark.
        with s._lock:top=s._heap.next
        if v2 and i==n-1:
            s.free(control);control=None
        gap = local + entry['storage_start'] - top
        assert gap >= 0, f'allocator passed probe {i}: {top:#x} > {local + offset:#x}'
        if gap:
            held.append(torch.empty(gap, dtype=torch.uint8, device='tnca'))
        storage=torch.empty(entry['storage_bytes']//4,dtype=torch.int32,device='tnca')
        held.append(storage)
        x=storage[entry['view_offset']//4:]
        assert x.numel()==128 and x.data_ptr()==local+offset,'probe allocation moved'
        # Distinct nonzero halves of each 64-bit bus word exercise aliasing;
        # address width is checked by the actual pointer, without requiring
        # unsupported vector-int64 arithmetic from the kernel compiler.
        lane=torch.arange(128,dtype=torch.int32)
        h=lane*1000003+i*7919+(1<<28)+(lane%2)*(1<<28)
        x.copy_(h)
        got.append((h, x))
        if v2 and i==0:control=s.allocate(65536,name='hbm-boundary-control-reserve')
    addrs = [x.data_ptr() for _, x in got]
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
                    first_virtual_address=addrs[0],last_virtual_address=addrs[-1],
                    high_32bit_virtual_address_tested=any(a>=1<<32 for a in addrs),
                    probe_storage=layout,launch_frame_reserve_bytes=65536,
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
    y, measurement = _launch_cycles(lambda: f(dx, dw))
    d = _delta(c0, _counters())
    ref = (x.float() @ w.float().t())
    err = ((y.cpu().float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    assert err < 2e-2, f'relative error {err}'
    nbytes = w.numel() * 2
    if d is not None:
        assert d['read_bytes'] >= nbytes, f'{d["read_bytes"]} device read bytes cannot carry {nbytes} bytes'
    return dict(working_set=nbytes + x.numel() * 2 + n_out * 2, rel_err=err,
                measurement=measurement, axi=d,
                request_read_bytes_per_graph_cycle=_rate(d['read_bytes'] if d else None,measurement['device_cycles']),
                mean_read_latency=d['mean_read_latency'] if d else None)


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
        _session().drain()                              # asynchronous warmups also complete first
        c0 = _counters()
        y, measurement = _launch_cycles(lambda: g(x))
        d = _delta(c0, _counters())
        host_x = x.cpu()
        ref = {'read': lambda: host_x.sum(), 'write': lambda: torch.full_like(host_x, 3.0),
               'copy': lambda: host_x * 1.5}[name]()
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=1e-2)
        out[name] = dict(logical_tensor_bytes=nbytes, measurement=measurement, axi=d,
                         logical_bytes_per_cycle=_rate(nbytes,measurement['device_cycles']),
                         request_read_bytes_per_graph_cycle=_rate(d['read_bytes'] if d else None,measurement['device_cycles']),
                         request_write_bytes_per_graph_cycle=_rate(d['write_bytes'] if d else None,measurement['device_cycles']),
                         mean_read_latency=d['mean_read_latency'] if d else None)
    return out


def main(argv=None):
    global torch, D
    import torch
    import triton
    import tnca.torch_device as D
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    ap.add_argument('--evidence', action='store_true')
    ap.add_argument('--small', action='store_true', help='dry run at simulator sizes')
    args = ap.parse_args(argv)
    if args.only and set(args.only)-{fn.__name__ for fn in CASES}:
        ap.error('--only contains an unknown case')
    if args.json and Path(args.json).exists():
        ap.error('--json must be a fresh path; preserve earlier attempts')
    global SMALL
    SMALL = args.small
    if SMALL and args.evidence:
        ap.error('--small is a dry run; it records no evidence')
    if args.evidence and os.environ.get('TNCA_BACKEND') not in ('fpga-u280','fpga-u280-multicore'):
        ap.error('--evidence requires an actual U280 backend')
    D.register()
    from archbench import provenance
    from archbench.platforms.tnca import knobs
    provenance._tree_cache.pop(provenance.ROOT, None)
    tree = provenance.software_tree()
    backend = os.environ.get('TNCA_BACKEND','sim-file')
    backend_args = json.loads(os.environ.get('TNCA_BACKEND_ARGS','{}'))
    image = backend_args.get('xclbin')
    identity = dict(software_tree=tree, backend=backend, backend_args=backend_args,
                    image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest() if image else None,
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    knobs=knobs(), torch=torch.__version__,triton=triton.__version__,
                    python=sys.version, contract=os.environ.get('R40_CONTRACT','v1'),
                    clock_hz=None,clock_qualification='requires matched image and qualified board clock reports',
                    performance_status='diagnostic-only; no FPGA bandwidth gate certification')
    from tnca.operating_clock import from_environment
    identity.update(from_environment(identity['image_sha256']))
    if identity['contract']=='v2':
        import contract_v2
        contract_v2.verify()
        identity['contract_sha256']=contract_v2.LOCK_SHA256
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
    provenance._tree_cache.pop(provenance.ROOT, None)
    unchanged=provenance.software_tree()==tree
    if not unchanged:
        results.append(dict(name='immutable-source',ok=False,detail='software changed during experiment'))
    passed = sum(r['ok'] for r in results)
    print(f"r40-hbm-board  {'PASS' if passed == len(results) else 'FAIL'}  ({passed}/{len(results)})")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(identity=identity,results=results,
            backend_args=os.environ.get('TNCA_BACKEND_ARGS'), hbm_bytes=os.environ.get('TNCA_HBM_BYTES')),
            indent=1, default=str) + '\n')
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
                EV.record('R40-04.hbm','not-measured' if unchanged and all(r['ok'] for r in need) else 'fail',checks,
                          detail='v2 allocation/boundary/64 MiB checkpoint; full-suite/model peak, all-port traffic and range/alias evidence still required')
                return 0 if passed==len(results) else 1
            sample = next(r for r in need if r['name'] == 'samples')
            full_map = sample['ok'] and sample['detail'].get('all_pc_coverage', False)
            checks.append(dict(name='board:all-32-PCs', ok=full_map,
                               detail='sampling must cover HBM[0:31], including the final PC'))
            EV.record('R40-04.hbm', 'capability-pass' if unchanged and all(r['ok'] for r in need) and full_map else 'fail', checks,
                      detail='U280 HBM image: window, whole-window samples, >=64 MiB standard Torch graph')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
