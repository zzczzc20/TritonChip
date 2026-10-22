#!/usr/bin/env python3
"""Standard Torch on the native model with an 8 GiB device window (one case per process).

torch.compile(backend='inductor', fullgraph=True) graphs execute through the
tnca device, the Session and the systemc-tlm-grid backend; the CPU computes
every reference independently. The device window and its BO/PC map come from
the environment (TNCA_TLM_MEMORY_BYTES, TNCA_DRAM_MAP, TNCA_DRAM_PATH). Tensor
placement uses only standard empty storages (no data written) to move the
Session heap's first-fit cursor; every placement is read back from the
storage and translated to its physical PC independently of the model.

Cases:
  placement  tensors above 4 GiB (PC16) and in the last MiBs of the window (PC31)
  crossing   a storage stored by one graph and loaded by another across the
             32-bit logical boundary and the physical PC15/PC16 boundary
  gemm       mm/addmm with operands and outputs in the upper 4 GiB
  stream     one storage over 2 GiB at the window top: read+reduce, and a copy
             whose output crosses 4 GiB; bounded host chunks with independent CPU references
  refusal    allocation past the window, the TMU page, an upload past the window

Development evidence on an uncalibrated model; never a board record.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
                str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]
MiB = 1 << 20
GiB = 1 << 30


def kib(path, field):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(field + ':'):
                return int(line.split()[1])
    except OSError:
        pass
    return None


class Sampler(threading.Thread):
    """Peak RSS of the model process and of its forked slow-path passes, and
    their combined PSS (shared copy-on-write pages counted once)."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.stop_flag = period, threading.Event()
        self.pid = None
        self.peak = dict(model_rss_kib=0, child_rss_kib=0, model_plus_children_pss_kib=0,
                         host_available_min_kib=None, samples=0, python_rss_kib=0, total_pss_kib=0)

    def run(self):
        while not self.stop_flag.wait(self.period):
            self.sample()

    def sample(self):
        p = self.peak
        available = kib('/proc/meminfo', 'MemAvailable')
        if available is not None:
            p['host_available_min_kib'] = min(available, p['host_available_min_kib'] or available)
        if not self.pid:
            return
        rss = kib(f'/proc/{self.pid}/status', 'VmRSS')
        if rss is None:
            return
        p['python_rss_kib'] = max(p['python_rss_kib'], kib(f'/proc/{os.getpid()}/status', 'VmRSS') or 0)
        p['samples'] += 1
        p['model_rss_kib'] = max(p['model_rss_kib'], rss)
        try:
            children = [int(c) for c in Path(f'/proc/{self.pid}/task/{self.pid}/children').read_text().split()]
        except OSError:
            children = []
        pss = kib(f'/proc/{self.pid}/smaps_rollup', 'Pss') or 0
        for child in children:
            p['child_rss_kib'] = max(p['child_rss_kib'], kib(f'/proc/{child}/status', 'VmRSS') or 0)
            pss += kib(f'/proc/{child}/smaps_rollup', 'Pss') or 0
        p['model_plus_children_pss_kib'] = max(p['model_plus_children_pss_kib'], pss)
        p['total_pss_kib'] = max(p['total_pss_kib'], pss + (kib(f'/proc/{os.getpid()}/smaps_rollup', 'Pss') or 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True, choices=('placement', 'crossing', 'gemm', 'stream', 'refusal'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--map-audit', type=Path, required=True, help='report.json of the map in TNCA_DRAM_MAP')
    ap.add_argument('--stream-bytes', type=int, default=(1 << 31) + (1 << 26))
    ap.add_argument('--stream-chunk-bytes', type=int, default=64 * MiB)
    ap.add_argument('--resident-extra-bytes', type=int, default=0, help='additional uploaded device residency, separate from graph footprint')
    args = ap.parse_args()
    if args.stream_bytes <= 0 or args.stream_bytes % 64 or args.stream_chunk_bytes <= 0 or args.stream_chunk_bytes % 64 or args.resident_extra_bytes < 0 or args.resident_extra_bytes % 64:
        ap.error('stream/chunk sizes must be positive multiples of 64; extra residency a nonnegative multiple')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('TNCA_TLM_OUT', str(out/'model'))
    os.environ.update(TRITON_CACHE_DIR=str(out/'triton-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    report = dict(schema='tnca-hbm8-window-case/1', case=args.case, status='running', evidence='development-only',
                  execution="torch.compile(backend='inductor', fullgraph=True) on tnca -> systemc-tlm-grid",
                  checks=[], placements={}, graphs=[], phases={})
    started = time.monotonic()
    sampler = Sampler()
    sampler.start()

    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def phase(name, t0):
        report['phases'][name] = round(time.monotonic() - t0, 3)
        save()

    save()
    model = None
    try:
        import torch
        import triton                                                    # noqa: F401
        import tnca
        from tnca.runtime import DEVICE_RESERVED_RANGES
        from tnca.triton_backend.driver import launch_log
        from archbench.provenance import software_tree
        import tnca.torch_device as device
        import tlm_backend
        from dram_address_map import translate
        if not Path(tnca.__file__).resolve().is_relative_to(ROOT):
            raise ValueError('tnca imported outside the requested source tree')
        audit = json.loads(args.map_audit.read_text())
        mapping = audit['mapping']
        model_class = tlm_backend.register()
        if os.environ.get('TNCA_BACKEND') != 'systemc-tlm-grid':
            raise ValueError('TNCA_BACKEND=systemc-tlm-grid required')
        report.update(software_tree=software_tree(ROOT, refresh=True),
                      toolchain=dict(torch=str(torch.__version__), triton=str(triton.__version__)),
                      knobs={k: v for k, v in sorted(os.environ.items()) if k.startswith('TNCA_')},
                      map=dict(path=os.environ.get('TNCA_DRAM_MAP'), sha256=audit['map_sha256'], status=audit['status']))
        device.register()
        t0 = time.monotonic()
        s = device.MEMORY.session()
        model = model_class.instances[-1]
        sampler.pid = model.proc.pid
        window = model.memory_bytes
        local, bo_top = mapping['local_limit'], mapping['local_limit'] + mapping['bo_size']
        top = min(window, bo_top)
        report['window'] = dict(model_memory_bytes=window, local_limit=local, bo_top=bo_top, effective_top=top,
                                heap=[s._heap.base, s._heap.limit], heap_reserved=[list(r) for r in s._heap.reserved])
        if window != 0x200000000 or window > bo_top:
            raise ValueError(f'exact 8 GiB model window required, got {window}; BO top {bo_top}')
        phase('session_start', t0)
        pads = []

        def pad_to(target):
            """Move the heap's first-fit cursor to `target` with empty storages."""
            while s._heap.next < target:
                n = target - s._heap.next
                for lo, _hi in DEVICE_RESERVED_RANGES:
                    if s._heap.next < lo < s._heap.next + n:
                        n = lo - s._heap.next
                pads.append(torch.empty(n // 64 * 64 or 64, dtype=torch.uint8, device='tnca'))

        def where(name, t):
            base = device.storage_address(t) + t.storage_offset() * t.element_size()
            n = t.numel() * t.element_size()
            first, last = translate(mapping, base & ~63), translate(mapping, (base + n - 1) & ~63)
            report['placements'][name] = dict(logical_first=base, logical_end=base + n, bytes=n,
                                              physical_first=first['physical_address'], first_pc=first['pc'],
                                              last_pc=last['pc'], end_below_top_bytes=top - (base + n))
            save()
            return base, base + n

        def check(name, ok, **detail):
            report['checks'].append(dict(name=name, status='pass' if ok else 'fail', **detail))
            save()
            if not ok:
                raise AssertionError(f'{name}: {detail}')

        def graph(name, fn, *inputs):
            before = model.counters_snapshot()
            t = time.monotonic()
            with launch_log(isolated=True) as launches:
                result = fn(*inputs)
                device.MEMORY.synchronize()
            wall = time.monotonic() - t
            after = model.counters_snapshot()
            delta = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
            report['graphs'].append(dict(
                graph=name, host_wall_seconds=round(wall, 3), launches=len(launches),
                kernels=[e['kernel'] for e in launches],
                launch_detail=[dict(kernel=e['kernel'], grid=list(e['grid']), device_cycles=e.get('device_cycles'),
                                    host_seconds=e.get('host_seconds'), resident_bytes=e.get('resident_bytes'))
                               for e in launches if e['kernel'] != '_copy_kernel' or e.get('resident_bytes', 0) > 64],
                device_cycles=sum(e.get('device_cycles') or 0 for e in launches),
                host_transfer_bytes=sum(e.get('upload_bytes', 0) + e.get('download_bytes', 0) for e in launches),
                pc_bytes={k: v for k, v in sorted(delta.items(), key=lambda kv: (len(kv[0]), kv[0]))
                          if k.startswith('pc_') and k.endswith('_bytes')},
                dram={k: v for k, v in delta.items() if k.startswith('dram_')}))
            save()
            return result

        compile_ = lambda f: torch.compile(f, backend='inductor', fullgraph=True, dynamic=False)
        boundary = local + (1 << 32)               # physical 4 GiB: PC15 | PC16
        torch.manual_seed(41)

        if args.case == 'placement':
            n = MiB // 4
            a_cpu, b_cpu = torch.randn(n), torch.randn(n)
            pad_to(boundary + MiB)
            a = a_cpu.to('tnca')
            pad_to(top - 8 * MiB)
            b = b_cpu.to('tnca')
            fn = compile_(lambda x, y: x * 2.0 + y)
            o = graph('a*2+b', fn, a, b)
            (a0, _), (b0, _), (o0, o1) = where('a', a), where('b', b), where('out', o)
            check('a above 4 GiB (logical and physical)', a0 >= boundary and report['placements']['a']['first_pc'] == 16)
            check('b and output in the window top 8 MiB (PC31)', b0 >= top - 8 * MiB and o1 <= top and
                  report['placements']['out']['last_pc'] == 31)
            got = o.cpu()
            check('a*2+b equals the CPU reference bitwise', torch.equal(got, a_cpu * 2.0 + b_cpu))
            fn2 = compile_(lambda x: (x.sum(), x.amax(), x.amin()))
            total, high, low = graph('reduce out', fn2, o)
            ref = (a_cpu * 2.0 + b_cpu)
            check('amax/amin exact, sum within 1e-4 of float64', float(high.cpu()) == float(ref.max()) and
                  float(low.cpu()) == float(ref.min()) and
                  abs(float(total.cpu()) - float(ref.double().sum())) <= 1e-4 * float(ref.abs().double().sum()),
                  sum=float(total.cpu()), reference=float(ref.double().sum()))

        elif args.case == 'crossing':
            n = 16 * MiB // 4
            x_cpu = torch.randint(-1000, 1000, (n,)).float()
            pad_to(3 * GiB)
            x = x_cpu.to('tnca')
            where('x', x)
            pad_to((1 << 32) - 8 * MiB)
            g1 = compile_(lambda t: t * 3.0 + 1.0)
            y = graph('store y=3x+1 across 4 GiB', g1, x)
            y0, y1 = where('y', y)
            check('y crosses the 32-bit logical boundary and physical PC15/PC16', y0 < (1 << 32) and y0 < boundary < y1 and
                  report['placements']['y']['first_pc'] == 15 and report['placements']['y']['last_pc'] == 16)
            g2 = compile_(lambda t: ((t - 1.0) / 3.0, t.amax(), t.amin()))
            z, high, low = graph('load y across 4 GiB', g2, y)
            where('z', z)
            y_ref = x_cpu * 3.0 + 1.0
            check('stored y equals 3x+1 bitwise', torch.equal(y.cpu(), y_ref))
            check('loaded z=(y-1)/3 equals x bitwise', torch.equal(z.cpu(), x_cpu))
            check('amax/amin of y exact', float(high.cpu()) == float(y_ref.max()) and float(low.cpu()) == float(y_ref.min()))
            pcs = report['graphs'][-2]['pc_bytes']
            check('store graph wrote PC15 and PC16', pcs.get('pc_15_bytes', 0) > 0 and pcs.get('pc_16_bytes', 0) > 0, pc_bytes=pcs)

        elif args.case == 'gemm':
            os.environ.setdefault('TNCA_GEMM_TILE', '16,32,16')
            from fragment_case import per_k_fma_reference, bit_compare
            pad_to(local + 5 * GiB)
            rows = []
            for name, (m, nn, k), dtype in (('mm', (96, 80, 160), torch.bfloat16), ('addmm', (33, 31, 65), torch.float16)):
                a_cpu = (torch.randn(m, k) * .1).to(dtype)
                b_cpu = (torch.randn(k, nn) * .1).to(dtype)
                c_cpu = (torch.randn(nn) * .1).to(dtype)
                a = a_cpu.to('tnca')
                pad_to(max(s._heap.next, top - 64 * MiB))
                b = b_cpu.to('tnca')
                c = c_cpu.to('tnca')
                fn = compile_((lambda p, q, r: p @ q) if name == 'mm' else (lambda p, q, r: torch.addmm(r, p, q)))
                o = graph(f'{name} {m}x{nn}x{k} {dtype}', fn, a, b, c)
                for label, t in (('A', a), ('B', b), ('bias', c), ('out', o)):
                    where(f'{name}.{label}', t)
                check(f'{name} operands and output in the upper 4 GiB',
                      all(report['placements'][f'{name}.{l}']['first_pc'] >= 16 for l in ('A', 'B', 'bias', 'out')))
                ref = (a_cpu.float() @ b_cpu.float()) + (c_cpu.float() if name == 'addmm' else 0)
                got = o.cpu()
                err = float((got.float() - ref).abs().max())
                check(f'{name} within rtol/atol 2e-3 of the CPU FP32 reference',
                      torch.allclose(got.float(), ref, rtol=2e-3, atol=2e-3), max_abs_error=err)
                exact = per_k_fma_reference(a_cpu, b_cpu, c_cpu if name == 'addmm' else None)
                rows.append(dict(graph=name, per_k_fma_bitwise=bit_compare(got, exact)))
                pad_to(s._heap.next + MiB)
            report['informational_bitwise'] = rows

        elif args.case == 'stream':
            nbytes = args.stream_bytes
            n = nbytes // 4
            extra = None
            if args.resident_extra_bytes:
                extra = torch.empty(args.resident_extra_bytes // 4, dtype=torch.float32, device='tnca')
                for start in range(0, extra.numel(), 4 * MiB):
                    cpu = torch.ones(min(4 * MiB, extra.numel() - start))
                    extra[start:start + cpu.numel()].copy_(cpu)
                del cpu
                where('resident_extra', extra)
            hold_at = boundary - nbytes // 2 // MiB * MiB
            pad_to(hold_at)
            hold = torch.empty(nbytes + 4 * MiB, dtype=torch.uint8, device='tnca')
            pad_to(top - nbytes - 2 * MiB)
            t0 = time.monotonic()
            x = torch.empty(n, dtype=torch.float32, device='tnca')
            # Bounded CPU staging: one large device storage, no host numerical
            # fallback and no simultaneous multi-GiB CPU input/output copies.
            chunk = 16 * MiB // 4
            rng = torch.Generator().manual_seed(41)
            ref_sum, ref_max, ref_min = 0.0, -float('inf'), float('inf')
            for start in range(0, n, chunk):
                cpu = torch.rand(min(chunk, n - start), generator=rng)
                ref_sum += float(cpu.numpy().sum(dtype='float64'))
                ref_max, ref_min = max(ref_max, float(cpu.max())), min(ref_min, float(cpu.min()))
                x[start:start + cpu.numel()].copy_(cpu)
            del cpu
            phase('streamed_upload_and_cpu_reference', t0)
            report['upload'] = dict(bytes=nbytes, chunk_bytes=chunk * 4, single_storage=True)
            del hold
            x0, x1 = where('x', x)
            report['development_size'] = nbytes <= 1 << 31
            check('x is one storage ending in the window top 4 MiB',
                  x1 <= top and top - x1 <= 4 * MiB and report['placements']['x']['last_pc'] == 31)
            if not report['development_size']:
                check('x is over 2 GiB (byte offsets beyond 2^31)', nbytes > 1 << 31, bytes=nbytes)
            # The device storages exceed 2 GiB; bound each model command's
            # demand log and Ramulator replay table by streaming views.
            block = args.stream_chunk_bytes // 4
            blocks = (n + block - 1) // block
            sums = torch.empty(blocks, device='tnca')
            maxima = torch.empty(blocks, device='tnca')
            minima = torch.empty(blocks, device='tnca')
            g1 = compile_(lambda t: (t.sum(), t.amax(), t.amin()))
            for i, start in enumerate(range(0, n, block)):
                total, high, low = graph(f'read+reduce x chunk {i}', g1, x[start:min(start + block, n)])
                sums[i:i + 1].copy_(total.reshape(1))
                maxima[i:i + 1].copy_(high.reshape(1))
                minima[i:i + 1].copy_(low.reshape(1))
            total, high, low = graph('combine reduction chunks',
                                    compile_(lambda a, b, c: (a.sum(), b.amax(), c.amin())), sums, maxima, minima)
            check('amax/amin exact; sum within 1e-4 of float64',
                  float(high.cpu()) == ref_max and float(low.cpu()) == ref_min and
                  abs(float(total.cpu()) - ref_sum) <= 1e-4 * abs(ref_sum), sum=float(total.cpu()), reference=ref_sum)
            y = torch.empty_like(x)
            g2 = compile_(lambda src, dst: dst.copy_(src))
            for i, start in enumerate(range(0, n, block)):
                end = min(start + block, n)
                graph(f'copy x to y chunk {i}', g2, x[start:end], y[start:end])
            y0, y1 = where('y', y)
            check('copy output crosses physical 4 GiB', y0 < boundary < y1)
            t0 = time.monotonic()
            rng = torch.Generator().manual_seed(41)
            mismatched_chunks = 0
            for start in range(0, n, chunk):
                cpu = torch.rand(min(chunk, n - start), generator=rng)
                got = y[start:start + cpu.numel()].cpu()
                mismatched_chunks += int(not torch.equal(got, cpu))
            del cpu, got
            phase('streamed_download_and_cpu_comparison', t0)
            check('copy equals independently regenerated CPU input bitwise', mismatched_chunks == 0,
                  compared_bytes=nbytes, mismatched_chunks=mismatched_chunks)
            report['touched_tensor_bytes'] = 2 * nbytes + args.resident_extra_bytes
            report['graph_tensor_bytes'] = 2 * nbytes
            report['stream_chunk_bytes'] = block * 4
            report['resident_extra_bytes'] = args.resident_extra_bytes
            pcs = {}
            for g in report['graphs']:
                for key, value in g['pc_bytes'].items():
                    pcs[key] = pcs.get(key, 0) + value
            if not report['development_size']:
                check('the stream touched every PC16-PC31', all(pcs.get(f'pc_{p}_bytes', 0) > 0 for p in range(16, 32)), pc_bytes=pcs)

        elif args.case == 'refusal':
            from tnca.errors import TncaError, DeviceLost
            free_top = s._heap.limit - s._heap.next
            try:
                torch.empty(free_top + MiB, dtype=torch.uint8, device='tnca')
                check('allocation past the window refused', False)
            except (torch.OutOfMemoryError, RuntimeError, TncaError) as error:
                check('allocation past the window refused', 'out of device memory' in str(error), error=str(error)[:300])
            hole_lo, hole_hi = DEVICE_RESERVED_RANGES[0]
            check('Session heap keeps the TMU page reserved in the 8 GiB window',
                  [list(r) for r in s._heap.reserved] == [[hole_lo, hole_hi]], reserved=[list(r) for r in s._heap.reserved])
            pad_to(hole_lo - MiB)
            t_cpu = torch.randn(2 * MiB // 4)
            t = t_cpu.to('tnca')
            t0_, t1_ = where('straddling-request', t)
            check('a storage that would straddle the TMU page is placed above it', t0_ >= hole_hi, address=t0_)
            o = graph('t*2 above the TMU page', compile_(lambda v: v * 2.0), t)
            check('t*2 equals the CPU reference bitwise', torch.equal(o.cpu(), t_cpu * 2.0))
            for lo, n in s._heap.free_list:
                check('no free block overlaps the TMU page', lo + n <= hole_lo or lo >= hole_hi, block=[lo, n])
            # Control: the exact model window's last 8 bytes transfer; BO top
            # is 128 KiB higher and is checked by test_hbm8_native.py.
            pattern = bytes(range(0xf8, 0x100))
            reply = s.dev.upload(window - 8, pattern)
            _, back = s.dev.download(window - 8, 8)
            check('the last 8 bytes below the model window top transfer both ways', reply.ok and back == pattern)
            # Last: a raw transfer past the window end. The model must refuse and
            # the backend report a lost device; nothing is substituted.
            try:
                s.dev.upload(window - 8, bytes(16))
                check('upload past the window refused by the model', False)
            except DeviceLost as error:
                check('upload past the window refused by the model',
                      'host access outside model memory window' in (error.detail or {}).get('stderr', '') or
                      'host access outside model memory window' in str(error), error=str(error)[:300])
            model = None

        if model is not None:
            report['model_identity'] = model.identity
            report['model_counters'] = {k: v for k, v in model.counters_snapshot().items()
                                        if k.startswith(('pc_', 'dram_', 'outlet_', 'chip_', 'host_'))}
            report['model_vmhwm_kib'] = kib(f'/proc/{model.proc.pid}/status', 'VmHWM')
            report['model_dir'] = str(model.dir)
        if software_tree(ROOT, refresh=True) != report['software_tree']:
            raise ValueError('source changed during the experiment')
        report['status'] = 'functional-pass'
    except Exception:
        report.update(status='fail', error=traceback.format_exc())
        raise
    finally:
        sampler.sample()
        sampler.stop_flag.set()
        report['host'] = dict(sampler.peak, python_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              case_wall_seconds=round(time.monotonic() - started, 3))
        save()
    print(f'hbm8 window case {args.case}: {report["status"]}', flush=True)


if __name__ == '__main__':
    main()
