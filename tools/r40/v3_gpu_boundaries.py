#!/usr/bin/env python3
"""R40-10 item 4: GPU timings at the boundaries TNCA's board records use.

    python tools/r40/v3_gpu_boundaries.py --variant inductor-cudagraph \
        --points-file spec_points.txt --out gpu_boundaries.jsonl

Supplementary to `python -m archbench.run --platform gpu` (which stays the
primary GPU record and is not modified). ArchBench's GPU record
(archbench/platforms/gpu.py) reports the CUPTI *sum* of one call's device
activities after an L2 flush, and CUDA-event latency. The TNCA board records
of the delivery identity (archbench/platforms/tnca.py, native device,
TNCA_TRANSFER_POLICY=resident) report, after one warm-up call and with no
cache flush:

    device-graph-makespan      first accepted launch to last device completion,
                               submission gaps included
    device-cycles-serial-grid  the sum of the launches' accept-to-idle cycles
    request-latency            host time.monotonic() around the call; module
                               and inputs are resident on the device and the
                               outputs stay there (0 bytes up/down), so it is
                               the resident-call-latency boundary

This harness builds, places and compiles the point exactly as gpu.run does
(same seed, .cuda() residency, mark_static_address for the CUDA-graph
variant, gpu._compile, 3 warm-up calls, IEEE fp32, the same flush buffer and
sample counts) and measures, warm (no flush, like TNCA) and cold (flush, like
ArchBench):

    cupti_sum    sum of device activity durations (== ArchBench time.device
                 when cold); counterpart of serial-grid
    cupti_span   first device activity start to last device activity end;
                 counterpart of device-graph-makespan
    resident_call_latency   time.monotonic() around fn(*inputs) and
                 torch.cuda.synchronize(); counterpart of TNCA's resident
                 request-latency
    request_latency  the same plus in-place host->device copies of every
                 input into the resident input buffers before the call and a
                 .cpu() of every output after it (TNCA's records have no
                 counterpart on the native path)

plus the CUDA kernel names of one call (which library ran) and nvidia-smi
power/clock samples during a sustained back-to-back loop.

Variant `inductor-triton-bigsm` is ArchBench's inductor-triton with
torch._inductor.utils.is_big_gpu forced to True: Inductor refuses Triton GEMM
templates on GPUs with fewer than 68 SMs (GB10 has 48), so the frozen variant
has no GEMM choice at all there. The override is recorded in every record.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'r40-v3-gpu-boundary/1'
VARIANTS = ('inductor-cudagraph', 'inductor-triton', 'inductor-triton-bigsm')
BIGSM_NOTE = ('torch._inductor.utils.is_big_gpu forced to return True: Inductor (utils.py, min_sms=68) '
              'otherwise disables Triton GEMM templates on this 48-SM GPU, and with '
              'max_autotune_gemm_backends=TRITON no GEMM choice remains')


def _summary(xs):
    s = sorted(xs)
    if not s:
        return None
    return dict(median=statistics.median(s), min=s[0], p90=s[min(len(s) - 1, int(0.9 * len(s)))],
                max=s[-1], n=len(s))


def _host_latency(torch, fn, inputs, reps, flush):
    out = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        t = time.monotonic()
        fn(*inputs)
        torch.cuda.synchronize()
        out.append(time.monotonic() - t)
    return out


def _request_latency(torch, fn, inputs, host_inputs, reps, flatten):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.monotonic()
        for d, h in zip(inputs, host_inputs):
            d.copy_(h)
        got = fn(*inputs)
        for o in flatten(got):
            o.cpu()
        torch.cuda.synchronize()
        out.append(time.monotonic() - t)
        got = None
    return out


def _cupti(torch, fn, inputs, reps, flush):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    rows, names = [], None
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(*inputs)
            torch.cuda.synchronize()
        acts = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
        if not acts:
            rows.append(dict(sum_s=None, span_s=None, activities=0))
            continue
        start = min(e.time_range.start for e in acts)
        end = max(e.time_range.end for e in acts)
        rows.append(dict(sum_s=sum(e.time_range.elapsed_us() for e in acts) * 1e-6,
                         span_s=(end - start) * 1e-6, activities=len(acts)))
        names = collections.Counter(e.name for e in acts)
    return rows, dict(names or {})


def _smi():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f'error: {type(e).__name__}'


def _sustained(torch, fn, inputs, seconds):
    """Back-to-back resident calls for `seconds`, nvidia-smi polled beside them."""
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append((round(time.monotonic(), 3), _smi()))
            stop.wait(0.1)
    before = _smi()
    th = threading.Thread(target=poll, daemon=True)
    torch.cuda.synchronize()
    th.start()
    t0, n = time.monotonic(), 0
    while n < 1 or time.monotonic() - t0 < seconds:
        fn(*inputs)
        torch.cuda.synchronize()
        n += 1
    dt = time.monotonic() - t0
    stop.set()
    th.join()
    parsed = []
    for t, q in samples:
        try:
            p, c, temp, u = (x.strip() for x in q.split(','))
            parsed.append(dict(t=t, power_w=float(p), sm_mhz=float(c), temp_c=float(temp), util=float(u)))
        except Exception:                                    # noqa: BLE001
            parsed.append(dict(t=t, raw=q))
    pw = [x['power_w'] for x in parsed if 'power_w' in x]
    return dict(seconds=round(dt, 4), calls=n, mean_call_s=dt / n, idle_before=before,
                power_w=_summary(pw), sm_mhz=_summary([x['sm_mhz'] for x in parsed if 'sm_mhz' in x]),
                samples=parsed,
                note='nvidia-smi power.draw as the driver reports it for GB10; not wall power')


def _kernel_class(name):
    n = name.lower()
    if n.startswith('triton') or 'triton_' in n:
        return 'triton'
    if 'memset' in n or 'memcpy' in n:
        return 'memory'
    if 'cudnn' in n or 'conv' in n:
        return 'cudnn-or-conv'
    if any(k in n for k in ('gemm', 'cutlass', 'cublas', 'nvjet', 'xmma', 'sm90_', 'sm100_', 'sm120_')):
        return 'cublas-or-cutlass'
    return 'other'


def measure(point_id, variant, seed=0, sustain_s=2.0):
    import torch
    from archbench import provenance, registry
    from archbench.bounds import gpu_spec
    from archbench.platforms import gpu
    from archbench.platforms.common import (classify, compare, environment, error_text, flatten,
                                            ieee_fp32, now, reference)
    p = registry.get(point_id)
    base = 'inductor-triton' if variant == 'inductor-triton-bigsm' else variant
    rec = dict(schema=SCHEMA, point_id=p.id, point_sha256=p.definition_sha256(), family=p.family,
               tier=p.tier, dtype=p.dtype, kind=p.kind, variant=variant, base_variant=base,
               override=None, status='not-measured', problems=[], correctness=None, timings=None,
               kernels=None, sustained=None, env=environment(), timestamp=now(),
               software_tree=provenance.software_tree(),
               harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if variant == 'inductor-triton-bigsm':
        import torch._inductor.utils as iu
        iu.is_big_gpu = lambda *a, **k: True
        rec['override'] = BIGSM_NOTE
    rec['device'] = gpu._device_info(torch)
    reps = 20 if p.tier == 'large' else 50                  # gpu.run's defaults
    prof_reps = max(5, reps // 2)
    warmup = 3
    ref_device = 'cuda' if p.tier == 'large' else 'cpu'
    rec['plan'] = dict(seed=seed, warmup=warmup, reps=reps, cupti_reps=prof_reps, ref_device=ref_device,
                       sustain_s=sustain_s)
    try:
        ref = reference(p, seed, device=ref_device)
    except Exception as e:                                   # noqa: BLE001
        rec['problems'] = [f'reference failed: {error_text(e)}']
        return rec
    torch._dynamo.reset()
    m, inputs = p.make(seed)
    m = m.eval().cuda()
    host_inputs = tuple(t.clone() for t in inputs)
    inputs = tuple(t.cuda() for t in inputs)
    if base == 'inductor-cudagraph':
        for t in inputs:
            torch._dynamo.mark_static_address(t)
    info = rec['device']
    wb = gpu_spec('gb10').get('caches', {}).get('write_back_bytes', info['l2_bytes'])
    flush = torch.empty(2 * max(wb, info['l2_bytes']) // 4, dtype=torch.float32, device='cuda')
    t0 = time.monotonic()
    try:
        with torch.no_grad(), ieee_fp32():
            fn = gpu._compile(torch, m, base)
            out = fn(*inputs)
            torch.cuda.synchronize()
            first = time.monotonic() - t0
            for _ in range(warmup - 1):
                out = fn(*inputs)
            torch.cuda.synchronize()
            got = tuple(t.detach().clone() for t in flatten(out))
            out = None
            res_warm = _host_latency(torch, fn, inputs, reps, None)
            res_cold = _host_latency(torch, fn, inputs, reps, flush)
            prof_warm, names = _cupti(torch, fn, inputs, prof_reps, None)
            prof_cold, _ = _cupti(torch, fn, inputs, prof_reps, flush)
            req = _request_latency(torch, fn, inputs, host_inputs, reps, flatten)
            sustained = _sustained(torch, fn, inputs, sustain_s)
            # The request loop rewrote the inputs with the same values; the
            # last outputs are checked against the reference once more.
            last = tuple(t.detach().clone() for t in flatten(fn(*inputs)))
            torch.cuda.synchronize()
    except Exception as e:                                   # noqa: BLE001
        rec.update(status=classify(e), problems=[error_text(e)])
        return rec
    cmp = compare(got, ref, p.tolerance)
    cmp_last = compare(last, ref, p.tolerance)
    rec['correctness'] = dict(max_rel_err=cmp['max_rel_err'], tol=p.tolerance, detail=cmp['detail'],
                              last_call_max_rel_err=cmp_last['max_rel_err'], last_call_ok=cmp_last['ok'])
    rec['status'] = 'pass' if cmp['ok'] and cmp_last['ok'] else 'numeric-error'
    if rec['status'] != 'pass':
        rec['problems'] = [cmp['detail'] or cmp_last['detail']]

    def col(rows, k):
        return [r[k] for r in rows if r[k] is not None]
    rec['timings'] = {
        'cupti-span-warm': dict(unit='s', **_summary(col(prof_warm, 'span_s')),
                                boundary='first device activity start to last device activity end, '
                                         'no flush (TNCA device-graph-makespan counterpart)'),
        'cupti-span-cold': dict(unit='s', **_summary(col(prof_cold, 'span_s')),
                                boundary='as cupti-span-warm, L2+system cache flushed before the call'),
        'cupti-sum-warm': dict(unit='s', **_summary(col(prof_warm, 'sum_s')),
                               boundary='sum of device activity durations, no flush '
                                        '(TNCA device-cycles-serial-grid counterpart)'),
        'cupti-sum-cold': dict(unit='s', **_summary(col(prof_cold, 'sum_s')),
                               boundary='ArchBench GPU time.device definition (cold L2)'),
        'resident-call-latency-warm': dict(unit='s', **_summary(res_warm),
                                           boundary='host time.monotonic() around fn(*inputs) + '
                                                    'torch.cuda.synchronize(); resident module/inputs, '
                                                    'outputs left on device, no flush'),
        'resident-call-latency-cold': dict(unit='s', **_summary(res_cold),
                                           boundary='as resident-call-latency-warm, flush before the call '
                                                    '(flush not timed)'),
        'request-latency': dict(unit='s', **_summary(req),
                                boundary='in-place host->device copy of every input, the call, .cpu() of '
                                         'every output, synchronize; no flush'),
    }
    rec['timings']['activities_per_call'] = prof_warm[-1]['activities'] if prof_warm else None
    rec['kernels'] = dict(names=names, classes=dict(collections.Counter(
        cls for n, c in names.items() for cls in [_kernel_class(n)] * c)))
    rec['sustained'] = sustained
    rec['host'] = dict(first_call_s=round(first, 3), wall_s=round(time.monotonic() - t0, 2))
    # Which TNCA code this CUDA run could have used: the tnca package's
    # modules imported by the end of the run (Triton backend discovery may
    # import the driver; tnca.inductor.configure() is the TNCA path's only
    # Inductor configuration), and Triton's active backend.
    rec['tnca_modules_loaded'] = sorted(k for k in sys.modules if k == 'tnca' or k.startswith('tnca.'))
    try:
        import triton
        rec['triton_active_backend'] = triton.runtime.driver.active.get_current_target().backend
    except Exception as e:                                   # noqa: BLE001
        rec['triton_active_backend'] = f'unknown: {type(e).__name__}'
    return rec


def _child(args):
    rec = measure(args.point[0], args.variant, seed=args.seed, sustain_s=args.sustain)
    print('R40-GPU-RECORD ' + json.dumps(rec, default=str), flush=True)


def _spawn(pid, args, logdir):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--child', '--point', pid,
           '--variant', args.variant, '--seed', str(args.seed), '--sustain', str(args.sustain)]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'python'), env.get('PYTHONPATH', '')])
    log = logdir / (pid.replace('/', '__') + '.log')
    t = time.monotonic()
    rec = None
    try:
        with open(log, 'w') as fh:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=fh, text=True,
                                  timeout=args.timeout)
        for line in proc.stdout.splitlines():
            if line.startswith('R40-GPU-RECORD '):
                rec = json.loads(line[len('R40-GPU-RECORD '):])
        why = f'child exited {proc.returncode} without a record; see {log}'
        status = 'runtime-fault'
    except subprocess.TimeoutExpired:
        why, status = f'timeout after {args.timeout}s; see {log}', 'timeout'
    if rec is None:
        rec = dict(schema=SCHEMA, point_id=pid, variant=args.variant, status=status, problems=[why])
    rec['process_s'] = round(time.monotonic() - t, 1)
    rec['log'] = str(log)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=VARIANTS, required=True)
    ap.add_argument('--point', nargs='+')
    ap.add_argument('--points-file', type=Path)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sustain', type=float, default=2.0)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return _child(args)
    pts = list(args.point or [])
    if args.points_file:
        pts += [x.strip() for x in args.points_file.read_text().splitlines() if x.strip()]
    if not pts or not args.out:
        ap.error('points and --out are required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logdir = args.out.parent / (args.out.stem + '_logs')
    logdir.mkdir(parents=True, exist_ok=True)
    n_pass = 0
    with open(args.out, 'a') as fh:
        for pid in pts:
            rec = _spawn(pid, args, logdir)
            fh.write(json.dumps(rec, default=str) + '\n')
            fh.flush()
            n_pass += rec['status'] == 'pass'
            t = rec.get('timings') or {}
            span = (t.get('cupti-span-warm') or {}).get('median')
            lat = (t.get('resident-call-latency-warm') or {}).get('median')
            print(f'{rec["status"]:16s} {args.variant:22s} {pid:58s} span={span} lat={lat} '
                  f'[{rec["process_s"]}s]' + ''.join(f'\n      {q[:300]}' for q in rec.get('problems', [])[:2]),
                  flush=True)
    print(f'{n_pass}/{len(pts)} pass -> {args.out}')
    return 0 if n_pass == len(pts) else 1


if __name__ == '__main__':
    sys.exit(main())
