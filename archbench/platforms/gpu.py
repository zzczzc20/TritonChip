"""GPU: torch.compile -> Inductor -> Triton (and the libraries Inductor calls).

Variants (archbench/PROTOCOL.md section 3):

    inductor-cudagraph   torch.compile(mode="reduce-overhead"): Inductor's
                         kernels replayed as a CUDA graph, so host launch
                         gaps are not device time. PRIMARY.
    inductor             torch.compile() default: same kernels, launched one
                         by one from Python (what a plain user gets).
    inductor-triton      max-autotune with GEMMs restricted to Triton
                         templates: every kernel is Triton, the same code
                         generator TNCA receives (no cuBLAS).

Timing (PROTOCOL.md section 3), every figure the median of N cold calls:

    device   the sum of the durations of every device activity (kernels,
             memsets, copies) one call of the compiled module runs, from
             CUPTI through torch.profiler. PRIMARY, and the counterpart of
             TNCA's sum of launch ROI cycles: neither counts the host gaps
             between launches.
    latency  CUDA events around the call: device-side latency including the
             gaps where the GPU waited for the host to launch the next thing.
             Reported, never used for eta.

Cold: L2 is overwritten by a 2x-L2 buffer before every timed call, so the
required bytes really come from DRAM and the DRAM term of the bound is a
valid necessary condition. Warm latency (no flush) is reported beside it.
IEEE fp32 throughout (no TF32).
"""
import statistics
import time

from ..bounds import VERSION, gpu_bound, gpu_spec
from .common import base_record, classify, compare, error_text, flatten, ieee_fp32, reference

VARIANTS = ('inductor-cudagraph', 'inductor', 'inductor-triton')


def _device_info(torch):
    p = torch.cuda.get_device_properties(0)
    info = dict(name=p.name, sm_count=p.multi_processor_count, l2_bytes=p.L2_cache_size,
                capability=f'{p.major}.{p.minor}', cuda=torch.version.cuda)
    try:
        import subprocess
        q = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,clocks.sm,clocks.max.sm,'
                            'temperature.gpu,power.draw', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        info['nvidia_smi'] = q
    except Exception:                                        # noqa: BLE001
        info['nvidia_smi'] = None
    return info


def _compile(torch, m, variant):
    import torch._inductor.config as cfg
    cfg.max_autotune_gemm = variant == 'inductor-triton'
    cfg.max_autotune_gemm_backends = 'TRITON' if variant == 'inductor-triton' else 'ATEN,TRITON'
    mode = 'reduce-overhead' if variant == 'inductor-cudagraph' else None
    return torch.compile(m, mode=mode)


def _time(torch, fn, inputs, reps, flush):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        start.record()
        fn(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e-3)
    return samples


def _summary(samples):
    s = sorted(samples)
    return dict(median=statistics.median(s), min=s[0], p90=s[min(len(s) - 1, int(0.9 * len(s)))],
                n=len(s))


def run(point, *, variant='inductor-cudagraph', seed=0, reps=None, warmup=3, spec='gb10',
        ref_device=None):
    import torch
    if variant not in VARIANTS:
        raise ValueError(f'variant {variant!r}; one of {VARIANTS}')
    info = _device_info(torch)
    rec = base_record(point, dict(kind='gpu', name=f'gpu:{spec}', device=info, variant=variant,
                                  spec=spec), 'measured-gpu')
    reps = reps or (20 if point.tier == 'large' else 50)
    ref_device = ref_device or ('cuda' if point.tier == 'large' else 'cpu')
    try:
        ref = reference(point, seed, device=ref_device)
    except Exception as e:                                   # noqa: BLE001
        rec.update(status='not-measured', problems=[f'reference failed: {error_text(e)}'])
        return rec

    torch._dynamo.reset()
    m, inputs = point.make(seed)
    m = m.eval().cuda()
    inputs = tuple(t.cuda() for t in inputs)
    if variant == 'inductor-cudagraph':
        # The benchmark reuses the same input tensors every call; saying so
        # keeps CUDA graphs from copying them into static buffers each
        # replay (a KV cache copy is not part of a decode step).
        for t in inputs:
            torch._dynamo.mark_static_address(t)
    # Twice the write-back capacity (L2 + system cache), so nothing the
    # point reads survives from the previous call.
    wb = gpu_spec(spec).get('caches', {}).get('write_back_bytes', info['l2_bytes'])
    flush = torch.empty(2 * max(wb, info['l2_bytes']) // 4, dtype=torch.float32, device='cuda')
    t0 = time.monotonic()
    try:
        with torch.no_grad(), ieee_fp32():
            fn = _compile(torch, m, variant)
            for _ in range(warmup):
                out = fn(*inputs)
            torch.cuda.synchronize()
            got = tuple(t.detach().clone() for t in flatten(out))
            cold = _time(torch, fn, inputs, reps, flush)
            warm = _time(torch, fn, inputs, reps, None)
            device, kernels = _device_time(torch, fn, inputs, max(5, reps // 2), flush)
    except Exception as e:                                   # noqa: BLE001
        rec.update(status=classify(e), problems=[error_text(e)])
        return rec
    cmp = compare(got, ref, point.tolerance)
    rec['correctness'] = dict(max_rel_err=cmp['max_rel_err'], tol=point.tolerance, detail=cmp['detail'])
    c, w, d = _summary(cold), _summary(warm), _summary(device)
    rec['time'] = dict(unit='s', device=d['median'], timing='cupti-device-activity-sum-cold-l2',
                       device_stats=d, latency_cold=c, latency_warm=w, activities_per_call=kernels)
    rec['host'] = dict(wall_s=round(time.monotonic() - t0, 2))
    rec['status'] = 'pass' if cmp['ok'] else 'numeric-error'
    rec['problems'] = [] if cmp['ok'] else [cmp['detail']]
    L, binding, terms = gpu_bound(point, spec)
    rec['bound'] = dict(version=VERSION, unit='s', value=L, binding=binding, terms=terms,
                        peak={k: v['value'] for k, v in gpu_spec(spec)['rates'].items()})
    if rec['status'] == 'pass':
        rec['eta'] = L / d['median']
        if rec['eta'] > 1:
            rec['problems'].append(f'AUDIT: eta {rec["eta"]:.4f} > 1 -- a measurement below its strict bound')
    return rec


def _device_time(torch, fn, inputs, reps, flush):
    """Per call: seconds of device activity (CUPTI), and how many activities."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    out, count = [], None
    for _ in range(reps):
        flush.zero_()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(*inputs)
            torch.cuda.synchronize()
        acts = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
        out.append(sum(e.time_range.elapsed_us() for e in acts) * 1e-6)
        count = len(acts)
    return out, count
