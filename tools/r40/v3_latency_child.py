#!/usr/bin/env python3
"""One ArchBench point with the two host latencies the R40 protocol defines.

    python tools/r40/v3_latency_child.py --point ID --profile P --warmup 5 --timed 30 \
        [--bundle-dir DIR]

archbench/PROTOCOL.md defines `resident-call-latency` (weights and inputs
resident; host monotonic time from the call until it is complete) and
`request-latency` (the same plus the explicit input copies to the device and
the output copies back). On the native device ArchBench's own timed call
starts from resident inputs and leaves the outputs on the device, so its
`request-latency` is in fact the resident call. This child runs the unchanged
ArchBench measurement (archbench.platforms.tnca.run: same compilation, cold
call, correctness, provenance and bundle) and only replaces the steady-call
loop: each timed call copies the inputs from host memory, calls, waits for
completion and copies every output back, timing both spans. Prints the record
as `ARCHBENCH-RECORD <json>` like `python -m archbench.run --child`.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]

CALLS = []


def steady(compiled, inputs, n, expected, launch_log, torch):
    from tnca.device_timing import clock_span
    tree_map = torch.utils._pytree.tree_map
    host_inputs = tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs)
    device_type = next((x.device.type for x in inputs if isinstance(x, torch.Tensor)), 'cpu')
    out = []
    for i in range(n):
        got = None
        t0 = time.monotonic()
        dev = tuple(x.to(device_type) if isinstance(x, torch.Tensor) else x for x in host_inputs)
        # Same completion boundary as ArchBench's own steady call: leaving
        # launch_log waits for queued events, so batched submission is timed
        # to completion.
        with torch.no_grad(), launch_log() as lg:
            t1 = time.monotonic()
            got = compiled(*dev)
        t2 = time.monotonic()
        tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, got)
        t3 = time.monotonic()
        c = dict(index=i, request_s=t3-t0, resident_call_s=t2-t1, input_copy_s=t1-t0,
                 output_copy_s=t3-t2, launches=len(lg), _launch_log=lg,
                 device_cycles=sum(e['device_cycles'] or 0 for e in lg),
                 upload_s=sum(e['host_seconds']['upload'] for e in lg),
                 run_s=sum(e['host_seconds']['run'] for e in lg),
                 download_s=sum(e['host_seconds']['download'] for e in lg))
        c['device_timing'] = clock_span([e.get('device_timing') for e in lg])
        if expected is not None and len(lg) != expected:
            c['problem'] = f'{len(lg)} launches, the wrapper makes {expected}'
        elif not c['device_cycles']:
            c['problem'] = 'no device time'
        if i == n - 1:
            c['out'] = got
        out.append(c)
    CALLS[:] = out
    return out


def _summary(samples, warmup, includes):
    s = sorted(samples)
    k = len(s)
    median = s[k//2] if k % 2 else 0.5*(s[k//2-1]+s[k//2])
    return dict(unit='s', samples=samples, median=median, p95=s[min(k-1, int(round(0.95*(k-1))))],
                warmup=warmup, includes=includes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--point', required=True)
    ap.add_argument('--profile', default='core-vrf-sync-v5')
    ap.add_argument('--opt-level', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warmup', type=int, required=True)
    ap.add_argument('--timed', type=int, required=True)
    ap.add_argument('--bundle-dir')
    a = ap.parse_args()
    if a.timed < 1:
        ap.error('host latencies need timed calls')
    from archbench import registry
    from archbench.platforms import tnca
    tnca._steady = steady
    p = registry.get(a.point)
    bundle = str(Path(a.bundle_dir)/p.id.replace('/', '__')) if a.bundle_dir else None
    rec = tnca.run(p, profile=a.profile, opt_level=a.opt_level, seed=a.seed,
                   warmup=a.warmup, timed=a.timed, bundle=bundle)
    timed = CALLS[a.warmup:]
    if rec.get('status') == 'pass':
        if len(timed) != a.timed or rec.get('host', {}).get('calls') is None:
            rec['status'] = 'not-measured'
            rec.setdefault('problems', []).append('v3 latency child: timed calls missing')
        else:
            rec['timings']['request-latency'] = _summary(
                [round(c['request_s'], 6) for c in timed], a.warmup,
                ['explicit input copies host->device', 'call to completion', 'explicit output copies device->host'])
            rec['timings']['resident-call-latency'] = _summary(
                [round(c['resident_call_s'], 6) for c in timed], a.warmup,
                ['call to completion with resident weights and inputs'])
            rec['host']['explicit_copies'] = [dict(input_s=round(c['input_copy_s'], 6),
                                                   output_s=round(c['output_copy_s'], 6)) for c in timed]
    rec['v3_latency_harness'] = dict(path='tools/r40/v3_latency_child.py',
                                     note='ArchBench run with explicit host copies around each timed call')
    print('ARCHBENCH-RECORD ' + json.dumps(rec, default=str), flush=True)


if __name__ == '__main__':
    main()
