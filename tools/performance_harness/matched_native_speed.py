#!/usr/bin/env python3
"""Compare complete native SystemC and RTL execution of one identical artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]
import triton
import triton.language as tl


@triton.jit
def repeated_vector_add(X, Y, Z, STEPS: tl.constexpr):
    lanes = tl.arange(0, 16)
    value = tl.load(X+lanes)
    increment = tl.load(Y+lanes)
    for _ in range(STEPS):
        value = value+increment
    tl.store(Z+lanes, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--steps', type=int, default=64)
    args = ap.parse_args()
    if args.samples < 3 or not 1 <= args.steps <= 1024:
        ap.error('at least three samples and 1..1024 arithmetic steps are required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from tnca import aot, device
    from tnca.runtime import Session
    import tlm_backend
    tlm_backend.register()
    profile = 'core-vrf-sync-v5'
    build_started = time.monotonic()
    device.prepare(profile, verbose=False)
    build_seconds = time.monotonic()-build_started
    compile_started = time.monotonic()
    artifact = aot.compile_aot(repeated_vector_add, profile=profile,
        signature=dict(X='*fp32', Y='*fp32', Z='*fp32', STEPS='constexpr'),
        constexprs=dict(STEPS=args.steps),
        options=dict(opt_level=2, rctx_spmd=0, physical_chunks=0),
        output=str(out/'matched.tnca'))
    compile_seconds = time.monotonic()-compile_started
    blob = Path(artifact.path).read_bytes()
    reference = struct.pack('<16f', *(i*.5+args.steps*(2*i+1) for i in range(16)))
    inputs = [struct.pack('<16f', *(i*.5 for i in range(16))),
              struct.pack('<16f', *(2*i+1 for i in range(16)))]
    report = dict(schema='tnca-matched-native-speed/1', status='running',
        artifact_sha256=hashlib.sha256(blob).hexdigest(), profile=profile,
        workload=dict(lanes=16, arithmetic_steps=args.steps, grid=[1, 1, 1]),
        measurement_boundary='Session.submit through terminal Session.wait; identical uploads/load outside timing; one warmup then >=3 samples',
        build_or_locate_rtl_seconds=build_seconds, compile_seconds=compile_seconds,
        reference_sha256=hashlib.sha256(reference).hexdigest(), runs=[],
        physical_performance_gate='not-measured', model_accuracy='uncalibrated')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for backend in ('sim-socket', 'systemc-tlm'):
            started = time.monotonic()
            with Session(profile=profile, backend=backend,
                         deadline_cycles=1000000, host_timeout=120) as session:
                module = session.load(artifact.path)
                x, y, z = [session.allocate(64) for _ in range(3)]
                session.upload(x, inputs[0]); session.upload(y, inputs[1])
                row = dict(backend=backend, artifact_sha256=report['artifact_sha256'],
                           buffer_addresses=[x.addr, y.addr, z.addr],
                           startup_load_upload_seconds=time.monotonic()-started, samples=[])
                report['runs'].append(row)
                for index in range(args.samples+1):
                    session.upload(z, bytes([0xA5])*64)
                    started = time.perf_counter()
                    event = session.submit(module, dict(X=x, Y=y, Z=z))
                    session.wait(event)
                    elapsed = time.perf_counter()-started
                    if not event.ok:
                        raise RuntimeError(f'{backend} failed: {event.error}')
                    actual = session.download(z)
                    if actual != reference:
                        raise ValueError(f'{backend} output differs from the independent exact FP32 reference')
                    sample = dict(host_execution_seconds=elapsed, device_cycles=event.device_cycles,
                                  output_sha256=hashlib.sha256(actual).hexdigest())
                    if index == 0:
                        row['warmup'] = sample
                    else:
                        row['samples'].append(sample)
                    save()
        rtl, model = report['runs']
        if rtl['buffer_addresses'] != model['buffer_addresses']:
            raise ValueError('matched workload uses different device buffer placements')
        rtl_median = statistics.median(s['host_execution_seconds'] for s in rtl['samples'])
        model_median = statistics.median(s['host_execution_seconds'] for s in model['samples'])
        report.update(status='completed-matched-functional-execution',
            rtl_median_execution_seconds=rtl_median, systemc_median_execution_seconds=model_median,
            rtl_over_systemc_host_time_ratio=rtl_median/model_median,
            scope='one identical compiled arithmetic/memory workload on serial-grid RTL and native SystemC; not a full-chip workload speedup',
            model_binary_sha256=tlm_backend._sha(os.environ['TNCA_TLM_BINARY']),
            model_config_sha256=tlm_backend._sha(os.environ['TNCA_TLM_CONFIG']))
    except Exception as error:
        report.update(status='failed-preserved', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print(json.dumps({k: report[k] for k in ('status', 'rtl_over_systemc_host_time_ratio')}, indent=2))


if __name__ == '__main__':
    main()
