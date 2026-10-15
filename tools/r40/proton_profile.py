#!/usr/bin/env python3
"""Real RTL Proton calibration/example, with paired device cycles and Chrome trace.

Diagnostic only: scope granularity sweeps do not constitute FPGA performance
or R40 acceptance. No new synthesis is needed. A is uninstrumented, B records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--iterations', type=int, default=128)
    ap.add_argument('--loop', action='store_true', help='scope encloses a runtime loop, not unrolled code')
    ap.add_argument('--pairs', type=int, default=6)
    ap.add_argument('--max-overhead', type=float, default=.01)
    ap.add_argument('--profile', default='core-vrf-sync-v5')
    args = ap.parse_args()
    if not 1 <= args.iterations <= 100_000 or args.pairs < 5:
        ap.error('iterations must be 1..100000; at least five pairs required')
    args.out.mkdir(parents=True, exist_ok=False)
    import triton  # plugin registration before runtime adapters
    from tnca import aot, artifact, device
    from tnca.runtime import Session
    from tnca.proton import chrome_trace, paired_overhead
    from tnca.costmodel import counter_diagnosis
    from tnca_cc.proton import lower_records
    from archbench.provenance import software_tree

    tree = software_tree(refresh=True)
    report = dict(status='running', performance_status='not-measured',
                  workload='integer dependent-chain calibration; not a production workload',
                  iterations=args.iterations, triton=triton.__version__)
    layouts, binaries, resources = {}, {}, {}
    try:
        device.prepare(args.profile)
        with Session(profile=args.profile, counters=True, host_timeout=600,
                     heap_base=0x40000) as session:
            output, capture = session.allocate(8), session.allocate(32)
            for mode in ('baseline', 'instrumented', 'plain'):
                def program(b, target):
                    out = b.sload(b.special('arg_base'))
                    if mode != 'plain':
                        b.proton_record('dependent-chain')
                    value = b.const(0)
                    if args.loop:
                        with b.loop(value, args.iterations):
                            pass
                    else:
                        for _ in range(args.iterations):
                            value = b.sbini('add', value, 1)
                    if mode != 'plain':
                        b.proton_record('dependent-chain', start=False)
                    b.sstore(out, value)
                    b.exit()
                    if mode != 'plain':
                        lowered, layout = lower_records(b.kernel, buffer_base=capture.addr,
                            buffer_bytes=capture.size, enabled=mode == 'instrumented')
                        b.kernel.__dict__.update(lowered.__dict__)
                        layouts[mode] = layout
                result = aot.compile_aot(program, kernel_name='proton_calibration',
                    signature={'out': '*i64'}, source_kind='tnca-ir', profile=args.profile,
                    options={'opt_level': 0, 'schedule': 0, 'rctx_spmd': 0, 'rctx_roles': 0},
                    output=str(args.out / f'{mode}.tnca'))
                _, segments = artifact.load(result.path)
                binaries[mode] = hashlib.sha256(segments['text']).hexdigest()
                resources[mode] = result.manifest.get('resources')
                if mode != 'plain':
                    binaries[mode + '_module'] = session.load(result.path)
            assert binaries['baseline'] == binaries['plain'], 'disabled records changed code'
            config = dict(profile=args.profile, target=result.manifest['target'],
                          backend='sim-file', iterations=args.iterations,
                          runtime_loop=args.loop, heap_base=0x40000)
            identity = dict(software_sha256=tree.removeprefix('files:'),
                config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                image_sha256=hashlib.sha256(Path(session.dev.bench).read_bytes()).hexdigest(),
                contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest(),
                clock_hz=100_000_000, backend='sim-file',
                baseline_code_sha256=binaries['baseline'],
                instrumented_code_sha256=binaries['instrumented'])
            report.update(identity=identity, configuration=config, resources=resources,
                          harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          clock_kind='virtual simulation clock; physical frequency not measured',
                          code_sha256={k: v for k, v in binaries.items() if not k.endswith('_module')})
            pairs, runs = [], []

            def run(mode, label):
                session.upload(output, bytes(8))
                session.upload(capture, bytes(capture.size))
                event = session.submit(binaries[mode + '_module'], {'out': output})
                session.wait(event)
                assert event.ok, event.error
                assert struct.unpack('<Q', session.download(output))[0] == args.iterations
                raw = session.download(capture)
                entry = dict(mode=mode, label=label, cycles=event.device_cycles,
                             counters=event.counters, device_timing=event.device_timing)
                runs.append(entry)
                if mode == 'instrumented':
                    trace = chrome_trace(raw, layouts[mode], identity=identity)
                    (args.out / 'capture.bin').write_bytes(raw)
                    (args.out / 'trace.chrome_trace').write_text(json.dumps(trace, indent=2)+'\n')
                # Preserve partial raw measurements on failure.
                (args.out/'runs.json').write_text(json.dumps(runs, indent=2)+'\n')
                return event.device_cycles

            for mode in ('baseline', 'instrumented'):
                run(mode, 'warmup')
            for i in range(args.pairs):
                order = ['baseline', 'instrumented'] if i % 2 == 0 else ['instrumented', 'baseline']
                measured = {mode: run(mode, f'pair-{i}') for mode in order}
                pairs.append(dict(order='AB' if i % 2 == 0 else 'BA', identity=identity,
                    baseline_cycles=measured['baseline'], instrumented_cycles=measured['instrumented'],
                    baseline_code_sha256=binaries['baseline'], instrumented_code_sha256=binaries['instrumented']))
            overhead = paired_overhead(pairs, identity=identity, max_relative_overhead=args.max_overhead)
            report.update(overhead=overhead, layout=layouts['instrumented'])
            trace_path = args.out/'trace.chrome_trace'
            trace = json.loads(trace_path.read_text())
            trace['metadata'].update(overhead_status=overhead['status'],
                median_relative_overhead=overhead['median_relative_overhead'],
                worst_absolute_relative_change=overhead['worst_absolute_relative_change'],
                raw_measurements='report.json and runs.json alongside this trace')
            trace_path.write_text(json.dumps(trace, indent=2)+'\n')
            last_base = next(r for r in reversed(runs) if r['mode'] == 'baseline')
            if last_base['counters'].get('available'):
                report['baseline_diagnosis'] = counter_diagnosis(last_base['counters']['programs'][0])
            assert software_tree(refresh=True) == tree, 'source changed during profiling'
            report['status'] = 'complete-diagnostic'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], overhead=report['overhead']['status'],
                         median=report['overhead']['median_relative_overhead'],
                         trace=str(args.out/'trace.chrome_trace'))), flush=True)


if __name__ == '__main__':
    main()
