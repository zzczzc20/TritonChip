#!/usr/bin/env python3
"""Audit a diagnostic Proton bundle and optionally screen architecture resources.

Read-only inputs, fresh output, no synthesis, board programming, training on
holdout, automatic gate updates or inferred missing measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from tnca.proton import check_identity, chrome_trace, paired_overhead
from tnca.costmodel import counter_diagnosis, resource_screen, implementation_screen, CalibratedModel


def inspect_report(path):
    report = json.loads(path.read_text())
    if report.get('status') != 'complete-diagnostic':
        raise ValueError('only completed diagnostic reports can be interpreted; retain failures separately')
    identity = report['identity']
    check_identity(identity)
    overhead = paired_overhead(report['overhead']['pairs'], identity=identity,
                               max_relative_overhead=report['overhead']['max_relative_overhead'])
    for pair in overhead['pairs']:
        for mode in ('baseline', 'instrumented'):
            if pair[f'{mode}_code_sha256'] != report['code_sha256'][mode]:
                raise ValueError('overhead pair code does not match the captured report')
    raw_path, trace_path, runs_path = [path.parent/n for n in ('capture.bin', 'trace.chrome_trace', 'runs.json')]
    raw = raw_path.read_bytes()
    regenerated = chrome_trace(raw, report['layout'], identity=identity)
    saved_trace = json.loads(trace_path.read_text())
    if (saved_trace['traceEvents'] != regenerated['traceEvents']
            or saved_trace['metadata']['identity'] != identity):
        raise ValueError('Chrome trace does not match raw capture/layout/identity')
    runs = json.loads(runs_path.read_text())
    paired_runs = [r for r in runs if r['label'] != 'warmup']
    expected = []
    for index, pair in enumerate(overhead['pairs']):
        order = ('baseline', 'instrumented') if pair['order'] == 'AB' else ('instrumented', 'baseline')
        expected.extend((mode, f'pair-{index}', pair[f'{mode}_cycles']) for mode in order)
    if [(r['mode'], r['label'], r['cycles']) for r in paired_runs] != expected:
        raise ValueError('overhead pairs do not match raw run order and device cycles')
    diagnosis = []
    for run in runs:
        counters = run.get('counters') or {}
        if run['mode'] == 'baseline' and run['label'] != 'warmup' and counters.get('available'):
            diagnosis.append(dict(label=run['label'], programs=[
                counter_diagnosis(c) for c in counters['programs']]))
    result = dict(schema='tnca-performance-inspection/1', status='audited-diagnostic',
                  identity=identity, overhead=overhead,
                  scope_cycles=[e['args']['cycles'] for e in regenerated['traceEvents'] if e['ph'] == 'X'],
                  baseline_counter_diagnosis=diagnosis,
                  region_timing_error_bound_cycles=None, performance_gate='not-measured',
                  analysis_sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__).resolve(), ROOT/'python/tnca/proton.py', ROOT/'python/tnca/costmodel.py')},
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (path, raw_path, trace_path, runs_path)})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--cost-input', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = inspect_report(args.report)
    if args.cost_input:
        data = json.loads(args.cost_input.read_text())
        if data['identity'] != result['identity']:
            raise ValueError('cost input must bind exactly the same reference measurement identity')
        screens = {name: resource_screen(data['demand'], capacities,
                   identity=data['identity'], cycle_budget=data.get('cycle_budget'))
                   for name, capacities in data['candidates'].items()}
        for name, impl in data.get('implementation', {}).items():
            screens[name]['implementation'] = implementation_screen(
                impl['floors'], impl['available'], identity=data['identity'])
        calibration = data.get('calibration')
        if calibration:
            name = calibration['candidate']
            model = CalibratedModel(data['identity'], data['candidates'][name])
            model.fit(calibration['dev']).calibrate(calibration['validation'])
            screens[name]['calibrated_estimate'] = model.predict(data['demand'], identity=data['identity'])
        result['screens'] = screens
        result['input_sha256'][str(args.cost_input)] = hashlib.sha256(args.cost_input.read_bytes()).hexdigest()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'{result["status"]}: overhead median '
          f'{result["overhead"]["median_relative_overhead"]:.3%}, observed range '
          f'{result["overhead"]["observed_relative_range"]}; performance gate not-measured')


if __name__ == '__main__':
    main()
