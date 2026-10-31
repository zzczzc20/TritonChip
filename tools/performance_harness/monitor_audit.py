#!/usr/bin/env python3
"""Audit matched native off/aggregate/trace results and bounded trace metadata."""
import argparse
import json
from pathlib import Path


def audit(paths):
    records = [json.loads(path.read_text()) for path in paths]
    modes = [r['knobs'].get('TNCA_TLM_MONITOR', 'aggregate') for r in records]
    if set(modes) != {'off', 'aggregate', 'trace'} or len(records) != 3:
        raise ValueError('require exactly one matched native run of each monitor mode')
    first = records[0]
    identities, traces = [], []
    for mode, record in zip(modes, records):
        model_files = sorted(Path(record['knobs']['TNCA_TLM_OUT']).glob('model-*/identity.json'))
        if not model_files:
            raise ValueError('missing model identities')
        per_run = []
        for identity_file in model_files:
            directory = identity_file.parent
            identity = json.loads(identity_file.read_text())
            per_run.append([identity['binary_sha256'], identity['config_sha256']])
            sockets = json.loads((directory/'sockets.json').read_text())
            trace = json.loads((directory/'trace.json').read_text())
            limit = int(identity['config']['trace_events'])
            dropped = trace['metadata']['dropped_events']
            if (sockets['mode'] != mode or not sockets['complete'] or
                    not trace['metadata']['complete'] or
                    not 0 <= len(trace['traceEvents']) <= limit or dropped < 0):
                raise ValueError('incomplete, unbounded or mislabelled trace')
            traces.append(dict(mode=mode, emitted=len(trace['traceEvents']), dropped=dropped, limit=limit))
        identities.append(per_run)
    if any(identity != identities[0] for identity in identities):
        raise ValueError('monitor runs used different model binaries or configuration')
    rows = lambda r: {(x['graph'], tuple(x['shape'])): x for x in r['rows']}
    reference = rows(first)
    for record in records:
        if record['status'] != 'functional-pass' or record['software_tree'] != first['software_tree']:
            raise ValueError('incomplete or mismatched source identity')
        values = rows(record)
        if values.keys() != reference.keys():
            raise ValueError('incomplete matched native workload')
        for key, value in values.items():
            for field in ('device_cycles', 'output_f32_sha256', 'model_counter_deltas'):
                if value[field] != reference[key][field]:
                    raise ValueError(f'monitor changed native {field}: {key}')
            if [a['code_sha256'] for a in value['artifacts']] != [a['code_sha256'] for a in reference[key]['artifacts']]:
                raise ValueError('monitor runs used different machine code')
    return dict(schema='tnca-native-monitor-audit/1', status='passed', modes=modes,
                cases=len(reference), identical_output_bits=True, identical_device_cycles=True,
                identical_native_counters=True, traces=traces, performance_acceptance='not-measured')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reports', nargs=3, type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.reports)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
