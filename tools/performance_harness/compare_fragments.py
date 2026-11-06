#!/usr/bin/env python3
"""Compare matched fragment development runs; model cycles stay predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data['status'] != 'functional-pass':
        raise ValueError('both cases must complete their numerical checks')
    model_root = data.get('knobs', {}).get('TNCA_TLM_OUT')
    models = []
    if model_root:
        for identity in sorted(Path(model_root).glob('model-*/identity.json')):
            model = json.loads(identity.read_text())
            models.append({k: model[k] for k in
                           ('binary_sha256', 'config_sha256', 'calibration', 'execution', 'dram_path')})
        if not models:
            raise ValueError('missing original model identity files')
    return data, models, hashlib.sha256(raw).hexdigest()


def compare(control_path, candidate_path):
    control, cm, cs = load(control_path)
    candidate, nm, ns = load(candidate_path)
    if (control['backend'] != candidate['backend'] or control['dtype'] != candidate['dtype'] or
            control['strided'] != candidate['strided'] or cm != nm):
        raise ValueError('incompatible execution, input layout or model identity')
    ignored = {'TNCA_GEMM_TILE', 'TNCA_TLM_OUT', 'TNCA_TLM_CONFIG'}
    for key in set(control['knobs']) | set(candidate['knobs']):
        if key not in ignored and control['knobs'].get(key) != candidate['knobs'].get(key):
            raise ValueError(f'unmatched knob: {key}')
    old = {(r['graph'], tuple(r['shape'])): r for r in control['rows']}
    new = {(r['graph'], tuple(r['shape'])): r for r in candidate['rows']}
    if old.keys() != new.keys():
        raise ValueError('incomplete matching graphs/shapes')
    rows = []
    for key, a in old.items():
        b = new[key]
        if any(a[k] != b[k] for k in ('strides', 'rtol', 'atol', 'status')):
            raise ValueError('input or numerical acceptance changed')
        def counts(row):
            return {name: sum(v for model in row['model_counter_deltas'].values()
                              for k, v in model.items() if k.endswith(name))
                    for name in ('_class_a_vread', '_class_a_row', '_instructions', '_mma_commands')}
        rows.append(dict(graph=key[0], shape=list(key[1]),
                         control_cycles=a['device_cycles'], candidate_cycles=b['device_cycles'],
                         predicted_speedup=a['device_cycles']/b['device_cycles'],
                         control_counts=counts(a), candidate_counts=counts(b),
                         control_code_bytes=sum(x['code_bytes'] for x in a['artifacts']),
                         candidate_code_bytes=sum(x['code_bytes'] for x in b['artifacts'])))
    return dict(schema='tnca-fragment-comparison/1', evidence='model-prediction-uncalibrated',
                performance_acceptance='not-measured',
                control=dict(report_sha256=cs, software_tree=control['software_tree'], columns=control['columns']),
                candidate=dict(report_sha256=ns, software_tree=candidate['software_tree'], columns=candidate['columns']),
                models=cm, rows=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.control, args.candidate)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
