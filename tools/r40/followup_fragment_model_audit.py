#!/usr/bin/env python3
"""Compare raw, identity-bound M2 records, refusing stale summary identities."""
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    def read(relative):
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest[relative]['sha256']:
            raise ValueError('raw evidence hash mismatch: ' + relative)
        return json.loads(data)
    state = read('model-m2-state.json')
    if len(state['runs']) != 4 or any(r['returncode'] for r in state['runs']):
        raise ValueError('incomplete model experiment')
    expected = {
        'fe1b-n16': ('fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75', '16,16,16'),
        'resident-n32': ('4bfc9ca1fa8565bb6f8e4d38ef8a43c5845646634ae3921d60aee1af1bc1e1f1', '16,32,16'),
        'lowering-n16': ('60cf7062521a3e5d84677016836f48bf331270c389aaf4166f1f8d3d5818ef4c', '16,16,16'),
        'lowering-n32': ('60cf7062521a3e5d84677016836f48bf331270c389aaf4166f1f8d3d5818ef4c', '16,32,16'),
    }
    rows = []
    common = None
    for case, (tree, tile) in expected.items():
        prefix = 'model-' + case
        candidates = list((root / prefix / 'runs').glob('*/record.json'))
        if len(candidates) != 1:
            raise ValueError('expected one original-point record')
        rel = str(candidates[0].relative_to(root))
        record = read(rel)
        env = read(str(candidates[0].with_name('env.json').relative_to(root)))
        knobs = record['platform']['knobs']
        if any(env.get(k) != v for k, v in knobs.items()):
            raise ValueError('record disagrees with effective environment')
        if (record['status'] != 'pass' or record['fallbacks'] or record['problems']
                or record['evidence_level'] != 'model-prediction-uncalibrated'
                or record['point_id'] != 'op.linear/medium/fp16-m128-k320-n256'
                or record['provenance']['software_tree'] != 'files:' + tree
                or knobs['TNCA_GEMM_TILE'] != tile or knobs['TNCA_PIPELINE'] != '1'):
            raise ValueError('incorrect experiment identity or unsuccessful point')
        error, tolerance = (record['correctness'][k] for k in ('max_rel_err', 'tol'))
        if not math.isfinite(error) or not 0 <= error <= tolerance:
            raise ValueError('numerical comparison failed')
        if record['model_graph']['device_cycles'] <= 0:
            raise ValueError('missing positive model cycles')
        identities = [m['identity'] for m in record['systemc_model']]
        if len(identities) != 1 or identities[0]['binary_sha256'] != state['model_sha256']:
            raise ValueError('model binary mismatch')
        config = identities[0]['config']
        if config['clock_hz'] != '90000000' or config['memory_bytes'] != '4294967296':
            raise ValueError('unexpected model clock/window')
        compare = (record['point_sha256'], config,
                   {k: v for k, v in knobs.items()
                    if k != 'TNCA_GEMM_TILE' and not k.startswith('TNCA_TLM_')})
        if common is not None and compare != common:
            raise ValueError('comparison changes more than source and tile')
        common = compare
        summary = read(prefix + '/results.json')['points'][0]
        mismatches = {k: dict(summary=summary['knobs'].get(k), effective=knobs[k])
                      for k in ('TNCA_GEMM_TILE', 'TNCA_PIPELINE')
                      if summary['knobs'].get(k) != knobs[k]}
        rows.append(dict(case=case, software_tree='files:' + tree, tile=tile,
                         model_cycles=record['model_graph']['device_cycles'],
                         max_rel_err=record['correctness']['max_rel_err'],
                         stale_summary_fields=mismatches))
    cycles = {r['case']: r['model_cycles'] for r in rows}
    return dict(evidence_level='model-prediction-uncalibrated', rows=rows,
                resident_to_lowering_speedup=cycles['resident-n32'] / cycles['lowering-n32'],
                remaining_wide_slowdown=cycles['lowering-n32'] / cycles['lowering-n16'],
                decision='Reject board performance qualification of this wide candidate.',
                not_measured=['board performance', 'full frozen-subset speedup', 'gate improvement'])


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2] / 'evidence/r40/followup/fragment_lowering_v2'
    print(json.dumps(audit(root), indent=2))
