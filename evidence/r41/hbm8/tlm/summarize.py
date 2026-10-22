#!/usr/bin/env python3
"""Verify retained case reports and summarize measured resources, not device speed."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def torch_case(path):
    d = read(path)
    assert d['status'] == 'functional-pass', (path, d['status'])
    assert all(c['status'] == 'pass' for c in d['checks']), path
    assert int(d['knobs']['TNCA_TLM_MEMORY_BYTES'], 0) == 0x200000000, path
    assert d['window']['model_memory_bytes'] == 0x200000000, path
    assert d['map']['status'] == 'planned-allocation-not-measured', path
    assert all(g['launches'] > 0 for g in d['graphs']), path
    pc = {}
    for g in d['graphs']:
        for k, v in g['pc_bytes'].items():
            pc[k] = pc.get(k, 0) + v
    counters = d.get('model_counters') or {}
    if not counters:
        for g in d['graphs']:
            for k, v in g['dram'].items():
                counters[k] = counters.get(k, 0) + v
    slow = d['knobs']['TNCA_DRAM_PATH'] == 'slow'
    bad = {k: counters.get(k, 0) for k in
           ('dram_slow_unconverged', 'dram_slow_unmatched', 'dram_child_failures')}
    timing = ('inconclusive-fixed-point' if any(bad.values()) else
              'uncalibrated-fixed-point-checked') if slow else 'uncalibrated'
    host = d['host']
    r = dict(status=d['status'], checks=len(d['checks']), graphs=len(d['graphs']),
             window_bytes=d['window']['model_memory_bytes'], placements=d['placements'],
             pc_bytes=pc, host=host, model_vmhwm_kib=d.get('model_vmhwm_kib'),
             model_timing_status=timing, timing_exceptions=bad,
             map_sha256=d['map']['sha256'], software_tree=d['software_tree'])
    if d['case'] == 'stream':
        n = d['placements']['x']['bytes']
        assert n > 1 << 31 and d['placements']['y']['bytes'] == n, path
        assert all(pc.get(f'pc_{p}_bytes', 0) > 0 for p in range(16, 32)), path
        copy = next(c for c in d['checks'] if c['name'].startswith('copy equals independently'))
        assert copy['compared_bytes'] == n and copy['mismatched_chunks'] == 0, path
        assert d['touched_tensor_bytes'] == 11 * (1 << 29), path  # 5.5 GiB
        resources = read(path.parent.with_suffix('.resources.json'))
        assert resources['returncode'] == 0 and resources['guard_refusal'] is None, resources
        assert len(resources['cpus']) <= 6, resources
        assert resources['peak_summed_rss_kib'] < 16 * (1 << 20), resources
        r.update(touched_tensor_bytes=d['touched_tensor_bytes'], graph_tensor_bytes=d['graph_tensor_bytes'],
                 resident_extra_bytes=d['resident_extra_bytes'], stream_chunk_bytes=d['stream_chunk_bytes'],
                 cpu_copy_compared_bytes=copy['compared_bytes'], resources=resources)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = dict(scope='functional model execution; measured host resources; planned BO; no board or calibrated timing claim',
               native={}, torch={}, prior_scale=[])
    for name in ('native-exact', 'native-full-bo'):
        d = read(a.raw/'final'/name/'report.json')
        assert d['status'] == 'pass' and len(d['negatives']) == 20, name
        assert all(p['status'] == 'pass' and p['pc_bytes'] == d['expected_pc_bytes'] and
                   p['checked_bytes'] == 2832 for p in d['positives']), name
        assert all(n['status'] == 'refused-as-expected' for n in d['negatives']), name
        out['native'][name] = d
    for path in ('fast', 'slow'):
        for case in ('placement', 'crossing', 'gemm', 'refusal'):
            name = f'{case}-{path}'
            out['torch'][name] = torch_case(a.raw/'final'/f'current-{name}'/'report.json')
        out['torch'][f'stream-{path}'] = torch_case(a.raw/'final'/f'stream-2112mib-v2-{path}'/'report.json')
    for p in sorted((a.raw/'prior').glob('*/report.json')):
        d = read(p)
        if d.get('case') == 'stream':
            out['prior_scale'].append(dict(case=p.parent.name, status=d['status'], window=d['window'],
                                           tensor_bytes=2*d['placements']['x']['bytes'], host=d['host']))
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    print('All 10 final Torch cases, 4 native positives and 40 negative rows verified.')


if __name__ == '__main__':
    main()
