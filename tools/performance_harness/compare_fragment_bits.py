#!/usr/bin/env python3
"""Bitwise K-order check of matched fragment runs (S1-02 ordered K).

Each input is a `fragment_case.py --bitwise` report. Runs that share backend,
dtype and input layout are matched: the 16x32 multi-fragment path's raw
output bits must equal the 16x16 path's, for the same inputs, and each must
equal the independent per-K FP32 FMA reference. Nothing is tolerance-based.
A mismatch is reported with its elements; it is never relaxed here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('status') != 'functional-pass' or not data.get('bitwise'):
        raise ValueError(f'{path}: a completed --bitwise fragment run is required')
    return data, hashlib.sha256(raw).hexdigest()


def compare(paths):
    groups = {}
    for path in paths:
        data, digest = load(path)
        key = (data['backend'], data['dtype'], data['strided'], bool(data.get('cancel')))
        if data['columns'] in groups.setdefault(key, {}):
            raise ValueError(f'two {data["columns"]}-column runs for {key}')
        groups[key][data['columns']] = (data, digest, str(path))
    result = dict(schema='tnca-fragment-bitwise/1',
                  claim='16x32 multi-fragment output bits == 16x16 output bits == per-K FP32 FMA reference',
                  groups=[], rows=0, failures=[])
    for (backend, dtype, strided, cancel), runs in sorted(groups.items()):
        if set(runs) != {16, 32}:
            raise ValueError(f'{(backend, dtype, strided, cancel)}: need one 16- and one 32-column run')
        (narrow, n_sha, n_path), (wide, w_sha, w_path) = runs[16], runs[32]
        if narrow['software_tree'] != wide['software_tree']:
            raise ValueError('the matched runs come from different software trees')
        # The tile is the variable; the model output directory and the host
        # wait limit do not reach compilation or execution.
        ignored = ('TNCA_GEMM_TILE', 'TNCA_TLM_OUT', 'TNCA_HOST_TIMEOUT')
        for key in set(narrow['knobs']) | set(wide['knobs']):
            if key not in ignored and narrow['knobs'].get(key) != wide['knobs'].get(key):
                raise ValueError(f'unmatched knob {key}')
        old = {(r['graph'], tuple(r['shape'])): r for r in narrow['rows']}
        new = {(r['graph'], tuple(r['shape'])): r for r in wide['rows']}
        if old.keys() != new.keys():
            raise ValueError('incomplete matching graphs/shapes')
        group = dict(backend=backend, dtype=dtype, strided=strided, cancelling_inputs=cancel,
                     software_tree=wide['software_tree'],
                     narrow=dict(report=n_path, report_sha256=n_sha), wide=dict(report=w_path, report_sha256=w_sha),
                     rows=[])
        for key in sorted(old):
            a, b = old[key], new[key]
            if [i['sha256'] for i in a['inputs']] != [i['sha256'] for i in b['inputs']]:
                raise ValueError(f'{key}: the two runs used different inputs')
            row = dict(graph=key[0], shape=list(key[1]), k_multiple_of_16=key[1][2] % 16 == 0,
                       narrow_tiles=a.get('logical_tiles'), wide_tiles=b.get('logical_tiles'),
                       narrow_vs_wide_bitwise=a['output_bits_sha256'] == b['output_bits_sha256'],
                       narrow_vs_reference=a['bitwise_reference']['equal'],
                       wide_vs_reference=b['bitwise_reference']['equal'],
                       narrow_mismatches=a['bitwise_reference']['mismatches'],
                       wide_mismatches=b['bitwise_reference']['mismatches'],
                       elements=b['bitwise_reference']['elements'],
                       cold_and_resident_same=a['cold_bitwise_same'] and b['cold_bitwise_same'],
                       other_k_orders_would_differ=b.get('order_sensitivity'),
                       output_bits_sha256=b['output_bits_sha256'],
                       reference_bits_sha256=b['reference_bits_sha256'])
            if [16, 32, 16] not in [list(t) for t in (b.get('logical_tiles') or [])]:
                row['wide_path_not_exercised'] = True
            ok = (row['narrow_vs_wide_bitwise'] and row['narrow_vs_reference'] and
                  row['wide_vs_reference'] and row['cold_and_resident_same'] and
                  not row.get('wide_path_not_exercised'))
            row['status'] = 'bitwise-identical' if ok else 'mismatch'
            if not ok:
                result['failures'].append(dict(group=[backend, dtype, strided, cancel], row=row,
                                               narrow_examples=a['bitwise_reference']['examples'],
                                               wide_examples=b['bitwise_reference']['examples']))
            group['rows'].append(row)
        result['groups'].append(group)
        result['rows'] += len(group['rows'])
    rows = [r for g in result['groups'] for r in g['rows']]
    result['order_discrimination'] = dict(
        rows_where_blocked16_differs=sum(1 for r in rows if (r['other_k_orders_would_differ'] or {}).get('blocked16_mismatches')),
        rows_where_reversed_differs=sum(1 for r in rows if (r['other_k_orders_would_differ'] or {}).get('reversed_mismatches')),
        note='a bitwise match discriminates the per-K order only on rows where another order differs')
    result['status'] = 'bitwise-identical' if not result['failures'] else 'mismatch'
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reports', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.reports)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], rows=report['rows'],
                          failures=len(report['failures'])), indent=2))
