#!/usr/bin/env python3
"""Quantify validation error at an accepted-request RTL boundary, without fitting.

Fixed RTL release times are conditional inputs. This comparison cannot establish
closed-loop chip makespan accuracy or production-workload prediction intervals.
"""
import argparse
import json
from pathlib import Path
import statistics

from run import sha


def analyze(capture_dir, model_dir):
    capture = json.loads((capture_dir/'capture.json').read_text())
    run = json.loads((model_dir/'report.json').read_text())
    if run['status'] != 'completed-prediction':
        raise ValueError('completed prediction required')
    if (sha(capture_dir/'rtl.capture') != capture['capture_sha256']
            or sha(capture_dir/'requests.trace') != capture['trace_sha256']
            or run['identity']['trace_sha256'] != capture['trace_sha256']
            or sha(model_dir/'responses.tsv') != run['output_sha256']['responses.tsv']):
        raise ValueError('capture/run provenance mismatch')
    requests, pending = {}, {}
    for line in (capture_dir/'rtl.capture').read_text().splitlines():
        f = line.split()
        if f[0] == 'Q':
            seq, start, src, tag = map(int, f[1:5])
            requests[seq] = dict(start=start)
            pending[src, tag] = seq
        elif f[0] == 'P':
            end, src, tag = map(int, f[1:4])
            seq = pending.pop((src, tag));requests[seq]['rtl_end'] = end
    seen = set()
    for line in (model_dir/'responses.tsv').read_text().splitlines():
        seq, end, value = line.split();seq, end = int(seq), int(end)
        if seq in seen or seq not in requests:
            raise ValueError('duplicate or unknown model response')
        requests[seq]['model_end'] = end;seen.add(seq)
    if pending or seen != requests.keys() or not seen:
        raise ValueError('incomplete response coverage')
    errors = [r['model_end']-r['rtl_end'] for r in requests.values()]
    absolute = sorted(abs(v) for v in errors)
    return dict(schema='tnca-tlm-boundary-error/1', samples=len(errors),
        scope='conditional accepted-request completion error, same reference cycles; fixed RTL arrival schedule',
        model_identity=run['identity'], capture_identity=capture['identity'],
        signed_range_cycles=[min(errors), max(errors)], mean_signed_cycles=statistics.mean(errors),
        mean_absolute_cycles=statistics.mean(absolute), observed_max_absolute_cycles=max(absolute),
        p95_absolute_cycles=absolute[(95*len(absolute)-1)//100],
        release_slip_cycles=run['summary']['counters'].get('release_slip_cycles'),
        production_timing_error_bound_cycles=None, whole_chip_accuracy='not-measured',
        performance_gate='not-measured')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.capture, a.run)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False);f.write('\n')
