#!/usr/bin/env python3
"""Turn run_graphs.py results into calibrate.py prediction rows.

    predictions.py --results RUN/results.json [--results ...] --config CFG --binary BIN --out predictions.jsonl

One row per passing run with a model graph makespan (or a one-launch graph
measured on the cold call whose only problem is the native path's cold-setup
launches, see run_graphs.py --timed-auto): the point, the knobs the
run used (exactly the board record's, see run_graphs.py --dataset), the model
graph cycles of the measured call (first launch start to last launch end,
host gaps included), its launches, and the model identity. A run that did not
pass is listed on stderr and produces no row: nothing is approximated.

Knobs come from run_graphs.py's verified `knobs` (the record's own
platform.knobs, model-harness keys excluded). A results file written before
that verification existed is checked against the record.json beside it
(RUN/runs/<key>/record.json): a row whose stated knobs differ from what the
record ran is refused (knob-mismatch); without a record the prediction is
kept but marked knobs_source=unverified-legacy-row.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from prediction_domain import prediction_status, VALID_STATUSES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_knobs(row, results):
    """(knobs, source) or (None, refusal) for one run_graphs.py results row."""
    if row.get('knobs_verified') is True and isinstance(row.get('knobs'), dict):
        return row['knobs'], 'record-platform-knobs'
    if row.get('knobs_verified') is False:
        return None, 'knob-mismatch'
    key = str(row.get('key') or '')
    record = Path(results).parent/'runs'/key.replace('/', '__').replace('@', '__')/'record.json'
    claimed = row.get('knobs') or {}
    if key and record.is_file():
        ran = (json.loads(record.read_text()).get('platform') or {}).get('knobs') or {}
        if any(ran.get(k) != str(v) for k, v in claimed.items()):
            return None, 'knob-mismatch'
        return claimed, 'legacy-row-checked-against-record'
    return claimed, 'unverified-legacy-row'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', action='append', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--binary', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    model = dict(binary_sha256=sha(a.binary), config_sha256=sha(a.config), config=str(a.config))
    configured_slow = any(line.split('#')[0].strip() == 'dram_path=1' for line in Path(a.config).read_text().splitlines())
    rows, missing = [], []
    for path in a.results:
        for r in json.loads(Path(path).read_text())['points']:
            g = r.get('model_graph')
            if not g or not (r.get('status') == 'pass' or (r.get('cold_setup_only') and g.get('measured_call') == 'cold')):
                missing.append((r.get('key'), r.get('status')))
                continue
            qualification = prediction_status(r, configured_slow=configured_slow)
            if qualification not in VALID_STATUSES:
                missing.append((r.get('key'), qualification))
                continue
            knobs, source = row_knobs(r, path)
            if knobs is None:
                missing.append((r.get('key'), source))
                continue
            rows.append(dict(point=r['point'], knobs=knobs, knobs_source=source, graph_cycles=g['graph_cycles'],
                             launches=g['launches'], device_cycles=g['device_cycles'], model=model,
                             model_timing_status=qualification,
                             run=dict(results=str(path), key=r['key'], process_wall_s=r.get('process_wall_s'))))
    Path(a.out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} predictions', file=sys.stderr)
    for key, status in missing:
        print(f'no prediction: {key} ({status})', file=sys.stderr)


if __name__ == '__main__':
    main()
