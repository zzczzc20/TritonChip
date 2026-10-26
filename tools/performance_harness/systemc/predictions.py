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
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', action='append', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--binary', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    model = dict(binary_sha256=sha(a.binary), config_sha256=sha(a.config), config=str(a.config))
    rows, missing = [], []
    for path in a.results:
        for r in json.loads(Path(path).read_text())['points']:
            g = r.get('model_graph')
            if not g or not (r.get('status') == 'pass' or (r.get('cold_setup_only') and g.get('measured_call') == 'cold')):
                missing.append((r.get('key'), r.get('status')))
                continue
            rows.append(dict(point=r['point'], knobs=r['knobs'], graph_cycles=g['graph_cycles'],
                             launches=g['launches'], device_cycles=g['device_cycles'], model=model,
                             run=dict(results=str(path), key=r['key'], process_wall_s=r.get('process_wall_s'))))
    Path(a.out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} predictions', file=sys.stderr)
    for key, status in missing:
        print(f'no prediction: {key} ({status})', file=sys.stderr)


if __name__ == '__main__':
    main()
