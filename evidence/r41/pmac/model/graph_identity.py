#!/usr/bin/env python3
"""Graph-level default identity: the same standard-Torch points under
(a) the unmodified model binary and (b) the what-if binary with the what-if
keys written at their defaults, against the what-if binary's default runs.

    graph_identity.py --runs RUNS_DIR --out graph_identity.json

Compared per point: every model counter of the session (final snapshot and
the snapshot at the untimed-cold/timed switch), and every launch's start/end
cycle, status and retired count (launches.json); host wall time is not
compared. The device-literal copies and argument blocks of the native path
depend only on the software tree, which is the same in all three.
"""
import argparse
import json
from pathlib import Path

REF = 'c-g1-pf0'
CONTROLS = ('i-base-g1-pf0', 'i-mod-keys-default')
LAUNCH_KEYS = ('kind', 'pc', 'grid', 'cores', 'role_mask', 'status', 'fault_code', 'start_cycle', 'end_cycle', 'cycles', 'retired')


def model_dir(run):
    rec = json.loads((run/'record.json').read_text())
    return Path(rec['systemc_model'][-1]['model_dir']), rec


def load(run):
    d, rec = model_dir(run)
    launches = json.loads((d/'launches.json').read_text())
    launches = launches['launches'] if isinstance(launches, dict) else launches
    return dict(final=json.loads((d/'counters.json').read_text()),
                cold=json.loads((d/'counters_untimed_cold.json').read_text()),
                launches=[{k: l.get(k) for k in LAUNCH_KEYS} for l in launches],
                device=rec['time']['device'], status=rec['status'], max_rel_err=rec['correctness']['max_rel_err'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows, ok = [], True
    for control in CONTROLS:
        cdir = a.runs/control/'runs'
        if not cdir.exists():
            continue
        for run in sorted(cdir.iterdir()):
            ref = a.runs/REF/'runs'/run.name
            x, y = load(ref), load(run)
            same = {k: x[k] == y[k] for k in ('final', 'cold', 'launches', 'device', 'status', 'max_rel_err')}
            ok &= all(same.values())
            rows.append(dict(control=control, point=run.name, identical=all(same.values()), fields=same,
                             counters=len(x['final'].get('counters', x['final'])), launches=len(x['launches']),
                             device_cycles=x['device']))
            print(control, run.name, 'IDENTICAL' if all(same.values()) else f'DIFFERENT {same}')
    a.out.write_text(json.dumps(dict(schema='tnca-pmac-graph-identity/1', reference=REF,
                                     status='identical' if ok and rows else 'DIFFERENT', rows=rows), indent=1) + '\n')
    print('identical' if ok and rows else 'DIFFERENT')


if __name__ == '__main__':
    main()
