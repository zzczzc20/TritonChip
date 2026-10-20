#!/usr/bin/env python3
"""Screen declared architecture resources before synthesis; never auto-approve.

JSON input: identity, demand, candidates{name:capacity-map}, optional cycle_budget.
Each resource demand has work/unit/source; capacity has per_cycle/unit/source.
Use actual per-PC/direction and shared NoC-cut keys, not aggregate HBM peaks.
Optional calibration: {candidate:name, dev:[records], validation:[records]}.
Frozen holdout data is not accepted for training/calibration.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from tnca.costmodel import resource_screen, CalibratedModel, implementation_screen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    result = {name: resource_screen(data['demand'], capacities,
              identity=data['identity'], cycle_budget=data.get('cycle_budget'))
              for name, capacities in data['candidates'].items()}
    calibration = data.get('calibration')
    for name, specification in data.get('implementation', {}).items():
        result[name]['implementation'] = implementation_screen(
            specification['floors'], specification['available'], identity=data['identity'])
    if calibration:
        name = calibration['candidate']
        model = CalibratedModel(data['identity'], data['candidates'][name])
        model.fit(calibration['dev']).calibrate(calibration['validation'])
        result[name]['calibrated_estimate'] = model.predict(data['demand'], identity=data['identity'])
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for name, item in result.items():
        print(f'{name}: {item["decision"]}, optimistic >= {item["lower_bound_cycles"]:g} cycles, '
              f'limiting {item["limiting_resources"][0]}')


if __name__ == '__main__':
    main()
