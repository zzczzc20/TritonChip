#!/usr/bin/env python3
"""Compute an ideal EOS-stop time bound from completed, recorded chat turns."""
import argparse
import json
import math
from pathlib import Path


def bounds(report, clock_hz=None):
    if report.get('device') != 'tnca':
        raise ValueError('the bound requires a device-feedback quality report')
    if clock_hz is not None and (type(clock_hz) is not int or clock_hz <= 0):
        raise ValueError('clock_hz must be a positive integer')
    rows = []
    for prompt in report.get('prompts', []):
        for index, turn in enumerate(prompt.get('turns', [])):
            tokens, steps = turn.get('tokens') or [], turn.get('steps') or []
            timing = turn.get('user_visible_timing') or {}
            wall = timing.get('prompt_to_return_seconds')
            if not tokens or len(tokens) > len(steps) or not isinstance(wall, (int, float)) or \
                    not math.isfinite(wall) or wall <= 0:
                raise ValueError(f"{prompt.get('id')}: incomplete timing or tokens")
            if steps[0].get('kind') != 'prefill' or any(step.get('kind') != 'decode' for step in steps[1:]):
                raise ValueError(f"{prompt.get('id')}: unexpected generation-step order")
            discarded = steps[len(tokens):]
            if any(step.get('kind') != 'decode' or not isinstance(step.get('seconds'), (int, float)) or
                   not math.isfinite(step['seconds']) or step['seconds'] < 0 for step in discarded):
                raise ValueError(f"{prompt.get('id')}: incomplete discarded-step timings")
            saved = sum(step['seconds'] for step in discarded)
            if saved >= wall:
                raise ValueError(f"{prompt.get('id')}: discarded steps exceed wall time")
            row = dict(prompt=prompt['id'], turn=index, returned_ids=len(tokens), executed_steps=len(steps),
                             discarded_decode_steps=len(discarded), discarded_decode_seconds=saved,
                             measured_wall_seconds=wall, ideal_wall_seconds=wall-saved,
                             ideal_reduction_fraction=saved/wall)
            decode = steps[1:]
            cycles = [(step.get('transfers_inside_step') or {}).get('device_cycles') for step in decode]
            if clock_hz is not None and decode and all(type(cycle) is int and cycle >= 0 for cycle in cycles):
                decode_wall = sum(step['seconds'] for step in decode)
                device_seconds = sum(cycles) / clock_hz
                if decode_wall <= 0:
                    raise ValueError(f"{prompt.get('id')}: invalid decode wall time")
                residual = decode_wall - device_seconds
                row.update(decode_device_cycles=sum(cycles), decode_wall_seconds=decode_wall,
                           non_device_residual_seconds=residual, non_device_residual_fraction=residual/decode_wall,
                           residual_reduction_to_30_percent=max(0., residual - device_seconds * 3 / 7),
                           residual_scope='wall minus device cycles; includes waits, transfers and counter overlap')
            rows.append(row)
    return dict(schema='r41-eos-ideal-bound/1', model=report.get('model'),
                scope='analytic only: free EOS decision, unchanged remaining work; not a measured speedup', rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('quality', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--clock-hz', type=int, help='measured device clock for the residual estimate')
    args = parser.parse_args()
    result = bounds(json.loads(args.quality.read_text()), args.clock_hz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
