#!/usr/bin/env python3
"""Check V3 bandwidth closure with the native SystemC streaming-add program."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.r41.wide_path_model_tests import streaming_add


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base_text = (ROOT/'tools/performance_harness/systemc/configs/r41-hbm8-device-service.cfg').read_text()
    base = dict(line.split('#')[0].strip().split('=', 1) for line in base_text.splitlines()
                if '=' in line.split('#')[0])
    base.update(clock_hz='100000000', sfu_occupancy='22', sfu_latency='23')
    variants = {'baseline': {}}
    for units in (8, 16):
        variants[f'v3_lsu_{units}'] = dict(tl_core_beat_bytes='64', tl_noc_payload_bytes='64',
            tl_lsu_units=str(units), tl_lsu_shared_issue='0', tl_memory_lanes='8',
            lsu_beat_slots=str(max(1, 64//units)), outlets='32', active_outlets='32',
            tl_pc_stripe_bytes='4096', mshrs='128', targets='256')
    rows = []
    for name, changes in variants.items():
        values = dict(base, **changes)
        run = args.out/name
        run.mkdir()
        (run/'chip.cfg').write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
        summary = streaming_add(args.binary.resolve(), run, unroll=16)
        cycles = summary['visible_cycles']
        logical_bytes = 3*3*4096*4
        row = dict(variant=name, configuration=changes, cycles=cycles,
            logical_bytes=logical_bytes, logical_gb_per_s_at_100mhz=logical_bytes/cycles*0.1,
            board_measured=False)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    baseline_cycles = rows[0]['cycles']
    for row in rows[1:]:
        row['predicted_stream_add_speedup'] = baseline_cycles/row['cycles']
        row['meets_50gbps'] = row['logical_gb_per_s_at_100mhz'] >= 50
        row['meets_5x'] = row['predicted_stream_add_speedup'] >= 5
    result = dict(evidence='native-SystemC-uncalibrated', rows=rows)
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, sort_keys=True))


if __name__ == '__main__':
    main()
