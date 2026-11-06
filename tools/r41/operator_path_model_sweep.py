#!/usr/bin/env python3
"""Run standard torch add/silu against coupled, unbuilt memory-path hypotheses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
POINTS = ['op.add/small/fp16-n16384', 'op.silu/small/fp16-n16384']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--torch-extensions-dir', type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    config = ROOT/'tools/performance_harness/systemc/configs/r41-hbm8-device-service.cfg'
    base = dict(line.split('#')[0].strip().split('=', 1) for line in config.read_text().splitlines()
                if '=' in line.split('#')[0])
    # Current and loaded Stage-2 source: VLEN=16, scalar SFU LAT=6.
    base.update(clock_hz='100000000', sfu_occupancy='22', sfu_latency='23')
    rows = []
    for name, units, block, slices in [('baseline', 2, 16, 1), ('wide4', 4, 64, 4), ('wide8', 8, 128, 4), ('v3', 16, 128, 4), ('v3_lsu32', 32, 128, 4), ('v3_lsu32_sfu8', 32, 128, 8)]:
        values = dict(base)
        if name != 'baseline':
            values.update(tl_core_beat_bytes='64', tl_noc_payload_bytes='64',
                          tl_lsu_units=str(units), lsu_beat_slots=str(max(1, 16//units)),
                          sfu_occupancy=str(16//slices+6), sfu_latency=str(16//slices+7))
        if name.startswith('v3'):
            values.update(tl_lsu_shared_issue='0', tl_memory_lanes='8', tl_prefetch_lines='64',
                          tl_pc_stripe_bytes='4096', outlets='32', active_outlets='32',
                          mshrs='128', targets='256')
        cfg = a.out/f'{name}.cfg'
        cfg.write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
        command = [sys.executable, str(ROOT/'tools/performance_harness/systemc/run_graphs.py'),
            '--source-root', str(ROOT), '--binary', str(a.binary.resolve()), '--config', str(cfg),
            '--out', str(a.out/name), '--active-cores', '3', '--env', f'TNCA_RTL_MEM_PORTS={32 if name.startswith("v3") else 3}',
            '--env', 'TNCA_RTL_OUTSTANDING=8', '--env', f'TNCA_MAX_BLOCK={block}',
            '--jobs', '1', '--timeout', '600', '--torch-extensions-dir', str(a.torch_extensions_dir)]
        for point in POINTS:
            command += ['--point', point]
        (a.out/f'{name}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        log = a.out/f'{name}.log'
        with log.open('w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MAX_JOBS='2'))
        points = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        points = [r for r in points if r.get('key') in POINTS]
        assert len(points)==len(POINTS) and all(r['status']=='pass' for r in points), log
        rows.append({'variant': name, 'lsu_units': units, 'block': block, 'sfu_slices': slices, 'points': points})
        print(name, [(r['key'],r['device_cycles']) for r in points], flush=True)
    reference = {p['key']: p['device_cycles'] for p in rows[0]['points']}
    for row in rows:
        for point in row['points']:
            point['predicted_speedup_vs_model_baseline'] = reference[point['key']]/point['device_cycles']
    (a.out/'sweep.json').write_text(json.dumps({'evidence': 'model-prediction-uncalibrated',
        'clock_hz_configured': 100000000, 'board_speedup_measured': False,
        'scope': 'two standard torch points only; no FPGA image or throughput qualification',
        'rows': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
