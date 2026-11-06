#!/usr/bin/env python3
"""Summarize complete strided RTL probe JSONL; never suppress failed records."""
import argparse
import glob
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pattern', help='glob matching result JSONL files')
args = parser.parse_args()
rows = []
for filename in sorted(glob.glob(args.pattern)):
    arm = Path(filename).name.split('.')[0]
    for line in Path(filename).read_text().splitlines():
        record = json.loads(line)
        rows.append((arm, record, filename))
if not rows:
    raise SystemExit('no completed records')
base = {r['point']: r['device_cycles_samples'][0] for a, r, _ in rows
        if a == 'g2' and r.get('status') == 'pass' and r.get('device_cycles_samples')}
for arm, record, filename in rows:
    point = record['point']
    print(f'RTL arm={arm} point={point} status={record.get("status")} source={filename}')
    print(f'  problems={record.get("problems")} software_tree={record.get("software_tree")}')
    samples = record.get('device_cycles_samples') or []
    if record.get('status') != 'pass' or not samples or not record.get('timed_launches'):
        print('  no valid timed measurement')
        continue
    cycle = samples[0]
    change = (cycle / base[point] - 1) * 100 if point in base else None
    tiles = [t for launch in record['timed_launches'] for t in launch['tiles'].values()]
    ports = [p for launch in record['timed_launches'] for p in launch['ports'].values()]
    tk = ('creq', 'creq_we', 'mreq_rd', 'mreq_wr', 'hit', 'miss', 'merge',
          'mshr_full', 'c_occ_sum', 'h_occ_sum', 'x_lsu', 'x_hazard', 'issued')
    pk = ('ar', 'aw', 'busy', 'rd_occ_sum', 'wr_occ_sum')
    print(f'  device_cycles={cycle} change_vs_g2_percent={change}')
    print('  tile_sums=' + repr({k: sum(t.get(k, 0) for t in tiles) for k in tk}))
    print('  port_sums=' + repr({k: sum(p.get(k, 0) for p in ports) for k in pk}))
