#!/usr/bin/env python3
"""Summarize the existing RTL records; no new performance run is implied."""
from pathlib import Path
import glob
import json

root = Path(__file__).resolve().parents[5]
evidence = root / 'evidence/r41/hbm_bw/g4/rtl_s4'
patterns = {'g2': 'controls/g2.mode0*.jsonl',
            's1': 'controls/s1.mode0*.jsonl',
            's4': 'old_compiler/s4.mode0*.jsonl'}
records = {}
for arm, pattern in patterns.items():
    for filename in sorted(glob.glob(str(evidence / pattern))):
        for line in Path(filename).read_text().splitlines():
            record = json.loads(line)
            if '-small-strided-' in record['point']:
                records[(arm, record['point'])] = (record, Path(filename).relative_to(root))
assert len(records) == 6, f'expected two records per arm, found {len(records)}'
for point in sorted(p for arm, p in records if arm == 'g2'):
    base = records['g2', point][0]['device_cycles_samples'][0]
    print(point)
    for arm in patterns:
        record, source = records[arm, point]
        cycle = record['device_cycles_samples'][0]
        launch = record['timed_launches'][0]
        tiles = launch['tiles'].values()
        keys = ('creq', 'creq_we', 'mreq_rd', 'mreq_wr', 'x_lsu', 'x_hazard', 'c_occ_sum')
        counters = {key: sum(tile.get(key, 0) for tile in tiles) for key in keys}
        print(f'  {arm}: status={record["status"]} cycles={cycle} '
              f'change_vs_g2={(cycle / base - 1) * 100:.2f}% '
              f'tree={record["software_tree"]} source={source}')
        print(f'    tile sums: {counters}')
