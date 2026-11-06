#!/usr/bin/env python3
"""Markdown table of the sealed v2 predictions: fast and slow graph cycles per configuration."""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
rows = {}
for name in ('prospective-prediction-v2.json', 'gemm-subset-predictions-v2.json'):
    doc = json.loads((repo/'evidence/r41/stage1/kickoff'/name).read_text())
    family = 'prospective' if name.startswith('prospective') else 'gemm'
    for e in doc['predictions']:
        rows.setdefault((family, e['point'], e['gemm_tile']), {})[e['path']] = e


def cell(e):
    if e is None:
        return 'missing', ''
    if e['status'] == 'not-run':
        return 'not run', ''
    tag = '' if e['status'] == 'sealed' else ' (inconclusive)' if 'inconclusive' in e['status'] else f' ({e["status"]})'
    return f'{e["graph_cycles"]:,}{tag}', f'{e["simulator"]["model_launch_host_s"]:.1f}'


print('| Set | Point | Tile | Fast cycles | Slow cycles | Fast model s | Slow model s |')
print('|---|---|---|---:|---:|---:|---:|')
for (family, point, tile), paths in sorted(rows.items(), key=lambda kv: (kv[0][0] != 'prospective', kv[0][1], kv[0][2])):
    fast, fs = cell(paths.get('fast'))
    slow, ss = cell(paths.get('slow'))
    print(f'| {family} | `{point}` | {tile} | {fast} | {slow} | {fs} | {ss} |')
