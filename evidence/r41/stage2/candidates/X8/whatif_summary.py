#!/usr/bin/env python3
"""Summarise the X8 what-if runs (run_graphs.py results, one directory per
<profile>-<config>): device cycles of the measured (steady) call, and the
model counters of that call alone (final snapshot minus the cold one).

    whatif_summary.py RUNS_DIR > table.md   (also writes RUNS_DIR/summary.json)

Uncalibrated model predictions; compare ratios between configurations.
"""
import json
from pathlib import Path
import re
import sys

KEYS = ('mma_commands', 'class_lsu', 'instructions', 'matrix_engine_busy_cycles', 'matrix_engine_wait_cycles',
        'dma_commands_issued', 'dma_read_words', 'lsu_wait_cycles')


def counters(p):
    if not p.exists():
        return {}
    c = json.loads(p.read_text())
    c = c.get('counters', c)
    out = {}
    for k, v in c.items():
        m = re.match(r'^core_\d+_(.*)$', k)
        if m and isinstance(v, (int, float)) and 'peak' not in k:
            out[m.group(1)] = out.get(m.group(1), 0) + v
    return out


def main():
    runs = Path(sys.argv[1])
    rows = {}
    for res in sorted(runs.glob('*/results.json')):
        cfg = res.parent.name
        for row in json.loads(res.read_text())['points']:
            key = row['key']
            rec = json.loads(Path(row['record']).read_text())
            mdir = Path(rec['systemc_model'][-1]['model_dir'])
            fin, cold = counters(mdir/'counters.json'), counters(mdir/'counters_untimed_cold.json')
            d = {k: fin.get(k, 0) - cold.get(k, 0) for k in KEYS}
            rows.setdefault(key, {})[cfg] = dict(status=row['status'], cycles=row.get('device_cycles'),
                                                 max_rel_err=row.get('max_rel_err'), **d)
    (runs/'summary.json').write_text(json.dumps(rows, indent=1, sort_keys=True) + '\n')
    base = 'core-vrf-sync-v5-g1-pf0'
    labels = {'core-vrf-sync-v5-g1-pf0': 'gather (today)', 'core-vrf-sync-v5-g4-pf0-pipe5': 'gather + V3',
              'core-vrf-sync-v5-x8-g1-pf0': 'X8', 'core-vrf-sync-v5-x8-g4-pf0-pipe5': 'X8 + V3'}
    print('| point | configuration | status | device cycles | speedup vs gather | LSU instr / MMA | '
          'MMA busy share | MMA wait share | max rel err |')
    print('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for key in sorted(rows):
        b = rows[key].get(base, {}).get('cycles')
        for cfg in labels:
            r = rows[key].get(cfg)
            if not r:
                continue
            mma = max(r['mma_commands'], 1)
            print(f"| `{key.split('/')[-1]}` | {labels[cfg]} | {r['status']} | {r['cycles']:,} | "
                  f"{b / r['cycles']:.2f}x | {r['class_lsu'] / mma:.1f} | "
                  f"{100 * r['matrix_engine_busy_cycles'] / (3 * r['cycles']):.1f} % | "
                  f"{100 * r['matrix_engine_wait_cycles'] / (3 * r['cycles']):.1f} % | {r['max_rel_err']:.2g} |")


if __name__ == '__main__':
    main()
