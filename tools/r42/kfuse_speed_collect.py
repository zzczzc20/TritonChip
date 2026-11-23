#!/usr/bin/env python3
"""Collect the compact result of a run_graphs.py output directory (domain: model prediction).

    python3 tools/r42/kfuse_speed_collect.py TLM_OUT_DIR OUT.json

Reads results.json and the per-point record.json / counters.json of one kfuse_speed_tlm_run.py output and writes, per point:
status, serial-grid device cycles of the steady call, the per-launch list (kernel, grid, cycles) and the summed core
counters. The raw run directories stay on the host that produced them (their sha256 is not needed: every number here is
read from the files named in `source`).
"""
import glob
import json
import sys
from pathlib import Path

CORE_KEYS = ('instructions', 'matrix_engine_busy_cycles', 'matrix_engine_commands', 'barrier_sleep_cycles',
             'dependency_wait_cycles', 'dma_admission_wait_cycles', 'lsu_wait_cycles', 'pipeline_freeze_cycles',
             'context_park_cycles', 'mma_commands', 'issue_queue_cycles', 'memory_port_service_cycles',
             'dma_commands_issued')


def main():
    root = Path(sys.argv[1])
    res = json.loads((root / 'results.json').read_text())
    out = dict(source=str(root), binary=res.get('binary'), points={})
    for p in res['points']:
        rec = {k: p.get(k) for k in ('status', 'device_cycles', 'first_call_device_cycles', 'launches', 'max_rel_err',
                                    'timing', 'model_retired', 'process_wall_s', 'problems', 'fallbacks')}
        rec['model_knobs'] = p.get('model_knobs')
        rf = p.get('record')
        if rf and Path(rf).exists():
            r = json.loads(Path(rf).read_text())
            rec['per_launch'] = (r.get('time') or {}).get('per_launch')
            rec['demand'] = r.get('demand')
            rec['bound'] = (r.get('bound') or {}).get('value')
        counters = {}
        for d in p.get('model_dirs') or []:
            cf = Path(d) / 'counters.json'
            if cf.exists():
                c = json.loads(cf.read_text())
                for k in CORE_KEYS:
                    counters[k] = sum(v for kk, v in c.items() if kk.startswith('core_') and kk.endswith('_' + k))
        rec['counters_total'] = counters
        out['points'][p['point'] if 'point' in p else p['key']] = rec
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(len(out['points']), 'points ->', sys.argv[2])


if __name__ == '__main__':
    main()
