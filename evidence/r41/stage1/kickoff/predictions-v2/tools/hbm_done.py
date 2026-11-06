#!/usr/bin/env python3
"""Print the state of every HBM-scale (group, path, point); exit 0 when all are resolved."""
import json
from pathlib import Path
H = Path.home()/'r41s1-c2c96d3'
POINTS = ['op.gemv/hbm/fp16-k4096-n11008', 'op.linear/hbm/fp16-m512-k4096-n11008', 'op.matmul/hbm/fp16-m4096-k4096-n4096']
state = json.loads((H/'hbm_sched.json').read_text()) if (H/'hbm_sched.json').exists() else dict(started={}, not_run={})
open_ = 0
for group in ('gemm-hbm-control', 'gemm-hbm-candidate'):
    for path in ('fast', 'slow'):
        for point in POINTS:
            key = f'{group}/{path}/{point}'
            row = None
            for d in (H/'runs'/f'{group}-{path}--{point.replace("/", "__")}', H/'runs'/f'{group}-{path}'):
                if (d/'results.json').exists():
                    row = next((r for r in json.loads((d/'results.json').read_text())['points'] if r['key'] == point), None) or row
            if row:
                g = row.get('model_graph') or {}
                print(f'{key:70s} {row["status"]:14s} {g.get("graph_cycles")} {row.get("model_timing_status")} {row.get("model_launch_host_s")}')
            elif key in state['not_run']:
                print(f'{key:70s} not-run        {state["not_run"][key]["reason"]}')
            else:
                print(f'{key:70s} {"running" if key in state["started"] or (group.endswith("control") and path == "fast") else "queued"}')
                open_ += 1
raise SystemExit(1 if open_ else 0)
