#!/usr/bin/env python3
"""Schedule the remaining HBM-scale runs: at most 2 HBM-scale model processes at once.

Fast candidates first; then slow runs, smallest estimate first. A slow run is
started only when its estimate (its own fast-path model seconds times the
largest slow/fast ratio measured so far on HBM-tier points, 7.5 before any) is
at most 3 h; otherwise it is recorded as not-run with that estimate. Started
slow runs have a 3 h run_graphs timeout. Decisions go to hbm_sched.json.
"""
import json
import os
from pathlib import Path
import subprocess
import time

H = Path.home()/'r41s1-c2c96d3'
PY = str(Path.home()/'tc_env/torchenv/bin/python')
BUDGET = 3*3600
POINTS = ['op.gemv/hbm/fp16-k4096-n11008', 'op.linear/hbm/fp16-m512-k4096-n11008',
          'op.matmul/hbm/fp16-m4096-k4096-n4096']
STATE = H/'hbm_sched.json'


def active():
    n = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'run_graphs.py --child' in cmd and '/runs/gemm-hbm' in os.readlink(p/'cwd'):
                n += 1
        except OSError:
            continue
    return n


def row(group, path, point):
    for d in [H/'runs'/f'{group}-{path}--{point.replace("/", "__")}', H/'runs'/f'{group}-{path}']:
        f = d/'results.json'
        if f.exists():
            for r in json.loads(f.read_text())['points']:
                if r['key'] == point:
                    return r
    return None


def ratio():
    values = []
    for group in ('gemm-hbm-control', 'gemm-hbm-candidate'):
        for point in POINTS:
            fast, slow = row(group, 'fast', point), row(group, 'slow', point)
            if fast and slow and fast.get('status') == slow.get('status') == 'pass':
                values.append(slow['model_launch_host_s']/fast['model_launch_host_s'])
    return max(values) if values else 7.5


def launch(group, path, point, timeout):
    subprocess.Popen(['setsid', '-f', PY, str(H/'drive_v2.py'), 'point', path, group, point, str(timeout)],
                     cwd=H, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


state = json.loads(STATE.read_text()) if STATE.exists() else dict(started={}, not_run={}, log=[])
queue = [('gemm-hbm-candidate', 'fast', p) for p in POINTS]
queue += [(g, 'slow', p) for p in POINTS for g in ('gemm-hbm-control', 'gemm-hbm-candidate')]
while True:
    pending = [t for t in queue if f'{t[0]}/{t[1]}/{t[2]}' not in state['started'] and f'{t[0]}/{t[1]}/{t[2]}' not in state['not_run']]
    if not pending:
        break
    if active() >= 2:
        time.sleep(30)
        continue
    # A slow run needs its own fast result first; choose the smallest estimate.
    ready = []
    for t in pending:
        group, path, point = t
        if path == 'fast':
            ready.append((0, t, None))
            continue
        fast = row(group, 'fast', point)
        if fast is None:
            continue
        estimate = fast['model_launch_host_s']*ratio() if fast.get('status') == 'pass' else None
        ready.append((estimate if estimate is not None else 0, t, estimate))
    if not ready:
        time.sleep(30)
        continue
    ready.sort(key=lambda x: (x[1][1] != 'fast', x[0]))
    _, (group, path, point), estimate = ready[0]
    key = f'{group}/{path}/{point}'
    if path == 'slow' and estimate is not None and estimate > BUDGET:
        state['not_run'][key] = dict(estimate_seconds=round(estimate), ratio=round(ratio(), 2),
            reason=(f'slow path estimated at {estimate/3600:.1f} h (fast-path model time '
                    f'{row(group, "fast", point)["model_launch_host_s"]:.0f} s x measured slow/fast ratio '
                    f'{ratio():.2f}) exceeds the 3 h budget; not run'))
    else:
        launch(group, path, point, BUDGET if path == 'slow' else 3*BUDGET)
        state['started'][key] = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), estimate_seconds=estimate,
                                     ratio=round(ratio(), 2))
        time.sleep(90)
    STATE.write_text(json.dumps(state, indent=1)+'\n')
STATE.write_text(json.dumps(dict(state, finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1)+'\n')
