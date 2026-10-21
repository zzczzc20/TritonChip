#!/usr/bin/env python3
"""Longest-first scheduling of the remaining HBM-tier runs.

Lanes: at most 2 HBM-scale (linear/hbm, matmul/hbm) model processes, and at
most 1 gemv/hbm process (a ~20 s fast-path point, not HBM-scale). All runs
share the 4 cores 16-19 (run_graphs is started under taskset by drive_v2.py).
A slow run starts only when its estimate -- its fast-path model seconds times
the largest slow/fast model-time ratio measured on an HBM-tier point (7.5
before any) -- is at most 3 h; otherwise it is recorded as not-run with the
estimate. Started slow runs have a 3 h run_graphs timeout.
"""
import json
import os
from pathlib import Path
import subprocess
import time

H = Path.home()/'r41s1-c2c96d3'
PY = str(Path.home()/'tc_env/torchenv/bin/python')
BUDGET = 3*3600
GEMV, LINEAR, MATMUL = ('op.gemv/hbm/fp16-k4096-n11008', 'op.linear/hbm/fp16-m512-k4096-n11008',
                        'op.matmul/hbm/fp16-m4096-k4096-n4096')
STATE = H/'hbm_sched.json'
C, K = 'gemm-hbm-control', 'gemm-hbm-candidate'
# Priority within each lane: the critical path (longest) first.
BIG = [(C, 'slow', MATMUL), (K, 'fast', MATMUL), (C, 'slow', LINEAR), (K, 'slow', LINEAR), (K, 'slow', MATMUL)]
SMALL = [(C, 'slow', GEMV), (K, 'slow', GEMV)]


def active(points):
    n = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(p/'cwd')
        except OSError:
            continue
        if 'run_graphs.py --child' in cmd and '/runs/gemm-hbm' in cwd and any(x.replace('/', '__') in cwd for x in points):
            n += 1
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
    for group in (C, K):
        for point in (GEMV, LINEAR, MATMUL):
            fast, slow = row(group, 'fast', point), row(group, 'slow', point)
            if fast and slow and fast.get('status') == slow.get('status') == 'pass':
                values.append(slow['model_launch_host_s']/fast['model_launch_host_s'])
    return max(values) if values else 7.5


def launch(group, path, point, timeout):
    subprocess.Popen(['setsid', '-f', PY, str(H/'drive_v2.py'), 'point', path, group, point, str(timeout)],
                     cwd=H, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


state = json.loads(STATE.read_text())
state.setdefault('log', []).append(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    note='longest-first scheduler replaced the first one; the first candidate matmul fast start was preempted '
         'after 80 s (preserved as runs/preempted-gemm-hbm-candidate-fast--op.matmul-1) and is requeued'))
state['started'].pop(f'{K}/fast/{MATMUL}', None)
STATE.write_text(json.dumps(state, indent=1)+'\n')


def key(t):
    return f'{t[0]}/{t[1]}/{t[2]}'


while True:
    progressed = False
    for lane, limit, points in ((BIG, 2, (LINEAR, MATMUL)), (SMALL, 1, (GEMV,))):
        pending = [t for t in lane if key(t) not in state['started'] and key(t) not in state['not_run']]
        if not pending or active(points) >= limit:
            continue
        for group, path, point in pending:
            if path == 'slow':
                fast = row(group, 'fast', point)
                if fast is None:
                    continue          # wait for its fast result
                if fast.get('status') != 'pass':
                    state['not_run'][key((group, path, point))] = dict(reason='fast path did not pass; slow not run')
                    progressed = True
                    break
                estimate = fast['model_launch_host_s']*ratio()
                if estimate > BUDGET:
                    state['not_run'][key((group, path, point))] = dict(
                        estimate_seconds=round(estimate), ratio=round(ratio(), 2),
                        reason=(f'slow path estimated at {estimate/3600:.1f} h (fast-path model time '
                                f'{fast["model_launch_host_s"]:.0f} s x largest measured slow/fast ratio '
                                f'{ratio():.2f}) exceeds the 3 h budget; not run'))
                    progressed = True
                    break
                launch(group, path, point, BUDGET)
                state['started'][key((group, path, point))] = dict(
                    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), estimate_seconds=round(estimate),
                    ratio=round(ratio(), 2))
            else:
                launch(group, path, point, 3*BUDGET)
                state['started'][key((group, path, point))] = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            progressed = True
            break
        STATE.write_text(json.dumps(state, indent=1)+'\n')
    remaining = [t for t in BIG + SMALL if key(t) not in state['started'] and key(t) not in state['not_run']]
    if not remaining:
        break
    time.sleep(60 if progressed else 30)
STATE.write_text(json.dumps(dict(state, finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1)+'\n')
