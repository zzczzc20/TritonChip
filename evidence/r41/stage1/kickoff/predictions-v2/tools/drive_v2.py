#!/usr/bin/env python3
"""Seal v2 prospective predictions on GB10: run_graphs.py over the board manifests' exact knobs.

    drive_v2.py CHAIN        CHAIN in medium, hbm-fast, hbm-slow

Every configuration is a (point, knob set) of spec/r41/stage1/experiments/
board-prospective.json or board-gemm-tile.json in the frozen c2c96d3 tree.
All runs: the B40 physical DRAM map, timed cold call then one measured steady
call (as the board's formal calls), except the HBM-scale points whose cold call
runs untimed (recorded per run; 48 KiB of cache state is irrelevant to their
~100 MB streams). Runs are confined to cores 16-19.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

H = Path.home()/'r41s1-c2c96d3'
SRC = H/'src'
PY = str(Path.home()/'tc_env/torchenv/bin/python')
BIN = H/'model-build/tnca-chip'
MAP = H/'map.txt'
CFG = dict(fast=SRC/'tools/performance_harness/systemc/configs/r40-board-m2.cfg',
           slow=SRC/'tools/performance_harness/systemc/configs/r40-board-m3.cfg')
HBM = ('op.matmul/hbm/', 'op.linear/hbm/', 'op.gemv/hbm/')


def configurations():
    groups = {}
    def add(name, cfgs):
        knobs = {json.dumps(c['knobs'], sort_keys=True) for c in cfgs}
        assert len(knobs) == 1, (name, knobs)
        groups[name] = dict(points=[c['point'] for c in cfgs], knobs=cfgs[0]['knobs'])
    pro = json.loads((SRC/'spec/r41/stage1/experiments/board-prospective.json').read_text())
    for c in pro['board_batches']['prospective']['configurations']:
        add(f'prospective-{c["arm"]}', [c])
    gem = json.loads((SRC/'spec/r41/stage1/experiments/board-gemm-tile.json').read_text())
    cfgs = gem['board_batches']['gemm']['configurations']
    for arm in ('control', 'candidate'):
        add(f'gemm-medium-{arm}', [c for c in cfgs if c['arm'] == arm and not c['point'].startswith(HBM)])
        add(f'gemm-hbm-{arm}', [c for c in cfgs if c['arm'] == arm and c['point'].startswith(HBM)])
    return groups


def invoke(name, group, path, *, jobs, timeout, untimed_cold, suffix=''):
    out = H/'runs'/f'{name}-{path}{suffix}'
    if (out/'results.json').exists() and (out/'DONE').exists():
        return
    cmd = ['taskset', '-c', '16-19', 'nice', PY, str(SRC/'tools/performance_harness/systemc/run_graphs.py'),
           '--source-root', str(SRC), '--binary', str(BIN), '--config', str(CFG[path]), '--out', str(out),
           '--jobs', str(jobs), '--timeout', str(timeout), '--env', f'TNCA_DRAM_MAP={MAP}']
    for point in group['points']:
        cmd += ['--point', point]
    for k, v in sorted(group['knobs'].items()):
        cmd += ['--env', f'{k}={v}']
    if untimed_cold:
        cmd.append('--untimed-cold')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    (H/'runs').mkdir(exist_ok=True)
    log = H/'runs'/f'{name}-{path}{suffix}.log'
    started = time.time()
    with log.open('w') as fh:
        fh.write(json.dumps(dict(cmd=cmd, started=started)) + '\n')
        fh.flush()
        code = subprocess.run(cmd, cwd=H, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    (out/'DONE').write_text(json.dumps(dict(returncode=code, seconds=time.time()-started)) + '\n')


def main():
    chain = sys.argv[1]
    groups = configurations()
    (H/'groups.json').write_text(json.dumps(groups, indent=1, sort_keys=True) + '\n')
    if chain == 'medium':
        for path in ('fast', 'slow'):
            for name in ('prospective-candidate', 'prospective-control', 'gemm-medium-control', 'gemm-medium-candidate'):
                invoke(name, groups[name], path, jobs=2, timeout=7200, untimed_cold=False)
    elif chain == 'hbm-fast':
        for name in ('gemm-hbm-control', 'gemm-hbm-candidate'):
            invoke(name, groups[name], 'fast', jobs=2, timeout=10800, untimed_cold=True)
    elif chain == 'point':
        # One configuration: drive_v2.py point PATH GROUP POINT TIMEOUT
        path, name, point, timeout = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        assert path in CFG and point in groups[name]['points']
        invoke(name, dict(groups[name], points=[point]), path, jobs=1, timeout=timeout,
               untimed_cold=name.startswith('gemm-hbm'), suffix='--' + point.replace('/', '__'))
    elif chain == 'slow-point':
        # One HBM-scale slow-path configuration: drive_v2.py slow-point GROUP POINT TIMEOUT
        name, point, timeout = sys.argv[2], sys.argv[3], int(sys.argv[4])
        assert point in groups[name]['points']
        invoke(name, dict(groups[name], points=[point]), 'slow', jobs=1, timeout=timeout, untimed_cold=True,
               suffix='--' + point.replace('/', '__'))
    elif chain == 'hbm-slow':
        for name in ('gemm-hbm-control', 'gemm-hbm-candidate'):
            invoke(name, groups[name], 'slow', jobs=2, timeout=10800, untimed_cold=True)
    else:
        raise SystemExit('unknown chain')


if __name__ == '__main__':
    main()
