#!/usr/bin/env python3
"""Collect the frozen three-block CPU/native comparison on the final board.

No point/size/sample-count override. Every side runs in an independent process;
the same immutable source, image and configuration are required throughout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]
import contract_v2
from archbench.provenance import software_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xclbin', type=Path, required=True)
    ap.add_argument('--image-sha256', required=True)
    ap.add_argument('--software-tree-sha256', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hbm-bytes',type=int,default=1<<30)
    ap.add_argument('--evidence', action='store_true')
    ap.add_argument('--operating-clock-evidence',type=Path)
    args = ap.parse_args()
    if args.hbm_bytes < 1<<30 or args.hbm_bytes % 64:
        ap.error('--hbm-bytes must be line aligned and at least 1 GiB')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', performance='not-measured', gate_status='not-measured')
    try:
        contract_v2.verify()
        assert software_tree() == args.software_tree_sha256, 'source identity mismatch'
        assert hashlib.sha256(args.xclbin.read_bytes()).hexdigest() == args.image_sha256, 'image identity mismatch'
        if args.operating_clock_evidence:
            os.environ['TNCA_OPERATING_CLOCK_EVIDENCE']=str(args.operating_clock_evidence.resolve())
        from tnca.operating_clock import from_environment
        operating_clock=from_environment(args.image_sha256)
        plan = json.loads((ROOT/'spec/r40/v2/plan.json').read_text())
        exp = plan['experiments']['R40-01.device-time']
        points = sorted(json.loads((ROOT/exp['subset']).read_text())['points'], key=lambda p:p['id'])
        assert len(points) == 77
        rng = random.Random(plan['sample_plan']['bootstrap']['seed'])
        config = dict(xclbin=str(args.xclbin.resolve()), image_profile='core-vrf-sync-v5',
                      core_count=3, memory_ports=3, hbm_dma=True, hbm_bytes=args.hbm_bytes,
                      device_clock=True)
        report['identity'] = dict(software_tree=args.software_tree_sha256,
                                image_sha256=args.image_sha256,
                                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                contract_sha256=contract_v2.LOCK_SHA256,
                                backend='fpga-u280-multicore', backend_args=config,
                                clock_policy='100MHz-undivided', clock_hz=100000000)
        report['identity'].update(operating_clock)
        report['identity']['toolchain'] = dict(torch=version('torch'),triton=version('triton'),python=sys.version)
        (args.out/'identity.json').write_text(json.dumps(report['identity'], indent=2)+'\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', R40_CONTRACT='v2', TNCA_ACTIVE_CORES='3')
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'python'),
                                           str(ROOT/'compiler/python'),str(ROOT/'tools/asm')])
        paths = {side:[] for side in ('control','native')}
        for block in range(3):
            for side in paths:
                path = args.out/f'{side}{block}.jsonl'
                path.touch(exist_ok=False)
                paths[side].append(path)
            for index, point in enumerate(points):
                order = ('control','native') if rng.randrange(2) == 0 else ('native','control')
                for side in order:
                    stem = f'block{block}-point{index:02d}-{side}'
                    raw = args.out/(stem+'.jsonl')
                    command = [sys.executable,'-u','-m','archbench.run',
                               '--platform','tnca','--backend','fpga-u280-multicore',
                               '--backend-args',json.dumps(config),'--profile','core-vrf-sync-v5',
                               '--point',point['id'],'--jobs','1','--seed','0',
                               '--warmup','5','--timed','30','--timeout','7200',
                               '--set','TNCA_TORCH_DEVICE='+('cpu' if side=='control' else 'native'),
                               '--set','TNCA_TRANSFER_POLICY=per-kernel',
                               '--out',str(raw.resolve())]
                    child_env = dict(env, ARCHBENCH_BLOCK_ID=str(block))
                    with (args.out/(stem+'.log')).open('w') as log:
                        done = subprocess.run(command,cwd=ROOT,env=child_env,
                                              stdout=log,stderr=subprocess.STDOUT)
                    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
                    assert len(rows)==1, f'{stem}: expected one original point record'
                    row = rows[0]
                    assert row['point_id']==point['id'] and row['point_sha256']==point['sha256']
                    p = row['platform']
                    assert p['software_tree']==args.software_tree_sha256
                    assert p['image']=='xclbin:'+args.image_sha256[:16]
                    assert p['image_sha256']==args.image_sha256
                    assert p['toolchain']==report['identity']['toolchain']
                    assert p['backend']=='fpga-u280-multicore'
                    with paths[side][-1].open('a') as aggregate:
                        aggregate.write(json.dumps(row)+'\n')
                    assert done.returncode==0 and row['status']=='pass', f'{stem}: preserve failed point; stop collection'
                    print(f'collected block={block} side={side} point={point["id"]}',flush=True)
        assert from_environment(args.image_sha256)==operating_clock, 'operating clock evidence changed during collection'
        command = [sys.executable,str(ROOT/'tools/r40/device_time_v2.py'),
                   '--control',*[str(p.resolve()) for p in paths['control']],
                   '--native',*[str(p.resolve()) for p in paths['native']],
                   '--json',str((args.out/'verdict.json').resolve())]
        if args.evidence:
            command.append('--evidence')
        with (args.out/'audit.log').open('w') as log:
            done = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['status'] = 'complete' if done.returncode==0 else 'audit-failed'
        report['gate_status'] = 'see verdict.json and audit.log'
    except Exception:
        report['status'] = 'collection-failed'
        report['error'] = traceback.format_exc()
    (args.out/'collection.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
