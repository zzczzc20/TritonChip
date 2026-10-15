#!/usr/bin/env python3
"""Select active cores for the unchanged native graph using device events.

Functional policy integration only: this is not the frozen holdout gate.
"""
import hashlib
import json
import os
import inspect
import argparse
import tempfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm')]
os.environ.update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND='fpga-multicore-sim',
                  TNCA_BACKEND_ARGS=json.dumps(dict(core_count=3,memory_ports=3,maintenance=True,hbm_dma=True,
                      noc_link_stages=2,dispatch_stages=2,home_stages=2,code_stages=2,board_top=True)),
                  TNCA_RCTX_SPMD='1')
os.environ.setdefault('TNCA_HOST_TIMEOUT','1800')
import torch
import triton
import tnca.torch_device as D
from tnca.autotune import GraphBudget,PersistentCache,measure_native,select,run_measurement
from tnca.triton_backend.driver import launch_log


def original(a,b):
    return (a+b)*.5


def source_digest(paths):
    digest=hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode()+b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isolated-workers',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--measure-worker',type=int,choices=(1,3))
    parser.add_argument('--worker-result',type=Path)
    args=parser.parse_args()
    budget=GraphBudget()
    if args.isolated_workers and (args.out is None or args.out.exists()):
        parser.error('isolated worker evidence requires a new --out directory')
    if args.out:args.out.mkdir(parents=True)
    D.register();torch.manual_seed(49)
    x,y=torch.randn(256),torch.randn(256)
    dx,dy=x.to('tnca'),y.to('tnca')
    graph=torch.compile(original,fullgraph=True,dynamic=False)
    if not args.isolated_workers:
        torch.testing.assert_close(graph(dx,dy).cpu(),original(x,y))
    session=D.MEMORY.session()
    if args.measure_worker is not None:
        if args.worker_result is None or args.worker_result.exists():
            parser.error('worker requires a new result path')
        session.active_cores=(1<<args.measure_worker)-1
        outputs=[]
        records=measure_native(lambda:outputs.append(graph(dx,dy)))
        for output in outputs:torch.testing.assert_close(output.cpu(),original(x,y))
        args.worker_result.write_text(json.dumps(records)+'\n')
        print('ISOLATED NATIVE WORKER PASS cores='+str(args.measure_worker),flush=True)
        return
    identity=dict(kernel_source=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                  shape=[list(dx.shape),list(dy.shape)],stride=[list(dx.stride()),list(dy.stride())],
                  dtype=str(dx.dtype),software_tree=source_digest(
                      list((ROOT/'python').rglob('*.py'))+list((ROOT/'compiler/python').rglob('*.py'))),
                  compiler=dict(torch=torch.__version__,triton=triton.__version__),profile=session.profile,
                  capabilities=session.dev.capabilities(),rtl=dict(cores=3,ports=3,pipeline=2,
                      source_sha=source_digest(list((ROOT/'rtl').rglob('*.v'))+list((ROOT/'rtl').rglob('*.vh')))),
                  memory_map=session.dev.axi_window,image=dict(kind='functional-RTL-model',
                      harness_sha=source_digest(list((ROOT/'tb').rglob('*.v'))+list((ROOT/'tb').rglob('*.vh')))),
                  clock_hz=100000000)
    candidates=[dict(knobs={'cores':n},resources={'cores':n},requires=['physical_grid']) for n in (1,3,8)]
    calls=[]
    def measure(candidate):
        count=candidate['knobs']['cores'];calls.append(count)
        if args.isolated_workers:
            return run_measurement([sys.executable,str(Path(__file__).resolve()),
                '--measure-worker',str(count),'--worker-result',str(args.out.resolve()/f'{count}.json')],
                budget=budget,result_path=args.out/f'{count}.json',
                log_path=args.out/f'{count}.log',cwd=ROOT,env=dict(os.environ))
        session.active_cores=(1<<count)-1
        outputs=[]
        records=measure_native(lambda:outputs.append(graph(dx,dy)))
        for output in outputs:
            torch.testing.assert_close(output.cpu(),original(x,y))
        return records
    cache_directory=tempfile.TemporaryDirectory(prefix='tnca-native-tuning-cache-')
    cache=PersistentCache(cache_directory.name)
    with launch_log() as outer:
        result=select(identity,candidates,capacities={'cores':3},capabilities=['physical_grid'],
                      budget=budget,measure=measure,cache=cache)
    assert not outer,'tuning polluted the formal launch log'
    assert calls==[1,3] and result['rows'][2]['status']=='refused',result
    hit=select(identity,candidates,capacities={'cores':3},capabilities=['physical_grid'],
               budget=GraphBudget(),measure=lambda c:(_ for _ in ()).throw(AssertionError('cache reran')),
               cache=PersistentCache(cache_directory.name))
    assert hit['cache_hit'] and hit['selected']==result['selected']
    session.active_cores=(1<<result['selected']['knobs']['cores'])-1
    with launch_log() as formal:
        output=graph(dx,dy)
    torch.testing.assert_close(output.cpu(),original(x,y))
    assert formal and all(row['upload_bytes']==row['download_bytes']==0 for row in formal)
    cache_directory.cleanup()
    print('AUTOTUNE NATIVE DEVICE EVENTS PASS '+json.dumps(dict(identity=identity,
        identity_key=result['identity_key'],device_records=[r.get('device_records') for r in result['rows']],selected=result['selected'],
        seconds=[row.get('seconds') for row in result['rows']],cache_hit=hit['cache_hit'],
        measured_candidates=len(calls),formal_launches=len(formal),
        isolated_workers=args.isolated_workers,tuning_seconds=result['tuning_seconds']),sort_keys=True),flush=True)


if __name__=='__main__':main()
