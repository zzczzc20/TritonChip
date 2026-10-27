#!/usr/bin/env python3
"""Two real standard Torch graphs through captured native locality off/on.

All numerical work is Inductor/Triton/native Session execution. Metadata AOT
helpers only access TMU controls; CPU references run outside timing windows.
"""
import argparse,hashlib,json,os
from pathlib import Path
import sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm')]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(TNCA_TORCH_DEVICE='native',TNCA_TRANSFER_POLICY='resident',TNCA_SUBMIT='batched',
        TNCA_TUNE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TRITON_CACHE_DIR=str(out/'triton-cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE','core-vrf-sync-v5')
    report=dict(schema='tnca-r41-locality-case/1',status='running',seed=4103,rtol=.002,atol=.002,
        board_acceptance='not-measured',measurement='native numerical replay plus separately reported metadata scope; CPU reference/download after counter snapshot',graphs=[])
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch,triton
        import tnca.inductor
        import tnca.torch_device as D
        from tnca.triton_backend.driver import launch_log
        from archbench.provenance import software_tree
        D.register();torch.manual_seed(4103);source=software_tree(ROOT,refresh=True)
        report.update(software_tree=source,torch=str(torch.__version__),triton=str(triton.__version__),backend=os.environ.get('TNCA_BACKEND','sim-file'))
        def residual(a,b):return torch.relu(a+b)*.5
        def projection(a,w,b):return torch.relu(a@w+b)
        definitions=[('residual',residual,[torch.randn(128),torch.randn(128)]),
            ('projection',projection,[(torch.randn(8,16)*.125).half(),(torch.randn(16,16)*.125).half(),(torch.randn(16)*.125).half()])]
        (out/'registration.json').write_text(json.dumps(dict(seed=4103,graphs=[dict(name=n,shapes=[list(x.shape) for x in xs],dtypes=[str(x.dtype) for x in xs]) for n,_,xs in definitions],modes=[False,True,False],policy='LRU only, no complete-access or dead/bypass claims',rtol=.002,atol=.002),indent=2)+'\n')
        for name,fn,host in definitions:
            entry=dict(name=name,inputs=[dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()) for x in host],runs=[])
            report['graphs'].append(entry);save()
            native=[x.to('tnca') for x in host]
            compiled=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
            compiled(*native).cpu()
            graph=torch.tnca.Graph()
            with launch_log() as capture:
                with torch.tnca.graph(graph):output=compiled(*native)
            entry['capture']=capture;entry['plan']=graph.locality_plan().report()
            session=D.MEMORY.session();entry['capabilities']=session.dev.capabilities()
            for mode in (False,True,False):
                before=session.dev.memory_port_metrics();started=time.perf_counter()
                with launch_log() as launches:
                    event=graph.replay(locality=mode)
                    D.MEMORY.synchronize()
                elapsed=time.perf_counter()-started;after=session.dev.memory_port_metrics()
                assert event.ok and launches
                assert all(e.get('replay') and e['upload_bytes']==e['download_bytes']==0 for e in launches)
                assert [e['artifact'] for e in launches]==[e['artifact'] for e in capture]
                expected=fn(*host);actual=output.cpu()
                torch.testing.assert_close(actual,expected,rtol=.002,atol=.002)
                row=dict(locality=mode,status='pass',host_scope_seconds=elapsed,launches=launches,
                    counters_before=before,counters_after=after,output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest())
                if mode:
                    row['controls']=graph.last_locality_report
                    assert row['controls']['status']=='completed'
                    assert row['controls']['registered']['policy']==1 and row['controls']['restored']['policy']==0
                    assert not row['controls']['plan']['aggressive_hints']
                entry['runs'].append(row);save()
            assert len({r['output_sha256'] for r in entry['runs']})==1
            assert all(not b.leases for b in session._buffers.values()) and not session.unfinished
            graph.reset();entry['status']='functional-pass';save()
        assert software_tree(ROOT,refresh=True)==source,'source changed during run'
        report['status']='functional-pass'
    except BaseException:
        report.update(status='failed-preserved',error=traceback.format_exc());raise
    finally:save()
    print(json.dumps(dict(status=report['status'],graphs=len(report['graphs']))))
if __name__=='__main__':main()
