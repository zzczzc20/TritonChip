"""Resident standard Torch service ledger under actual outlet/depth controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),
              str(ROOT/'tools/performance_harness/systemc')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ports',type=int,choices=(1,3),required=True)
    ap.add_argument('--outstanding',type=int,choices=(1,8),required=True)
    ap.add_argument('--elements',type=int,default=4097)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.elements<=0:ap.error('positive element count required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'artifacts').mkdir()
    os.environ.update(TNCA_RTL_MEM_PORTS=str(args.ports),TNCA_RTL_OUTSTANDING=str(args.outstanding),
        TNCA_ACTIVE_CORES='3',TNCA_RCTX_SPMD='1',TNCA_MAX_BLOCK='16',TNCA_TUNE='0',
        TNCA_TENSOR_DESCRIPTORS='1',TNCA_TORCH_DEVICE='native',
        TRITON_CACHE_DIR=str(out/'triton-cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor-cache'))
    os.environ.setdefault('TNCA_PROFILE','core-vrf-sync-v5')
    os.environ.setdefault('TNCA_BACKEND','systemc-tlm-grid')
    report=dict(schema='tnca-memory-service-case/1',status='running',ports=args.ports,
        outstanding=args.outstanding,elements=args.elements,measurement_boundary=
        'one resident steady graph after warm graph; no explicit CPU download between calls; '
        'backend ABI uploads and their ownership handoffs remain inside the native call path',
        cycle_metric='sum of completed launch device cycles; prelaunch uploads/maintenance are outside these launch windows; not whole-call latency',
        counter_boundary='snapshots surround the native call and can include prelaunch ABI maintenance',
        performance_gate='not-measured')
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        import triton
        import tnca.inductor
        import tnca.torch_device as device
        from tnca import artifact
        from tnca.triton_backend.driver import launch_log
        from archbench.provenance import software_tree
        backend=os.environ.get('TNCA_BACKEND','sim-file');model=backend.startswith('systemc-tlm')
        if model:
            import tlm_backend
            tlm_backend.register()
        device.register();torch.manual_seed(4006)
        source=software_tree(ROOT,refresh=True)
        a,b=torch.randn(args.elements),torch.randn(args.elements);reference=(a+b)*.5
        inputs=[]
        for name,t in (('a',a),('b',b)):
            data=t.numpy().tobytes();p=out/f'{name}.bin';p.write_bytes(data)
            inputs.append(dict(name=name,path=p.name,sha256=hashlib.sha256(data).hexdigest(),
                               shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype)))
        da,db=a.to('tnca'),b.to('tnca');session=device.MEMORY.session()
        report.update(backend=backend,software_tree=source,inputs=inputs,capabilities=session.dev.capabilities(),
            placements=[dict(base=t.untyped_storage().data_ptr(),offset=t.storage_offset()) for t in (da,db)],
            tools=dict(torch=str(torch.__version__),triton=str(triton.__version__)))
        fn=torch.compile(lambda a,b:(a+b)*.5,backend='inductor',fullgraph=True,dynamic=False)
        from torch._inductor.utils import run_and_get_code
        with launch_log(isolated=True) as cold:
            warm,wrappers=run_and_get_code(fn,da,db)
        if tnca.inductor.host_fallbacks(wrappers):raise ValueError('numerical host fallback emitted')
        for i,wrapper in enumerate(wrappers):(out/f'wrapper-{i}.py').write_text(wrapper)
        # Keep warm output resident to avoid an extra CPU-download handoff.
        # Native ABI uploads can still trigger backend ownership maintenance;
        # this is not a claim that the physical call starts with a warm cache.
        if model:
            live=session.dev.identity['config']
            assert int(live['active_outlets'])==args.ports and int(live['outstanding'])==args.outstanding,live
            before=session.dev.counters_snapshot()
        else:
            live=session.dev.memory_configuration()
            assert live['active_ports']==args.ports and live['outstanding']==args.outstanding,live
            before=session.dev.memory_port_metrics()
        with launch_log(isolated=True) as launches:
            result=fn(da,db)
        after=session.dev.counters_snapshot() if model else session.dev.memory_port_metrics()
        activity=None if model else session.dev.physical_grid_metrics()
        assert launches and all(e['upload_bytes']==e['download_bytes']==0 for e in launches),launches
        torch.testing.assert_close(warm.cpu(),reference,rtol=1e-5,atol=1e-5)
        actual=result.cpu();torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-5)
        report.update(cold_launches=cold,launches=launches,memory_configuration=live,
            counter_before=before,counter_after=after,physical_grid_activity=activity,
            output_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
            output_placement=dict(base=result.untyped_storage().data_ptr(),offset=result.storage_offset()),
            device_cycles=sum(e['device_cycles'] for e in launches))
        if model:
            from prediction_domain import timing_status
            delta={k:v-before.get(k,0) for k,v in after.items()};report['counter_delta']=delta
            report['model_identity']=session.dev.identity
            report['model_timing_status']=timing_status(after,slow=session.dev.identity['dram_path']=='slow')
            assert all(delta.get(f'core_{i}_programs_completed',0)>0 for i in range(3)),delta
            report['outlets']=[dict(port=p,read_lines=delta.get(f'outlet_{p}_reads',0),
                write_lines=delta.get(f'outlet_{p}_writes',0)) for p in range(3)]
        else:
            report['outlets']=[dict(port=b['port'],read_lines=b['read_lines']-a['read_lines'],
                write_lines=b['write_lines']-a['write_lines']) for a,b in zip(before,after)]
            assert activity['peak_active']==3 and all(c['issued']==c['completed'] for c in activity['cores']),activity
        active=[p for p in report['outlets'] if p['read_lines'] or p['write_lines']]
        assert len(active)==args.ports and sum(p['read_lines'] for p in active)>0,report['outlets']
        kernels=[]
        for entry in launches:
            manifest,segments=artifact.load(entry['artifact_path']);code=hashlib.sha256(segments['text']).hexdigest()
            shutil.copyfile(entry['artifact_path'],out/'artifacts'/f'{code}.tnca')
            kernels.append(dict(code_sha256=code,code_bytes=len(segments['text']),resources=manifest['resources'],
                plan=manifest['optional_metadata']['provenance'].get('execution_resources',{})))
        if software_tree(ROOT,refresh=True)!=source:raise ValueError('source changed during service case')
        report.update(status='functional-pass',artifacts=kernels)
    except Exception:
        report.update(status='failed-preserved',error=traceback.format_exc());raise
    finally:save()
    print(json.dumps(dict(status=report['status'],ports=args.ports,outstanding=args.outstanding,
        cycles=report['device_cycles'],outlets=report['outlets']),indent=2))


if __name__=='__main__':main()
