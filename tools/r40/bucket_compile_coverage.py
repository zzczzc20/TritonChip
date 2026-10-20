#!/usr/bin/env python3
"""Compile the full frozen model's padded prefill and bucket decode on GB10.

This is a compiler preflight, not numerical or native-device evidence. Every
Triton kernel is lowered and its real container retained, while session opens
are explicitly forbidden. It avoids discovering unsupported lowering on the
FPGA host after an expensive full functional queue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--software-tree',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    root=a.source_root.resolve()
    sys.path[:0]=[str(root),str(root/'python'),str(root/'compiler/python'),str(root/'tools/asm'),str(root/'tools/r40')]
    os.environ.update(CUDA_VISIBLE_DEVICES='',TNCA_LAUNCH='skip',TNCA_TORCH_DEVICE='cpu',TNCA_PROFILE='core-vrf-sync-v5',TNCA_TRITON='1',TNCA_KERNEL_DUMP='1')
    report=dict(status='incomplete',software_tree=a.software_tree,phases=[],required_gate_promotions=[],
                numerical_correctness='not-measured',native_execution='not-measured',performance='not-measured',
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        q=a.out/'report.tmp';q.write_text(json.dumps(report,indent=2)+'\n');q.replace(a.out/'report.json')
    save()
    try:
        import torch
        import triton
        import tnca.inductor
        from tnca.triton_backend import driver
        from tnca.artifact import load
        from archbench import registry
        from archbench.ops import rope_tables
        from archbench.provenance import software_tree,GraphCapture,write_bundle,check_bundle
        from bucket_decode import GenerationPrefillState,BucketGenerateStep
        from torch._inductor.utils import run_and_get_code
        import contract_v2
        contract_v2.verify();assert software_tree(refresh=True)==a.software_tree
        workload=json.loads((root/'spec/r40/v2/plan.json').read_text())['workloads']['llama_generate']
        point=registry.get(workload['points']['prefill']['id'])
        assert point.definition_sha256()==workload['points']['prefill']['sha256']
        model,(prompt,)=point.make(0);model.eval()
        assert sum(t.numel()*t.element_size() for t in model.parameters())==221267968
        capacity=workload['prefill']+workload['decode_steps']
        c=model.c;cos,sin=rope_tables(capacity,c.hd,'fp16')
        keys=torch.zeros((c.n_layers,prompt.shape[0],c.n_kv_heads,capacity,c.hd),dtype=torch.float16)
        values=torch.zeros_like(keys);position=torch.tensor([workload['prefill']],dtype=torch.int64)
        modules=[('prefill',GenerationPrefillState(model,capacity),(prompt,)),
                 ('decode',BucketGenerateStep(model,cos,sin,capacity),(prompt[:,:1],keys,values,position))]
        report.update(workload=workload,contract_sha256=contract_v2.LOCK_SHA256,bucket_capacity=capacity,
                      generation_harness_sha256=hashlib.sha256((root/'tools/r40/bucket_decode.py').read_bytes()).hexdigest(),
                      torch=str(torch.__version__),triton=str(triton.__version__))
        def forbidden(*args,**kwargs):raise RuntimeError('compiler preflight forbids every device session')
        driver._Sessions.get=classmethod(forbidden)
        # The production utility gets these profile-only fields by first
        # opening a native heap. Compiler preflight supplies the identical
        # declared fields directly; it does not model or execute memory.
        from tnca import device
        from tnca.adapters import r1
        target,_info=r1.target_for(device.base_profile('core-vrf-sync-v5'))
        declared=dict(max_shared_mem=0,
            multiprocessor_count=int(getattr(target,'num_cores',1) or 1),
            max_num_regs=int(target.phys_vregs),warpSize=1,
            sm_clock_rate=0,mem_clock_rate=0,mem_bus_width=64,
            max_threads_per_block=int(getattr(target,'vlen',16) or 16))
        report['compile_target_properties']=declared
        report['target_property_source']='immutable compile profile, declared input; no live heap or hardware measurement'
        driver.TNCAUtils.get_device_properties=lambda self,index:dict(declared)
        original=driver.TNCALauncher.__call__;captured=[]
        def capture(self,*args,**kwargs):
            assert os.environ['TNCA_LAUNCH']=='skip'
            function=Path(args[4] if len(args)>4 else kwargs['function'])
            captured.append(function)
            return original(self,*args,**kwargs)
        driver.TNCALauncher.__call__=capture
        tnca.inductor.configure()
        for name,module,args in modules:
            dump=a.out/name/'compiler';os.environ['TNCA_DUMP_DIR']=str(dump);captured.clear()
            compiled=torch.compile(module.eval(),fullgraph=True,dynamic=False)
            with torch.no_grad(),GraphCapture() as graph,driver.launch_log() as log:
                _output,codes=run_and_get_code(compiled,*args)
            assert codes and log and len(log)==len(captured)
            assert all(e.get('skipped') is True and e['device_cycles'] is None for e in log)
            assert not tnca.inductor.host_fallbacks(codes)
            for event,path in zip(log,captured):
                raw=path.read_bytes();load(path)  # Validate the actual sealed compiler output.
                event.update(artifact_path=str(path),artifact=hashlib.sha256(raw).hexdigest())
            bundle=write_bundle(a.out/name/'bundle',point_id=point.id+':'+name,
                                graph_text=graph.text,wrapper_code=codes,launches=log,dump_dir=dump)
            assert not check_bundle(bundle)
            report['phases'].append(dict(name=name,status='compiled-not-executed',launches=log,bundle=bundle));save()
            del compiled,_output
        assert not driver._Sessions.open
        assert software_tree(refresh=True)==a.software_tree
        report['status']='full-frozen-model-compiler-preflight-pass'
    except Exception:
        report.update(status='failed-preserved',error=traceback.format_exc());save();raise
    save()
    print('FULL FROZEN BUCKET COMPILER PREFLIGHT PASS; no device opens, no numerical/performance claim',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
