#!/usr/bin/env python3
"""Compare original full-size large Llama boundaries through standard Torch.

Extra returned boundaries can change fusion, so this is diagnostic evidence,
never replacement acceptance data. CPU reference work runs on GB10; matching
model/input byte hashes are mandatory before physical execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()+b'\0');h.update(tensor_sha(value).encode()+b'\n')
    return h.hexdigest()


def observer(torch,model,kind):
    class Boundaries(torch.nn.Module):
        def __init__(self):super().__init__();self.model=model
        def forward(self,token,keys=None,values=None):
            m=self.model;x=torch.nn.functional.embedding(token,m.tok)
            taps=[x];ks=[];vs=[]
            for i,layer in enumerate(m.layers):
                x,k,v=layer(x,m.cos,m.sin,keys[i] if kind=='decode' else None,
                            values[i] if kind=='decode' else None)
                taps.append(x);ks.append(k);vs.append(v)
            logits=m._head(x[:,-1:,:])
            return (*taps,logits,torch.stack(ks),torch.stack(vs))
    return Boundaries()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--kind',choices=('decode','prefill'),default='decode')
    p.add_argument('--prepare-cpu',action='store_true')
    p.add_argument('--check-cpu-half',action='store_true')
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-software-tree',help='Explicit baseline snapshot for a controlled software repair; model/input bytes must still match')
    p.add_argument('--original-graph',action='store_true',help='Execute the original frozen model forward without extra boundary outputs')
    p.add_argument('--export-inputs',action='store_true',help='Export original native-host seed-0 inputs/state without device execution')
    p.add_argument('--data',type=Path,help='Exact native-host dataset for offloaded CPU reference')
    p.add_argument('--inputs',type=Path,help='Byte-matched native input export; reuse an existing native weight dataset for another frozen point')
    p.add_argument('--xclbin',type=Path);p.add_argument('--image-sha256')
    p.add_argument('--software-tree',required=True)
    p.add_argument('--operating-clock-evidence',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare_cpu and a.export_inputs:p.error('CPU reference and export are separate modes')
    if a.data and not a.prepare_cpu:p.error('--data is only for the offloaded CPU reference')
    if a.inputs and not (a.prepare_cpu and a.data):p.error('--inputs requires --prepare-cpu and --data')
    a.out.mkdir(parents=True,exist_ok=False)
    root=a.source_root.resolve();sys.path[:0]=[str(root),str(root/'python'),str(root/'compiler/python'),str(root/'tools/asm'),str(root/'tools/r40')]
    report=dict(status='incomplete',scope='Original full shape with extra observable boundaries; fusion may change; no gate promotion',required_gate_promotions=[],harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.original_graph:
        report['scope']='Original frozen model forward and full shape, independent byte-matched FP32 reference; functional replay only, not performance acceptance'
    def save():
        q=a.out/'report.tmp';q.write_text(json.dumps(report,indent=2)+'\n');q.replace(a.out/'report.json')
    save()
    try:
        import torch
        import triton
        from archbench import registry
        from archbench.platforms.common import compare,ieee_fp32
        from archbench.provenance import software_tree
        import contract_v2
        contract_v2.verify();assert software_tree(refresh=True)==a.software_tree
        name=f'model.llama_{a.kind}/large/fp16-b1-'+('p512' if a.kind=='decode' else 's512')
        point=registry.get(name)
        expected=next(x for x in json.loads((root/'spec/r40/suites/large.json').read_text())['points'] if x['id']==name)
        assert point.definition_sha256()==expected['sha256']
        model,inputs=point.make(0);model.eval()
        if a.data:
            data=torch.load(a.data,map_location='cpu',weights_only=True)
            if not a.inputs:
                assert data['identity']['point_sha256']==point.definition_sha256()
            assert data['identity']['software_tree']==a.software_tree
            model.load_state_dict(data['state'],strict=True);inputs=tuple(data['inputs'])
            report['native_dataset_sha256']=file_sha(a.data)
            del data
        if a.inputs:
            exported=torch.load(a.inputs,map_location='cpu',weights_only=True)
            assert exported['identity']['point_sha256']==point.definition_sha256()
            assert exported['identity']['software_tree']==a.software_tree
            assert exported['identity']['model_state_sha256']==state_sha(model)
            inputs=tuple(exported['inputs'])
            assert exported['identity']['input_sha256']==[tensor_sha(t) for t in inputs]
            report['native_input_export_sha256']=file_sha(a.inputs)
        weight_sha=state_sha(model)
        input_shas=[tensor_sha(t) for t in inputs]
        report['identity']=dict(point_id=name,point_sha256=point.definition_sha256(),software_tree=a.software_tree,
            contract_sha256=contract_v2.LOCK_SHA256,model_state_sha256=weight_sha,input_sha256=input_shas,
            weight_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),
            input_shapes=[list(t.shape) for t in inputs],torch=str(torch.__version__),triton=str(triton.__version__))
        names=['embedding']+[f'layer{i}' for i in range(model.c.n_layers)]+['logits','new_keys','new_values']
        report['boundary_names']=names;save()
        if a.export_inputs:
            path=a.out/'original_data.pt'
            torch.save(dict(identity=report['identity'],state=model.state_dict(),inputs=inputs),path)
            report.update(status='original-native-inputs-exported',data_file=str(path),data_sha256=file_sha(path),data_bytes=path.stat().st_size)
        elif a.prepare_cpu:
            half=None
            if a.check_cpu_half:
                with torch.no_grad(),ieee_fp32():half=observer(torch,model,a.kind).eval()(*inputs)
            wrapped=observer(torch,model.float(),a.kind).eval()
            args=tuple(t.float() if t.is_floating_point() else t for t in inputs)
            start=time.monotonic()
            with torch.no_grad(),ieee_fp32():reference=wrapped(*args)
            path=a.out/'reference.pt';torch.save(dict(identity=report['identity'],names=names,values=reference),path)
            if half is not None:
                report['cpu_half_boundaries']=[dict(name=n,**compare([g],[r],point.tolerance)) for n,g,r in zip(names,half,reference)]
            report.update(status='cpu-reference-ready',reference_file=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_seconds=time.monotonic()-start)
        else:
            assert a.reference and a.xclbin and a.image_sha256 and a.operating_clock_evidence
            ref=torch.load(a.reference,map_location='cpu',weights_only=True)
            for key in ('point_id','point_sha256','contract_sha256','model_state_sha256','input_sha256'):
                assert ref['identity'][key]==report['identity'][key],f'CPU/native original data identity mismatch: {key}'
            assert ref['identity']['software_tree']==(a.reference_software_tree or a.software_tree)
            report['reference_identity']=ref['identity']
            assert ref['names']==names
            if a.original_graph:
                names=['logits','new_keys','new_values'] if a.kind=='decode' else ['logits']
                ref=dict(ref,values=[ref['values'][ref['names'].index(n)] for n in names])
                report['boundary_names']=names
            assert hashlib.sha256(a.xclbin.read_bytes()).hexdigest()==a.image_sha256
            from tnca.operating_clock import qualified_clock
            clock=qualified_clock(a.operating_clock_evidence,a.image_sha256)
            config=dict(xclbin=str(a.xclbin.resolve()),image_profile='core-vrf-sync-v5',core_count=3,memory_ports=3,hbm_dma=True,hbm_bytes=4<<30,device_clock=True)
            os.environ.update(TNCA_BACKEND='fpga-u280-multicore',TNCA_BACKEND_ARGS=json.dumps(config),TNCA_PROFILE='core-vrf-sync-v5',TNCA_ACTIVE_CORES='3',TNCA_TRANSFER_POLICY='resident',TNCA_TORCH_DEVICE='native',TNCA_DEADLINE='24000000000',TNCA_HOST_TIMEOUT='7200',TNCA_OPERATING_CLOCK_EVIDENCE=str(a.operating_clock_evidence),TNCA_KERNEL_DUMP='1',TNCA_DUMP_DIR=str(a.out/'compiler'))
            import tnca.inductor
            import tnca.torch_device as D
            from tnca.triton_backend.driver import launch_log
            from torch._inductor.utils import run_and_get_code
            tnca.inductor.configure();D.register()
            model=model.to('tnca')
            report['parameter_buffers']=[dict(name=n,address=t.data_ptr(),bytes=t.numel()*t.element_size()) for n,t in model.named_parameters()]
            wrapped=model if a.original_graph else observer(torch,model,a.kind).eval()
            args=tuple(t.to('tnca') for t in inputs);compiled=torch.compile(wrapped,fullgraph=True,dynamic=False)
            report['identity'].update(image_sha256=a.image_sha256,backend_args=config,**clock)
            report['calls']=[];save()
            for label in ('cold','steady'):
                with torch.no_grad(),launch_log() as log:
                    if label=='cold':got,codes=run_and_get_code(compiled,*args)
                    else:got=compiled(*args)
                if label=='cold':
                    for i,code in enumerate(codes):(a.out/f'wrapper_{i}.py').write_text(code)
                    assert not tnca.inductor.host_fallbacks(codes)
                actual=[t.cpu() for t in (got if isinstance(got,(tuple,list)) else (got,))]
                rows=[dict(name=n,**compare([g],[r],point.tolerance)) for n,g,r in zip(names,actual,ref['values'])]
                assert len(actual)==len(ref['values'])==len(names)
                report['calls'].append(dict(label=label,boundaries=rows,launches=log))
                print(label,[(r['name'],r['max_rel_err']) for r in rows],flush=True)
                save();torch.save(dict(names=names,values=actual),a.out/(label+'_outputs.pt'));del actual,got
            assert software_tree(refresh=True)==a.software_tree
            report['status']='original-graph-replay-pass' if a.original_graph else 'layer-diagnostic-complete'
            if a.original_graph:
                assert all(row['ok'] for call in report['calls'] for row in call['boundaries']), 'original frozen numerical tolerance failed'
    except Exception:
        report.update(status='failed-preserved',error=traceback.format_exc());save();raise
    save();return 0


if __name__=='__main__':raise SystemExit(main())
