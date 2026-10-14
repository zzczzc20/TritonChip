#!/usr/bin/env python3
"""Native Torch descriptor GEMM with bounded per-core role/pair partitioning."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(path))
parser=argparse.ArgumentParser();parser.add_argument('--cores',type=int,default=3,choices=(1,2,3,4,8))
parser.add_argument('--pair',action='store_true');args=parser.parse_args()
os.environ.update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND='sim-multicore',
                  TNCA_BACKEND_ARGS=json.dumps({'core_count':args.cores}),
                  TNCA_PHYSICAL_CHUNKS='1',TNCA_RCTX_SPMD='1',TNCA_RCTX_ROLES='2' if args.pair else '0')
os.environ.setdefault('TNCA_HOST_TIMEOUT','1800')
import torch
import triton
import tnca.torch_device as D
from tnca import artifact
from tnca.triton_backend.driver import launch_log
from tnca_isa import decode_fields,MAJ

def main():
    D.register();torch.manual_seed(40)
    a=torch.randn(48,80,dtype=torch.float16)/8;b=torch.randn(80,48,dtype=torch.float16)/8
    da,db=a.to('tnca'),b.to('tnca')
    fn=torch.compile(lambda x,y:x@y,fullgraph=True,dynamic=False)
    with launch_log() as warm_launches:warm=fn(da,db)
    reports=[]
    for launch in warm_launches:
        manifest,segments=artifact.load(launch['artifact_path'])
        execution=manifest['execution'];provenance=manifest['optional_metadata']['provenance']
        assert execution['dynamic_constraints'].get('program_partition')=='physical-x-chunk-v1'
        mapping=execution['program_to_role'];width=len(mapping);rpp=len(mapping['0'])
        words=[decode_fields(int.from_bytes(segments['text'][i:i+8],'little')) for i in range(0,len(segments['text']),8)]
        descriptors=sum(word['major']==MAJ['TXFER'] and word['func']==0 for word in words)
        # A generated transpose/copy kernel is a legitimate device operation.
        # Check all kernels' partition contract, then require descriptor/pair
        # lowering on the actual GEMM rather than on its auxiliary copies.
        if descriptors>=2:
            if args.pair:assert rpp==2 and provenance.get('rctx_roles')==2,(mapping,provenance.get('rctx_roles_refused'))
            else:assert width>1 and rpp==1,mapping
        else:assert width>1 and rpp==1,mapping
        reports.append({'kernel':manifest['identity']['kernel_name'],'width':width,'roles_per_program':rpp,'descriptor_loads':descriptors,'resources':manifest['resources']})
    assert any(row['descriptor_loads']>=2 for row in reports),reports
    torch.testing.assert_close(warm.cpu(),a@b);del warm
    session=D.MEMORY.session();before=len(session.dev.history)
    with launch_log() as launches:out=fn(da,db)
    torch.testing.assert_close(out.cpu(),a@b)
    assert launches and all(row['upload_bytes']==row['download_bytes']==0 for row in launches)
    history=session.dev.history[before:]
    assert sum(row['op']==11 for row in history)==len(launches)
    assert not any(row['op'] in (3,9) for row in history)
    metrics=session.dev.physical_grid_metrics()
    last=launches[-1];programs=last['grid'][0]*last['grid'][1]*last['grid'][2]
    assert sum(row['issued'] for row in metrics['cores'])==programs,metrics
    assert all(row['issued']==row['completed'] for row in metrics['cores']),metrics
    print('NATIVE MULTICORE FEATURES PASS '+json.dumps({'cores':args.cores,'pair':args.pair,'artifacts':reports,'metrics':metrics},sort_keys=True),flush=True)
if __name__=='__main__':main()
