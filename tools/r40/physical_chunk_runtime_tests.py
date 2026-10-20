#!/usr/bin/env python3
"""Real Triton chunk artifact, 3-D tail and exactly-once atomic oracle."""
import argparse,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(path))
import triton
import triton.language as tl
from tnca import aot
from tnca.runtime import Session

@triton.jit
def global_chunk(out_ptr,counter):
    x,y,z=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    linear=x+tl.num_programs(0)*(y+tl.num_programs(1)*z)
    tl.store(out_ptr+linear,linear+4096)
    tl.atomic_add(counter,1,sem='relaxed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cores',type=int,default=3,choices=(1,2,3,4,8));args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='r40-chunk-runtime-') as temp:
        path=str(Path(temp)/'chunk.tnca')
        aot.compile_aot(global_chunk,kernel_name='global_chunk',signature={'out_ptr':'*i32','counter':'*i32'},profile='core-vrf-sync-v5',options={'rctx_spmd':1,'physical_chunks':1},output=path)
        with Session(profile='core-vrf-sync-v5',backend='sim-multicore',core_count=args.cores,host_timeout=1800) as session:
            module=session.load(path)
            mapping=module.manifest['execution']['program_to_role'];width=len(mapping)
            assert width>1,mapping
            assert module.manifest['execution']['dynamic_constraints']['program_partition']=='physical-x-chunk-v1'
            assert module.manifest['execution']['grid_dims_supported']==[True,True,True]
            out=session.allocate(456*4);counter=session.allocate(8)
            session.upload(out,bytes(out.size));session.upload(counter,bytes(8))
            before=len(session.dev.history)
            event=session.submit(module,{'out_ptr':out,'counter':counter},grid=(13,7,5));session.wait(event)
            assert sum(row['op']==11 for row in session.dev.history[before:])==1
            assert event.program_cycles is None and event.device_cycles>0
            assert list(struct.unpack('<456i',session.download(out)))==list(range(4096,455+4096))+[0]
            assert struct.unpack('<2i',session.download(counter))==(455,0)
            metrics=session.dev.physical_grid_metrics()
            assert sum(row['issued'] for row in metrics['cores'])==455,metrics
            assert all(row['issued']==row['completed'] for row in metrics['cores']),metrics
            table=event._task['block'];stride=event._task['arg_stride']
            assert len(table)==args.cores*stride
            for core in range(args.cores):assert struct.unpack_from('<QQ',table,core*stride)==(out.addr,counter.addr)
            print(f'PHYSICAL CHUNK RUNTIME PASS cores={args.cores} width={width} grid=13x7x5 count=455 host_launches=1',flush=True)
if __name__=='__main__':main()
