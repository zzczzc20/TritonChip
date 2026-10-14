#!/usr/bin/env python3
"""Public Session/AOT -> one grid request -> real physical cores/NUCA."""
from pathlib import Path
import argparse
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(path))
import triton
import triton.language as tl
from tnca import aot
from tnca.runtime import Session
from tnca.errors import AdmissionError, DeviceLost
from tnca.transport import Reply

@triton.jit
def role_loop(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(x_ptr+index,index<n,other=0)+1.0
    tl.store(o_ptr+index,value,index<n)

def kernel(b,target):
    base=b.special('arg_base');out=b.sload(base,0);counter=b.sload(base,8)
    x,y,z=[b.program_id(i) for i in range(3)]
    linear=b.sbin('add',x,b.sbin('mul',b.num_programs(0),b.sbin('add',y,b.sbin('mul',b.num_programs(1),z))))
    b.sstore(b.sbin('add',out,b.sbini('mul',linear,8)),b.sbini('add',linear,4096),size='d')
    b.atomic('add',counter,b.const(1),size='d');b.fence();b.exit()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--cores',type=int,choices=(1,2,3,4,8),default=3)
    options=parser.parse_args()
    for status in (0,1):
        stub=Session.__new__(Session);stub.active_cores=1
        stub.dev=SimpleNamespace(launch_grid=lambda *args,**kw:Reply(1,status,0,0),
                                 quiesce=lambda:Reply(2,1,0,0))
        event=SimpleNamespace(_to=lambda state:None)
        task={'event':event,'module':SimpleNamespace(code_addr=0),'arg_addr':0x20000,
              'grid':(1,1,1),'arg_stride':256}
        try:stub._run_physical_grid(task)
        except DeviceLost:pass
        else:raise AssertionError('failed quiesce reported as drained')
    with tempfile.TemporaryDirectory(prefix='r40-physical-session-') as temp:
        temp=Path(temp);bench=temp/'device.vvp'
        command=['iverilog','-g2012','-I','rtl/common','-I','rtl/noc','-s','tnca_multicore_device','-o',str(bench)]
        for directory in ('common','core','gen','nuca','noc','tile'):command.extend(['-y','rtl/'+directory])
        command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
        command.extend(['rtl/tnca_top.v','rtl/tnca_grid_dispatch.v','rtl/tnca_grid_deadline.v',
                        'rtl/tnca_multicore_shell.v','tb/common/tnca_dram_model.v','tb/device/tnca_multicore_device.v'])
        if not options.automatic:
            subprocess.run(command,cwd=ROOT,check=True,timeout=180)
        bundle=aot.compile_aot(kernel,kernel_name='physical_grid_runtime',signature={'out':'*i64','counter':'*i64'},
                               source_kind='tnca-ir',profile='core-vrf-sync-v5',options={'rctx_spmd':0},output=str(temp/'kernel.tnca'))
        aot.compile_aot(role_loop,kernel_name='role_loop',signature={'x_ptr':'*fp32','o_ptr':'*fp32','n':'i32','BLOCK':'constexpr'},
                        constexprs={'BLOCK':16},profile='core-vrf-sync-v5',options={'rctx_spmd':1},output=str(temp/'role_loop.tnca'))
        aot.compile_aot(role_loop,kernel_name='role_loop',signature={'x_ptr':'*fp32','o_ptr':'*fp32','n':'i32','BLOCK':'constexpr'},
                        constexprs={'BLOCK':16},profile='core-vrf-sync-v5',options={'rctx_spmd':0},output=str(temp/'vector.tnca'))
        config=({'backend':'sim-multicore','core_count':options.cores} if options.automatic
                else {'bench':str(bench),'physical_grid':True})
        cores=options.cores if options.automatic else 3
        with Session(profile='core-vrf-sync-v5',host_timeout=1800,**config) as session:
            out=session.allocate(455*8);counter=session.allocate(8)
            session.upload(out,bytes(out.size));session.upload(counter,bytes(8))
            module=session.load(str(temp/'kernel.tnca'))
            unsupported=session.load(str(temp/'role_loop.tnca'))
            assert unsupported.manifest['execution']['program_to_role']!={'0':['role0']}
            before=len(session.dev.history)
            try: session.submit(unsupported,{'x_ptr':out,'o_ptr':out,'n':16},grid=(13,7,5))
            except AdmissionError: pass
            else: raise AssertionError('unpartitioned role loop accepted')
            assert len(session.dev.history)==before and not session.unfinished
            before=len(session.dev.history)
            try:session.submit(module,{'out':out,'counter':counter},grid=(1.5,1,1))
            except AdmissionError:pass
            else:raise AssertionError('fractional physical axis accepted')
            assert len(session.dev.history)==before and not session.unfinished
            event=session.submit(module,{'out':out,'counter':counter},grid=(13,7,5))
            session.wait(event)
            table=event._task['block'];stride=event._task['arg_stride']
            assert stride%256==0 and len(table)==cores*stride
            for core in range(cores):
                assert struct.unpack_from('<QQ',table,core*stride)==(out.addr,counter.addr)
            assert event.device_cycles>0 and event.program_cycles is None
            assert sum(record['op']==11 for record in session.dev.history[before:])==1
            assert list(struct.unpack('<455Q',session.download(out)))==list(range(4096,4096+455))
            assert struct.unpack('<Q',session.download(counter))[0]==455
            x=session.allocate(257*4);y=session.allocate(257*4)
            values=[float(i%7) for i in range(257)]
            session.upload(x,struct.pack('<257f',*values))
            vector=session.load(str(temp/'vector.tnca'))
            before=len(session.dev.history)
            vector_event=session.submit(vector,{'x_ptr':x,'o_ptr':y,'n':257},grid=(17,1,1))
            session.wait(vector_event)
            assert list(struct.unpack('<257f',session.download(y)))==[v+1 for v in values]
            assert sum(record['op']==11 for record in session.dev.history[before:])==1
            print('PHYSICAL GRID TRITON VECTOR PASS elements=257 programs=17 host_launches=1',flush=True)
            print('PHYSICAL GRID RUNTIME PASS programs=455 exact_atomic_count=455 host_launches=1',flush=True)
if __name__=='__main__':main()
