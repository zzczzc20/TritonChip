#!/usr/bin/env python3
"""One resident Torch graph, one RTL device, four runtime memory configurations.

Functional mechanism evidence only; simulator wall time and this small graph
are not R40 board performance or frozen-inventory acceptance evidence.
"""
import argparse,os,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(p))
os.environ.update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND='fpga-multicore-sim',
    TNCA_BACKEND_ARGS=json.dumps(dict(core_count=3,memory_ports=3,memory_controls=True,
                                    maintenance=True,hbm_dma=True,board_top=True)),TNCA_RCTX_SPMD='1')
os.environ.setdefault('TNCA_HOST_TIMEOUT','1800')
import torch
import triton
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal',action='store_true',help='record timed MMIO/DMA stimuli for functional replay')
    args=parser.parse_args()
    if args.journal:
        configuration=json.loads(os.environ['TNCA_BACKEND_ARGS'])
        configuration['journal']=True
        os.environ['TNCA_BACKEND_ARGS']=json.dumps(configuration)
    D.register();torch.manual_seed(40)
    x,y=torch.randn(4097),torch.randn(4097)
    a,b=x.to('tnca'),y.to('tnca')
    fn=torch.compile(lambda a,b:(a+b)*0.5,fullgraph=True,dynamic=False)
    out=fn(a,b);torch.testing.assert_close(out.cpu(),(x+y)*0.5);del out
    dev=D.MEMORY.session().dev
    if args.journal:
        print('MMIO_JOURNAL '+json.dumps(dict(directory=dev.link.journal_dir)),flush=True)
    for phase,(ports,limit) in enumerate([(3,8),(1,8),(1,1),(3,1)]):
        # Fresh values distinguish current output from stale data in another
        # outlet view or an allocator-reused output buffer.
        x,y=torch.randn(4097)+phase,torch.randn(4097)-phase*0.25
        a,b=x.to('tnca'),y.to('tnca')
        config=dev.configure_memory(active_ports=ports,outstanding=limit)
        before=dev.memory_port_metrics()
        with launch_log() as launches:out=fn(a,b)
        after=dev.memory_port_metrics()
        torch.testing.assert_close(out.cpu(),(x+y)*0.5);del out
        assert launches and all(row['upload_bytes']==row['download_bytes']==0 for row in launches)
        delta=[dict(port=i,read_lines=e['read_lines']-s['read_lines'],write_lines=e['write_lines']-s['write_lines']) for i,(s,e) in enumerate(zip(before,after))]
        assert all(row['read_lines']>0 and row['write_lines']>0 for row in delta[:ports]),delta
        assert all(row['read_lines']==row['write_lines']==0 for row in delta[ports:]),delta
        assert config==dev.memory_configuration()
        assert dev.physical_grid_metrics()['peak_active']==3
        print('NATIVE MEMORY CONTROL PASS '+json.dumps(dict(configuration=config,outlet_delta=delta,numerical=True,zero_tensor_staging=True,scope='functional RTL only'),sort_keys=True),flush=True)
    dev.configure_memory(active_ports=3,outstanding=8)
if __name__=='__main__':main()
