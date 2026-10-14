#!/usr/bin/env python3
"""Native resident Torch with actual independent RTL controller traffic."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(path))
parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('sim-multicore','fpga-multicore-sim'),default='sim-multicore');parser.add_argument('--cores',type=int,default=3,choices=(1,2,3,4,8));parser.add_argument('--ports',type=int,default=3);parser.add_argument('--pipeline',type=int,default=0);parser.add_argument('--board-top',action='store_true');parser.add_argument('--dma',action='store_true');args=parser.parse_args()
extra={}
if args.pipeline or args.board_top:
    if args.backend!='fpga-multicore-sim':parser.error('pipeline/board-top selects the MMIO AXI RTL stand-in')
    extra=dict(noc_link_stages=args.pipeline,dispatch_stages=args.pipeline,home_stages=args.pipeline,code_stages=args.pipeline,board_top=args.board_top)
if args.dma:
    if args.backend!='fpga-multicore-sim':parser.error('DMA ownership selects the MMIO AXI RTL stand-in')
    extra.update(maintenance=True,hbm_dma=True)
os.environ.update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND=args.backend,TNCA_BACKEND_ARGS=json.dumps({'core_count':args.cores,'memory_ports':args.ports,**extra}),TNCA_RCTX_SPMD='1')
os.environ.setdefault('TNCA_HOST_TIMEOUT','1800')
import torch
import triton
import tnca.torch_device as D
from tnca.triton_backend.driver import launch_log

def main():
    D.register();torch.manual_seed(40)
    x,y=torch.randn(4097),torch.randn(4097);dx,dy=x.to('tnca'),y.to('tnca')
    fn=torch.compile(lambda a,b:(a+b)*0.5,fullgraph=True,dynamic=False)
    warm=fn(dx,dy);torch.testing.assert_close(warm.cpu(),(x+y)*0.5);del warm
    session=D.MEMORY.session();before=session.dev.memory_port_metrics();start=len(session.dev.history)
    with launch_log() as launches:out=fn(dx,dy)
    after=session.dev.memory_port_metrics();activity=session.dev.physical_grid_metrics()
    torch.testing.assert_close(out.cpu(),(x+y)*0.5)
    assert launches and all(row['upload_bytes']==row['download_bytes']==0 for row in launches)
    assert sum(row['op']==11 for row in session.dev.history[start:])==len(launches)
    assert not any(row['op'] in (3,9) for row in session.dev.history[start:])
    delta=[{'port':b['port'],'read_lines':b['read_lines']-a['read_lines'],'write_lines':b['write_lines']-a['write_lines']} for a,b in zip(before,after)]
    assert len(delta)==args.ports and all(row['read_lines']>0 and row['write_lines']>0 for row in delta),delta
    assert activity['peak_active']==args.cores,activity
    assert sum(row['issued'] for row in activity['cores'])==257 and all(row['issued']==row['completed'] for row in activity['cores']),activity
    print('NATIVE MEMORY PORTS PASS '+json.dumps({'cores':args.cores,'ports':args.ports,'controller_traffic':delta,'physical_activity':activity,'zero_tensor_staging':True,'dma_ownership':args.dma},sort_keys=True),flush=True)
if __name__=='__main__':main()
