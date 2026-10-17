#!/usr/bin/env python3
"""Real MMIO/AXI ownership handoff and cache-bypassing BO model access."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from tnca.backends.multicore_mmio_sim import stand_in
from tnca.backends.multicore_mmio import OP_CACHE_HANDOFF


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores',type=int,default=3)
    parser.add_argument('--ports',type=int,default=3)
    parser.add_argument('--pipeline',type=int,default=2)
    parser.add_argument('--board-top',action='store_true')
    args=parser.parse_args()
    dev=stand_in(core_count=args.cores,memory_ports=args.ports,maintenance=True,hbm_dma=True,
                 noc_link_stages=args.pipeline,dispatch_stages=args.pipeline,
                 home_stages=args.pipeline,code_stages=args.pipeline,
                 board_top=args.board_top,host_timeout=1800)
    try:
        dev.start()
        assert dev.capabilities()['hbm_dma'] and dev.capabilities()['cache_ownership_handoff']
        base=0x30000
        expected=bytearray((i*17+3)&255 for i in range(256))
        assert dev.upload(base,expected).ok
        # Fill clean home copies, then leave a dirty word in every line.
        for offset in range(0,256,8):
            want=int.from_bytes(expected[offset:offset+8],'little')
            assert dev._window(base+offset,None,write=False)==want
        for offset in range(8,256,64):
            value=0x1122334455667788+offset
            dev._window(base+offset,value,write=True)
            expected[offset:offset+8]=value.to_bytes(8,'little')
        start=len(dev.history)
        reply,got=dev.download(base,256)
        assert reply.ok and got==expected,'dirty cache bytes never reached BO'
        assert any(row['op']==OP_CACHE_HANDOFF for row in dev.history[start:])
        # Clean cached values must be invalidated before replacing BO bytes.
        for offset in range(0,256,8):dev._window(base+offset,None,write=False)
        replacement=bytes((255-i)&255 for i in range(256))
        assert dev.upload(base,replacement).ok
        for offset in range(0,256,8):
            assert dev._window(base+offset,None,write=False)==int.from_bytes(replacement[offset:offset+8],'little')
        expected=bytearray(replacement)
        # A dirty neighbour outside a one-byte range must survive handoff.
        dev._window(base+16,0xfedcba9876543210,write=True)
        expected[16:24]=(0xfedcba9876543210).to_bytes(8,'little')
        assert dev.request(OP_CACHE_HANDOFF,base+9,1).ok
        dev.link.hbm_write(base+9-dev.local_limit,b'\xaa')
        expected[9]=0xaa
        assert dev.download(base,256)[1]==expected,'partial DMA lost neighbouring dirty bytes'
        # Range and zero-work rules; rejected operations cannot transfer data.
        assert dev.request(OP_CACHE_HANDOFF,base,0).ok
        for addr,size in ((0,1),(dev.memory_bytes,1),(base,0xffffffffffffffff)):
            assert dev.request(OP_CACHE_HANDOFF,addr,size).status==3
        assert dev.download(base,256)[1]==expected
        print(f'MULTICORE DMA OWNERSHIP PASS cores={args.cores} ports={args.ports} pipeline={args.pipeline} board={args.board_top}',flush=True)
        print('dirty download, clean invalidation, partial-line preservation, zero/range guards PASS',flush=True)
    finally:
        dev.stop()


if __name__=='__main__':main()
