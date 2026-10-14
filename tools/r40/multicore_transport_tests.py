#!/usr/bin/env python3
"""Real mailbox -> synthesizable shell -> cores/NUCA; not Torch acceptance."""
from pathlib import Path
import argparse
import struct
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
import triton
from tnca.transport import Transport

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--programs',type=int,default=455)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='r40-grid-mailbox-') as temp:
        temp=Path(temp);image=temp/'code.hex';bench=temp/'device.vvp'
        subprocess.run(['python3','tools/asm/tnca_asm.py','tb/core/progs/r40_mailbox_grid.s',
                        '-o',str(image),'--size','256'],cwd=ROOT,check=True)
        command=['iverilog','-g2012','-I','rtl/common','-I','rtl/noc','-s','tnca_multicore_device','-o',str(bench)]
        for directory in ('common','core','gen','nuca','noc','tile'):
            command.extend(['-y','rtl/'+directory])
        command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
        command.extend(['rtl/tnca_top.v','rtl/tnca_grid_dispatch.v','rtl/tnca_grid_deadline.v',
                        'rtl/tnca_multicore_shell.v','tb/common/tnca_dram_model.v','tb/device/tnca_multicore_device.v'])
        subprocess.run(command,cwd=ROOT,check=True,timeout=180)
        dev=Transport(str(bench),workdir=str(temp/'session'),host_timeout=1800,deadline_cycles=2_000_000)
        try:
            dev.start()
            assert dev.physical_grid_capability()==(3,16777216)
            code=b''.join(int(line,16).to_bytes(8,'little') for line in image.read_text().splitlines())
            assert dev.upload(0,code).ok
            assert dev.physical_chunk_capability()==4
            # Capability alone must not authorize incompatible role resources.
            assert dev.launch_grid(0,0x40000,(5,1,1),group_width=4,role_mask=1).status==3
            assert dev.launch_grid(0,0x40000,(5,1,1),group_width=2,role_mask=3).status==3
            grid=(13,7,5) if args.programs==455 else (args.programs,1,1)
            before=len(dev.history)
            reply=dev.launch_grid(0,0x40000,grid,arg_stride=256)
            assert reply.ok,reply
            launches=[r for r in dev.history[before:] if r['op']==11]
            assert len(launches)==1 and not any(r['op'] in (3,9) for r in dev.history[before:])
            metrics=dev.physical_grid_metrics()
            assert metrics['peak_active']==min(3,args.programs),metrics
            assert sum(row['issued'] for row in metrics['cores'])==args.programs,metrics
            assert all(row['issued']==row['completed'] for row in metrics['cores']),metrics
            assert dev.request(12,3).status==3
            assert dev.request(13,3).status==3
            reply,data=dev.download(0x20000,args.programs*8)
            assert reply.ok,reply
            assert list(struct.unpack('<'+'Q'*args.programs,data))==list(range(4096,4096+args.programs))
            assert dev.request(15).status==3
            print(f'MULTICORE TRANSPORT PASS programs={args.programs} host_launches=1',flush=True)
        finally:
            dev.stop()
if __name__=='__main__':
    main()
