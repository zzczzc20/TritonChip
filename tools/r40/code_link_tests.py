#!/usr/bin/env python3
"""Code upload/readback acknowledgement and private-RAM geometry matrix."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]

def main():
    with tempfile.TemporaryDirectory(prefix='r40-code-links-') as temp:
        for cores,stages in ((1,0),(3,0),(3,1),(3,2),(3,4),(8,2)):
            out=str(Path(temp)/f'{cores}-{stages}.vvp')
            cmd=['iverilog','-g2012','-Irtl/common','-Irtl/noc','-s','tb_multicore_code_links',
                 f'-Ptb_multicore_code_links.N={cores}',f'-Ptb_multicore_code_links.STAGES={stages}','-o',out]
            for lib in ('rtl','rtl/common','rtl/core','rtl/gen','rtl/nuca','rtl/noc','rtl/tile'):
                cmd+=['-y',lib]
            subprocess.run(cmd+['tb/fpga/tb_multicore_code_links.v'],cwd=ROOT,check=True,timeout=180)
            subprocess.run(['vvp',out],cwd=ROOT,check=True,timeout=180)
    print('CODE LINK MATRIX PASS 6 configurations',flush=True)
if __name__=='__main__':main()
