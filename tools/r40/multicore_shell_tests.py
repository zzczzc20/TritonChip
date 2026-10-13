#!/usr/bin/env python3
"""Synthesizable persistent shell port tests; not native Torch/board acceptance."""
from pathlib import Path
import argparse
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, choices=(1,2,3,4,8), action='append')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='r40-shell-') as temp:
        image=str(Path(temp)/'code.hex')
        subprocess.run(['python3','tools/asm/tnca_asm.py','tb/core/progs/r40_grid_memory.s',
                        '-o',image,'--size','256'],cwd=ROOT,check=True,timeout=30)
        for n in args.cores or (1,3,8):
            output=str(Path(temp)/f'shell{n}.vvp')
            command=['iverilog','-g2012','-I','rtl/common','-I','rtl/noc',
                     '-s','tb_multicore_shell',f'-Ptb_multicore_shell.N={n}','-o',output]
            for library in ('common','core','gen','nuca','noc','tile'):
                command.extend(['-y','rtl/'+library])
            command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
            command.extend(['rtl/tnca_top.v','rtl/tnca_grid_dispatch.v','rtl/tnca_multicore_shell.v',
                            'tb/common/tnca_dram_model.v','tb/device/tb_multicore_shell.v'])
            subprocess.run(command,cwd=ROOT,check=True,timeout=180)
            subprocess.run(['vvp',output,'+hex='+image],cwd=ROOT,check=True,timeout=900)
    print('r40-multicore-shell-tests PASS',flush=True)
if __name__=='__main__':
    main()
