#!/usr/bin/env python3
"""Independent outlet protocol matrix; real-system/native gates are separate."""
from pathlib import Path
import subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2]
def main():
    cases=0
    with tempfile.TemporaryDirectory(prefix='r40-memory-ports-') as temp:
        for n in (1,2,3,4,8):
            for ports in sorted({1,n,min(2,n),min(3,n),min(4,n)}):
                output=str(Path(temp)/f'ports-{n}-{ports}.vvp')
                subprocess.run(['iverilog','-g2012','-s','tb_mem_ports_r40',f'-Ptb_mem_ports_r40.N={n}',f'-Ptb_mem_ports_r40.P={ports}','-o',output,'rtl/nuca/tnca_mem_arbiter.v','rtl/nuca/tnca_mem_ports.v','tb/tb_mem_ports_r40.v'],cwd=ROOT,check=True,timeout=60)
                subprocess.run(['vvp',output],cwd=ROOT,check=True,timeout=60);cases+=1
        image=str(Path(temp)/'memory.hex')
        subprocess.run(['python3','tools/asm/tnca_asm.py','tb/core/progs/r40_grid_memory.s','-o',image,'--size','256'],cwd=ROOT,check=True,timeout=30)
        for n,ports in ((1,1),(2,2),(3,2),(3,3),(4,4),(8,3),(8,8)):
            output=str(Path(temp)/f'cores-{n}-{ports}.vvp')
            command=['iverilog','-g2012','-I','rtl/common','-I','rtl/noc','-s','tb_grid_cores',f'-Ptb_grid_cores.N={n}',f'-Ptb_grid_cores.PORTS={ports}','-Ptb_grid_cores.MEMORY=1','-Ptb_grid_cores.HOST=1',f'-Ptb_grid_cores.INJECT_BAD_SOURCE={int(n==1)}','-o',output]
            for library in ('common','core','gen','nuca','noc','tile'):command+=['-y','rtl/'+library]
            command += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v'))]
            command += ['rtl/tnca_top.v','rtl/tnca_grid_dispatch.v','rtl/tnca_grid_deadline.v','tb/common/tnca_dram_model.v','tb/top/tb_grid_cores.v']
            subprocess.run(command,cwd=ROOT,check=True,timeout=180)
            subprocess.run(['vvp',output,'+hex='+image],cwd=ROOT,check=True,timeout=1200)
    print(f'MEMORY PORTS MATRIX PASS {cases} protocol and 7 real-core configurations',flush=True)
if __name__=='__main__':main()
