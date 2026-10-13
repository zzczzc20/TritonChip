#!/usr/bin/env python3
"""Coherent host ports on real three-core NUCA; diagnostic, not Torch acceptance."""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
def run(command, timeout=180):
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)
def main():
    with tempfile.TemporaryDirectory(prefix='r40-host-') as temp:
        mux = str(Path(temp)/'mux.vvp')
        run(['iverilog', '-g2012', '-s', 'tb_idle_host_mux', '-o', mux,
             'rtl/noc/tnca_idle_host_mux.v', 'tb/tb_idle_host_mux.v'])
        run(['vvp', mux])
        image = str(Path(temp)/'memory.hex')
        run(['python3', 'tools/asm/tnca_asm.py', 'tb/core/progs/r40_grid_memory.s',
             '-o', image, '--size', '256'])
        output = str(Path(temp)/'hostgrid.vvp')
        command = ['iverilog', '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                   '-s', 'tb_grid_cores', '-Ptb_grid_cores.N=3',
                   '-Ptb_grid_cores.MEMORY=1', '-Ptb_grid_cores.HOST=1', '-o', output]
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command.extend(['-y', 'rtl/'+library])
        command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
        command.extend(['rtl/tnca_top.v', 'rtl/tnca_grid_dispatch.v',
                        'tb/common/tnca_dram_model.v', 'tb/top/tb_grid_cores.v'])
        run(command)
        run(['vvp', output, '+hex='+image], timeout=600)
    print('r40-coherent-host-tests PASS', flush=True)
if __name__ == '__main__':
    main()
