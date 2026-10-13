#!/usr/bin/env python3
"""Bounded dispatcher protocol and real core pipelines; no Torch/board gate credit."""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
def main():
    with tempfile.TemporaryDirectory(prefix='r40-dispatch-') as temp:
        image = str(Path(temp) / 'grid_memory.hex')
        subprocess.run(['python3', 'tools/asm/tnca_asm.py', 'tb/core/progs/r40_grid_memory.s',
                        '-o', image, '--size', '256'], cwd=ROOT, check=True, timeout=30)
        for bench, memory in (('tb_mem_arbiter_r40', 0), ('tb_grid_dispatch', 0), ('tb_grid_cores', 0), ('tb_grid_cores', 1)):
            for n in (1, 2, 3, 4, 8):
                output = str(Path(temp) / f'{bench}_{n}.vvp')
                command = ['iverilog', '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                           '-s', bench, f'-P{bench}.N={n}', '-o', output]
                if bench == 'tb_grid_cores':
                    for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
                        command.extend(['-y', 'rtl/' + library])
                    command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
                    command.extend(['rtl/tnca_top.v', 'tb/common/tnca_dram_model.v', f'-P{bench}.MEMORY={memory}'])
                if bench == 'tb_mem_arbiter_r40':
                    command.extend(['rtl/nuca/tnca_mem_arbiter.v', 'tb/tb_mem_arbiter_r40.v'])
                else:
                    command.extend(['rtl/tnca_grid_dispatch.v', 'rtl/tnca_grid_deadline.v',
                                    'tb/top/tb_grid_cores.v' if bench == 'tb_grid_cores' else 'tb/tb_grid_dispatch.v'])
                subprocess.run(command, cwd=ROOT, check=True, timeout=180)
                subprocess.run(['vvp', output, '+hex=' + image], cwd=ROOT, check=True, timeout=600 if memory else 180)
    print('r40-grid-dispatch-tests PASS (20 configurations)', flush=True)
if __name__ == '__main__':
    main()
