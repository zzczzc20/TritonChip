#!/usr/bin/env python3
"""Bounded dispatcher protocol and real core pipelines; no Torch/board gate credit."""
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
def main():
    with tempfile.TemporaryDirectory(prefix='r40-dispatch-') as temp:
        for bench in ('tb_grid_dispatch', 'tb_grid_cores'):
            for n in (1, 2, 3, 4, 8):
                output = str(Path(temp) / f'{bench}_{n}.vvp')
                command = ['iverilog', '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                           '-s', bench, f'-P{bench}.N={n}', '-o', output]
                if bench == 'tb_grid_cores':
                    for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
                        command.extend(['-y', 'rtl/' + library])
                    command.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v')))
                    command.append('rtl/tnca_top.v')
                command.extend(['rtl/tnca_grid_dispatch.v',
                                'tb/top/tb_grid_cores.v' if bench == 'tb_grid_cores' else 'tb/tb_grid_dispatch.v'])
                subprocess.run(command, cwd=ROOT, check=True, timeout=180)
                subprocess.run(['vvp', output], cwd=ROOT, check=True, timeout=180)
    print('r40-grid-dispatch-tests PASS (10 configurations)', flush=True)
if __name__ == '__main__':
    main()
