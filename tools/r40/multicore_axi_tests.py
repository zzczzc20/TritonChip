#!/usr/bin/env python3
"""Public persistent multicore shell through elastic links and AXI outlets."""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, choices=(1, 2, 3, 4, 8), action='append')
    parser.add_argument('--write-delay', type=int, default=15)
    parser.add_argument('--pipelined-issue', action='store_true')
    args = parser.parse_args()
    if args.write_delay < 0:
        parser.error('write delay must be nonnegative')
    cases = ((1, 1, 0), (2, 2, 1), (3, 2, 2), (3, 3, 2),
             (4, 4, 2), (8, 3, 2), (8, 8, 4))
    if args.cores:
        cases = tuple(case for case in cases if case[0] in args.cores)
    with tempfile.TemporaryDirectory(prefix='r40-multicore-axi-') as temp:
        image = str(Path(temp) / 'grid.hex')
        subprocess.run(['python3', 'tools/asm/tnca_asm.py',
                        'tb/core/progs/r40_grid_memory.s', '-o', image, '--size', '256'],
                       cwd=ROOT, check=True, timeout=30)
        for cores, ports, stages in cases:
            output = str(Path(temp) / f'axi-{cores}-{ports}-{stages}.vvp')
            command = ['iverilog', '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                       '-s', 'tb_multicore_axi', f'-Ptb_multicore_axi.N={cores}',
                       f'-Ptb_multicore_axi.PORTS={ports}',
                       f'-Ptb_multicore_axi.STAGES={stages}',
                       f'-Ptb_multicore_axi.WRITE_DELAY={args.write_delay}',
                       f'-Ptb_multicore_axi.AXI_PIPELINED_ISSUE={int(args.pipelined_issue)}', '-o', output]
            for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
                command += ['-y', 'rtl/' + library]
            command += ['-y', 'rtl', '-y', 'rtl/fpga']
            command += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'rtl/common').glob('*.v'))]
            command += ['tb/common/tnca_axi_line_model.v', 'tb/fpga/tb_multicore_axi.v']
            subprocess.run(command, cwd=ROOT, check=True, timeout=180)
            subprocess.run(['vvp', output, '+hex=' + image], cwd=ROOT, check=True, timeout=1200)
    print(f'MULTICORE AXI MATRIX PASS {len(cases)} configurations', flush=True)


if __name__ == '__main__':
    main()
