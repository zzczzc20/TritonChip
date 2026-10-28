#!/usr/bin/env python3
"""Finite shared vector-line planner: byte oracle, backpressure and throughput."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
SLOT_COUNTS=(2,4,8,16,32,64)


def reset_tests(temp):
    binary=str(Path(temp)/'reset.vvp')
    for slots in SLOT_COUNTS:
        subprocess.run(['iverilog','-g2012','-Irtl/common','-s','tb_lsu_line_reset',
            f'-Ptb_lsu_line_reset.SLOTS={slots}',f'-Ptb_lsu_line_reset.TAG_W={slots.bit_length()-1}',
            '-o',binary,'rtl/core/tnca_lsu_line.v','tb/core/tb_lsu_line_reset.v'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['vvp',binary],cwd=ROOT,check=True,timeout=30)
    subprocess.run(['iverilog','-g2012','-Irtl/common','-s','tb_lsu_line_reset',
        '-Ptb_lsu_line_reset.BAD_RESPONSE=1','-o',binary,'rtl/core/tnca_lsu_line.v',
        'tb/core/tb_lsu_line_reset.v'],cwd=ROOT,check=True,timeout=60)
    bad=subprocess.run(['vvp',binary],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert bad.returncode and 'invalid/duplicate response' in bad.stdout,bad
    print('LINE LSU UNOWNED RESPONSE REFUSED',flush=True)


def main():
    configurations=0
    with tempfile.TemporaryDirectory(prefix='tnca-lsu-line-') as temp:
        for slots in SLOT_COUNTS:
            for stall in (0,1):
                for stream in (0,1):
                    for seed in (1,39):
                        binary=str(Path(temp)/'line.vvp')
                        subprocess.run(['iverilog','-g2012','-Irtl/common','-s','tb_lsu_line',
                            f'-Ptb_lsu_line.SLOTS={slots}',f'-Ptb_lsu_line.TAG_W={slots.bit_length()-1}',
                            f'-Ptb_lsu_line.STALL={stall}',f'-Ptb_lsu_line.STREAM={stream}',
                            f'-Ptb_lsu_line.SEED={seed}','-o',binary,
                            'rtl/core/tnca_lsu_line.v','tb/core/tb_lsu_line.v'],cwd=ROOT,check=True,timeout=60)
                        subprocess.run(['vvp',binary],cwd=ROOT,check=True,timeout=60)
                        configurations+=1
        reset_tests(temp)
    print(f'LINE LSU ALL {configurations} CONFIGURATIONS PASS',flush=True)


if __name__=='__main__':
    main()
