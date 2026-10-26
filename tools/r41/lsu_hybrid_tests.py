#!/usr/bin/env python3
"""Full hybrid LSU completions and memory effects versus the legacy pair."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-lsu-hybrid-') as tmp:
        for seed in (1,7,39):
            for stall in (0,30):
                for burst in (0,1):
                    binary=str(Path(tmp)/'hybrid.vvp')
                    subprocess.run(['iverilog','-g2012','-Irtl/common','-y','rtl/core','-s','tb_lsu_hybrid',
                        f'-Ptb_lsu_hybrid.SEED={seed}',f'-Ptb_lsu_hybrid.STALL_PCT={stall}',
                        f'-Ptb_lsu_hybrid.HOLD_PCT={5 if stall else 0}',f'-Ptb_lsu_hybrid.BURST={burst}',
                        '-Ptb_lsu_hybrid.MAXLAT=64','-Ptb_lsu_hybrid.ACCESSES=600',
                        '-o',binary,'tb/core/tb_lsu_hybrid.v'],cwd=ROOT,check=True,timeout=60)
                    subprocess.run(['vvp',binary],cwd=ROOT,check=True,timeout=120)
        reset_binary=str(Path(tmp)/'reset.vvp')
        subprocess.run(['iverilog','-g2012','-Irtl/common','-y','rtl/core','-s','tb_lsu_hybrid_reset',
            '-o',reset_binary,'tb/core/tb_lsu_hybrid_reset.v'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['vvp',reset_binary],cwd=ROOT,check=True,timeout=30)
    print('HYBRID LSU ALL 12 CONFIGURATIONS PASS',flush=True)


if __name__=='__main__':
    main()
