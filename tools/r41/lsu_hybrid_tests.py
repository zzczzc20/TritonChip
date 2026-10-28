#!/usr/bin/env python3
"""Full hybrid LSU completions and memory effects versus the legacy pair."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]


def main():
    configurations=0
    failures=[]
    with tempfile.TemporaryDirectory(prefix='tnca-lsu-hybrid-') as tmp:
        for beat_w in (4,5,6,7):
            for seed in (1,7,39):
                for stall in (0,30):
                    for burst in (0,1):
                        binary=str(Path(tmp)/'hybrid.vvp')
                        subprocess.run(['iverilog','-g2012','-Irtl/common','-y','rtl/core','-s','tb_lsu_hybrid',
                            f'-Ptb_lsu_hybrid.BEAT_W={beat_w}',f'-Ptb_lsu_hybrid.SEED={seed}',
                            f'-Ptb_lsu_hybrid.STALL_PCT={stall}',
                            f'-Ptb_lsu_hybrid.HOLD_PCT={5 if stall else 0}',f'-Ptb_lsu_hybrid.BURST={burst}',
                            f'-Ptb_lsu_hybrid.BURST_PERIOD={256 if beat_w>6 else 128 if beat_w>4 else 32}',
                            f'-Ptb_lsu_hybrid.BURST_DEPTH={240 if beat_w>6 else 120 if beat_w>4 else 24}',
                            # 64 fast slots need more than 64 consecutive burst accesses
                            f'-Ptb_lsu_hybrid.CHAIN_PERIOD={256 if beat_w>6 else 64}',
                            f'-Ptb_lsu_hybrid.BURST_BASE={0 if beat_w>4 else 2048}',
                            f'-Ptb_lsu_hybrid.BURST_LINES={64 if beat_w>4 else 16}',
                            f'-Ptb_lsu_hybrid.MAXLAT={64 if beat_w<=5 else 1024}',
                            '-Ptb_lsu_hybrid.ACCESSES=600',
                            '-o',binary,'tb/core/tb_lsu_hybrid.v'],cwd=ROOT,check=True,timeout=60)
                        try:
                            subprocess.run(['vvp',binary],cwd=ROOT,check=True,timeout=600)
                            configurations+=1
                        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                            failures.append((beat_w,seed,stall,burst))
                            print(f'HYBRID LSU FAIL beat_w={beat_w} seed={seed} stall={stall} burst={burst}',flush=True)
        reset_binary=str(Path(tmp)/'reset.vvp')
        subprocess.run(['iverilog','-g2012','-Irtl/common','-y','rtl/core','-s','tb_lsu_hybrid_reset',
            '-o',reset_binary,'tb/core/tb_lsu_hybrid_reset.v'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['vvp',reset_binary],cwd=ROOT,check=True,timeout=30)
    if failures:
        raise RuntimeError(f'HYBRID LSU {len(failures)} CONFIGURATIONS FAILED: {failures}')
    print(f'HYBRID LSU ALL {configurations} CONFIGURATIONS PASS',flush=True)


if __name__=='__main__':
    main()
