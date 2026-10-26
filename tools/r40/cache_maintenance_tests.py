#!/usr/bin/env python3
"""Range maintenance and all-home drain; this never enables runtime DMA."""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(command):
    result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
    if result.returncode or 'FATAL:' in result.stdout or '\nFAIL' in result.stdout:
        raise RuntimeError(result.stdout+result.stderr)
    if result.stderr:
        raise RuntimeError('unexpected compile/run diagnostic: '+result.stderr)
    return result.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--global-grid',action='store_true',help='also execute real three-core ownership handoffs')
    args=parser.parse_args()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='r40-maintenance-') as directory:
        image = str(Path(directory)/'case.vvp')
        for tag in (0,1):
            for latency in (1,8,31):
                run(['iverilog','-g2012','-I','rtl/common','-s','tb_nuca_maintenance',
                     f'-Ptb_nuca_maintenance.FULL_TAG={tag}',
                     f'-Ptb_nuca_maintenance.LATENCY={latency}','-o',image,
                     'rtl/nuca/tnca_nuca_slice.v','tb/common/tnca_dram_model.v',
                     'tb/nuca/tb_nuca_maintenance.v'])
                output=run(['vvp',image])
                assert 'NUCA MAINTENANCE PASS' in output,output
                print(output.strip(),flush=True);cases+=1
        for homes in (1,3,8):
            run(['iverilog','-g2012','-s','tb_cache_handoff',f'-Ptb_cache_handoff.N={homes}',
                 '-o',image,'rtl/nuca/tnca_cache_handoff.v','tb/nuca/tb_cache_handoff.v'])
            output=run(['vvp',image])
            assert 'CACHE HANDOFF PASS' in output,output
            print(output.strip(),flush=True);cases+=1
        # Exercise the normal path with maintenance both compiled out and
        # enabled but idle, under pressure and DCO. No subset of reads is
        # sampled: the existing differential bench checks every response.
        for enabled in (0,1):
            for policy in (0,74):
                run(['iverilog','-g2012','-I','rtl/common','-s','tb_nuca_slice',
                     f'-Ptb_nuca_slice.MAINTENANCE={enabled}',f'-Ptb_nuca_slice.POLICY={policy}',
                     '-Ptb_nuca_slice.DRAM_LAT=40','-Ptb_nuca_slice.DRAM_STALL=1',
                     '-Ptb_nuca_slice.DRAM_OOO=1','-o',image,'rtl/nuca/tnca_nuca_slice.v',
                     'tb/common/tnca_dram_model.v','tb/nuca/tb_nuca_slice.v'])
                for seed in (1,7):
                    output=run(['vvp',image,'+nreq=20000',f'+seed={seed}','+min_wb=1'])
                    assert 'PASS' in output.splitlines(),output
                    print(f'PRESSURE PASS maintenance={enabled} policy={policy} seed={seed} requests=20000',flush=True)
                    cases+=1
        if args.global_grid:
            program=str(Path(directory)/'grid.hex')
            run(['python3','tools/asm/tnca_asm.py','tb/core/progs/r40_grid_memory.s',
                 '-o',program,'--size','256'])
            for stages in (0,2):
                command=['iverilog','-g2012','-I','rtl/common','-I','rtl/noc','-s','tb_grid_cores',
                         '-Ptb_grid_cores.N=3','-Ptb_grid_cores.MEMORY=1','-Ptb_grid_cores.HOST=1',
                         '-Ptb_grid_cores.MAINTENANCE=1',f'-Ptb_grid_cores.PIPELINE={stages}',
                         '-o',image]
                for library in ('common','core','gen','nuca','noc','tile'):
                    command+=['-y','rtl/'+library]
                command += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'rtl/common').glob('*.v'))]
                command += ['rtl/tnca_top.v','rtl/tnca_grid_dispatch.v','rtl/tnca_grid_deadline.v',
                            'tb/common/tnca_dram_model.v','tb/top/tb_grid_cores.v']
                run(command)
                output=run(['vvp',image,'+hex='+program])
                assert 'GLOBAL CACHE HANDOFF PASS N=3' in output,output
                print(output.strip(),flush=True);cases+=1
    print(f'r40-cache-maintenance PASS ({cases}/{cases})',flush=True)


if __name__=='__main__':
    main()
