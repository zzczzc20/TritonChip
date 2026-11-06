#!/usr/bin/env python3
"""Wide-home byte shadow, original narrow regression, atomics and MMIO."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]


def run_case(binary,top,parameters,source,args=()):
    command=['iverilog','-g2012','-Irtl/common','-y','rtl/common','-y','rtl/nuca','-y','tb/common','-s',top,'-o',binary]
    command += [f'-P{top}.{k}={v}' for k,v in parameters.items()]
    subprocess.run(command+[source],cwd=ROOT,check=True,timeout=60)
    result=subprocess.run(['vvp',binary,*args],cwd=ROOT,text=True,capture_output=True,check=True,timeout=120)
    print(result.stdout,end='',flush=True)
    assert 'PASS' in result.stdout and 'FAIL' not in result.stdout,result.stdout
    return result.stdout


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-wide-home-') as temp:
        binary=str(Path(temp)/'home.vvp')
        run_case(binary,'tb_nuca_slice',{},'tb/nuca/tb_nuca_slice.v',['+nreq=2000'])
        for pipeline in (0,1):
            for seed in (1,39):
                run_case(binary,'tb_nuca_wide',{'CACHE_DATA_PIPELINE':pipeline},
                         'tb/nuca/tb_nuca_wide.v',['+nreq=2000',f'+seed={seed}'])
            pressure=run_case(binary,'tb_nuca_wide',{'CACHE_DATA_PIPELINE':pipeline,'PRESSURE':1,'DRAM_LAT':200,
                'REGISTER_VICTIM_OWNER':1,'DCO_REGISTERED_RANGES':1},'tb/nuca/tb_nuca_wide.v',['+nreq=100','+min_wb=0'])
            import re
            assert int(re.search(r'target_full=(\d+)',pressure).group(1))>0,pressure
            run_case(binary,'tb_nuca_wide_special',{'PIPE':pipeline},'tb/nuca/tb_nuca_wide_special.v')
    print('WIDE HOME ALL 9 CONFIGURATIONS PASS',flush=True)


if __name__=='__main__':main()
