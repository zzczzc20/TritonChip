#!/usr/bin/env python3
"""Execute existing ISA programs on the real hybrid core and a wide memory fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/asm'))
from tnca_asm import Assembler,build_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    binary=a.out/'core.vvp'
    command=['iverilog','-g2012','-Irtl/common','-y','rtl/common','-y','rtl/core','-y','rtl/gen','-y','tb/common',
        '-s','tb_core_hybrid','-Ptb_core_hybrid.MEM_LATENCY=6','-Ptb_core_hybrid.RANDOM_STALL=1',
        '-Ptb_core_hybrid.MEM_REORDER=1','-o',str(binary),'tb/core/tb_core_hybrid.v']
    with (a.out/'build.log').open('w') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    rows=[]
    for name in ('t02_mem','t22_atomic','t12_tensor','t37_lsu_straddle'):
        source=ROOT/f'tb/core/progs/{name}.s'
        image=a.out/f'{name}.hex'
        image.write_text(''.join(f'{word:016x}\n' for word in build_image(Assembler().assemble(source.read_text()),size_words=65536)))
        with (a.out/f'{name}.log').open('w') as f:
            subprocess.run(['vvp',str(binary),f'+hex={image}','+maxcycles=1000000'],cwd=ROOT,
                           stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        log=(a.out/f'{name}.log').read_text()
        assert re.search(r'^PASS$',log,re.M) and not re.search(r'FAIL|FATAL',log),a.out/f'{name}.log'
        stats=re.search(r'HYBRID CORE fast=(\d+) slow=(\d+) retired=(\d+) switches=(\d+) wb_stall=(\d+) peak=(\d+)',log)
        assert stats,log
        counters=dict(zip(('fast','slow','retired','switches','wb_stall','peak'),map(int,stats.groups())))
        assert counters['fast']+counters['slow']==counters['retired'],counters
        if name in ('t02_mem','t37_lsu_straddle'):assert counters['fast']>0,counters
        rows.append({'program':name,'status':'pass','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'counters':counters})
        print(name,'PASS',counters,flush=True)
    sources=list((ROOT/'rtl/core').glob('*.v'))+list((ROOT/'rtl/common').glob('*v*'))+list((ROOT/'rtl/gen').glob('*.v'))
    sources+=[ROOT/'tb/core/tb_core_hybrid.v',ROOT/'tb/common/tnca_line_mem_model.v']
    (a.out/'results.json').write_text(json.dumps({'evidence':'RTL-functional-with-synthetic-wide-memory',
        'board_speedup_measured':False,'image_built':False,'build_command':command,
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'sources_sha256':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources if s.is_file()},
        'rows':rows},indent=2)+'\n')
    print('HYBRID CORE NATIVE ISA ALL 4 PROGRAMS PASS',flush=True)


if __name__=='__main__':main()
