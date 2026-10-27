#!/usr/bin/env python3
"""Check cold service at image-like home settings and larger finite windows."""
import argparse
import json
from pathlib import Path
import re
import tempfile
from tools.r41.wide_home_tests import run_case

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=[]
with tempfile.TemporaryDirectory(prefix='r41-home-window-') as temp:
    for mshrs,targets in ((4,8),(8,8),(8,16),(16,32)):
        params=dict(WORD_BITS=512,MSHR_N=mshrs,TARGET_N=targets,HIT_CHAIN=1,SERVICE_MODE=2,
            SETS=32,SET_W=5,WAYS=8,WAY_W=3,REGISTER_VICTIM_OWNER=1,DCO_REGISTERED_RANGES=1,
            FULL_LINE_TAG=1,REGISTER_VICTIM_SELECT=1,REGISTER_VICTIM_RANK=1)
        name=f'm{mshrs}-t{targets}'
        log=run_case(str(Path(temp)/'home.vvp'),'tb_nuca_wide',params,'tb/nuca/tb_nuca_wide.v',
            ['+nreq=4096','+min_wb=0'])
        (a.out/(name+'.log')).write_text(log)
        m=re.search(r'HOME SERVICE mode=2 word_bytes=64 accepted=(\d+) responses=(\d+) span_cycles=(\d+) hit=(\d+) miss=(\d+)',log)
        assert m,name
        accepted,responses,cycles,hit,miss=map(int,m.groups())
        assert accepted==responses==4096,name
        row=dict(configuration=params,accepted=accepted,responses=responses,cycles=cycles,
            hit=hit,miss=miss,gb_per_s_at_100mhz=accepted*64/cycles*0.1,
            actual_image_window=(mshrs,targets)==(8,16))
        rows.append(row);print(name,json.dumps(row,sort_keys=True),flush=True)
(a.out/'results.json').write_text(json.dumps(dict(evidence='standalone-RTL-synthetic-memory-window-sweep',
    board_measured=False,larger_window_integrated=False,
    scope='8/16 is the current image home capacity; 16/32 needs a wider memory tag and new integration',
    rows=rows),indent=2)+'\n')
