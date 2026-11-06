#!/usr/bin/env python3
"""Measure accepted/completed home-word service with checked read responses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from tools.r41.wide_home_tests import ROOT,run_case

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=[]
with tempfile.TemporaryDirectory(prefix='r41-home-service-') as temp:
    for width,chain in ((64,0),(512,0),(512,1)):
        for mode in (1,2):
            # Match returned useful bytes, not just request counts. The
            # streaming cases then traverse the same number of cache lines.
            count=262144//(width//8)
            params=dict(WORD_BITS=width,HIT_CHAIN=chain,SERVICE_MODE=mode,SETS=32,SET_W=5,
                WAYS=8,WAY_W=3,REGISTER_VICTIM_OWNER=1,DCO_REGISTERED_RANGES=1)
            label=f'w{width}-chain{chain}-mode{mode}'
            log=run_case(str(Path(temp)/'home.vvp'),'tb_nuca_wide',params,'tb/nuca/tb_nuca_wide.v',
                [f'+nreq={count}','+min_wb=0'])
            (a.out/(label+'.log')).write_text(log)
            match=re.search(r'HOME SERVICE mode=(\d+) word_bytes=(\d+) accepted=(\d+) responses=(\d+) span_cycles=(\d+) hit=(\d+) miss=(\d+)',log)
            assert match,label
            observed=dict(zip(('mode','word_bytes','accepted','responses','span_cycles','hit','miss'),map(int,match.groups())))
            assert observed['accepted']==observed['responses']==count,observed
            observed.update(configuration=params,useful_bytes=count*(width//8),
                gb_per_s_at_100mhz=count*(width//8)/observed['span_cycles']*0.1)
            rows.append(observed)
            print(label,json.dumps(observed,sort_keys=True),flush=True)
sources=[ROOT/'rtl/nuca/tnca_nuca_slice.v',ROOT/'tb/nuca/tb_nuca_wide.v',Path(__file__).resolve()]
(a.out/'results.json').write_text(json.dumps(dict(evidence='standalone-home-RTL-with-synthetic-DRAM',
    board_measured=False,clock_hz_assumed=100000000,dram_latency_configured=40,
    response_backpressure=False,outstanding_request_tags=32,
    scope='home word service only; no AXI, NoC, RF, torch or board throughput guarantee',
    sources_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},rows=rows),indent=2)+'\n')
