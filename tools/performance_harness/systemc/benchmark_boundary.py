#!/usr/bin/env python3
"""Offline RTL/TLM memory-boundary cost and error; never whole-chip speedup."""
import argparse
import json
from pathlib import Path
import statistics
from capture import capture
from run import run
from validate import config
from analyze_replay import analyze


def benchmark(binary,out,count):
    out.mkdir(exist_ok=False)
    captured=capture(out/'rtl',count=count,seed=317)
    cfg=config(out/'model.cfg',nx=2,ny=2,sets=16,ways=4,full_line_tag=0,
               mshrs=4,targets=8,outlets=1,active_outlets=1,trace_events=0)
    runs=[run(binary,cfg,out/'rtl/requests.trace',out/f'tlm-{i}',monitor='off') for i in range(4)]
    assert all(r['status']=='completed-prediction' for r in runs)
    error=analyze(out/'rtl',out/'tlm-0')
    elapsed=[r['host_elapsed_seconds'] for r in runs[1:]]
    report=dict(schema='tnca-tlm-boundary-benchmark/1',status='completed-diagnostic',
                rtl_execution_seconds=captured['execution_seconds'],rtl_build_seconds=captured['build_seconds'],
                tlm_execution_seconds=elapsed,rtl_to_tlm_host_time_ratio=captured['execution_seconds']/statistics.median(elapsed),
                comparison_scope='standalone NUCA RTL testbench versus fixed accepted-request TLM replay; one RTL run, three TLM runs after warmup; both include process/output; different frontend work',
                limitations='RTL generates requests and checks protocol; TLM reads captured arrivals and oracle expectations. This ratio is a boundary replay cost comparison, not a matched native-chip speedup.',
                whole_chip_speedup=None,prediction_error=error,performance_gate='not-measured',
                rtl_capture=captured,model_identity=runs[0]['identity'])
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=8000)
    a=p.parse_args()
    if a.count<1:p.error('positive request count required')
    print(json.dumps(benchmark(a.binary,a.out,a.count)))
