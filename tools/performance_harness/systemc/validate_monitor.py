#!/usr/bin/env python3
"""Matched off/aggregate/trace experiments; host overhead is workload-specific."""
import argparse
import json
from pathlib import Path
import statistics
from run import run, sha
from validate import config


def campaign(binary, out, requests=30000, pairs=5):
    out.mkdir(exist_ok=False)
    cfg=config(out/'model.cfg', trace_events=requests*8)
    trace=out/'requests.trace'
    # Read-only independent transactions: reordering cannot hide a value error.
    with trace.open('x') as f:
        for i in range(requests):
            f.write(f'{i+1} 0 {i%3} 0 {0x100000+(i%4096)*64} 0 255 0 1 0 T\n')
    reports=[]
    for index in range(-1,pairs):
        modes=('off','aggregate','trace') if index%2==0 else ('trace','aggregate','off')
        for mode in modes:
            r=run(binary,cfg,trace,out/f'run-{index}-{mode}',monitor=mode)
            assert r['status']=='completed-prediction', r
            reports.append((index,mode,r))
    baseline=reports[0][2]
    for _,mode,r in reports:
        assert r['summary']==baseline['summary'], 'monitor changed modeled results/counters'
        assert r['output_sha256']['responses.tsv']==baseline['output_sha256']['responses.tsv']
        socket=json.loads((Path(r['command'][-1])/'sockets.json').read_text())
        assert socket['complete'] and socket['mode']==mode
        assert bool(socket['sockets'])==(mode!='off')
        for label,s in socket['sockets'].items():
            assert s['active']==s['errors']==0 and sum(s['latency_log2_histogram'])==s['calls']
            assert s['busy_cycles']<=socket['cycles'] and s['peak']>=1
        events=json.loads((Path(r['command'][-1])/'trace.json').read_text())['traceEvents']
        if mode=='trace':
            socket_events=[e for e in events if e['cat']=='tlm-socket']
            assert len(socket_events)==sum(s['calls'] for s in socket['sockets'].values())
            assert len({(e['tid'],e['args']['transaction']) for e in socket_events})==len(socket_events)
        else:
            assert not events
    times={(i,m):r['host_elapsed_seconds'] for i,m,r in reports if i>=0}
    overhead={}
    for mode in ('aggregate','trace'):
        values=[times[i,mode]/times[i,'off']-1 for i in range(pairs)]
        overhead[mode]=dict(pairs=values,observed_relative_range=[min(values),max(values)],
                            median_relative_overhead=statistics.median(values))
    result=dict(schema='tnca-tlm-monitor-validation/1',status='passed',requests=requests,
                overhead=overhead,simulated_cycle_delta=0,response_difference=False,
                scope='memory subsystem read stream; process startup/execution/output, excludes build and hashing; warmup excluded',
                uncertainty='observed host overhead range, not a confidence interval or prediction-error bound',
                rtl_speedup=None,timing_error_bound_cycles=None,performance_gate='not-measured',
                inputs_sha256={'config':sha(cfg),'trace':sha(trace)},
                reports=[dict(pair=i,mode=m,path=str(Path(r['command'][-1])/'report.json'),
                              sha256=sha(Path(r['command'][-1])/'report.json')) for i,m,r in reports])
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--requests',type=int,default=30000)
    p.add_argument('--pairs',type=int,default=5)
    a=p.parse_args()
    if a.requests<1 or a.pairs<3:p.error('positive requests and at least three pairs required')
    print(json.dumps(campaign(a.binary,a.out,a.requests,a.pairs)))
