#!/usr/bin/env python3
"""Validate the P8 FP8 board follow-up from read-only archived records.

The campaign is agy/Gemini commit 63acb94, run by an agy/Gemini session that
the lead launched on the author's instruction of 2026-10-26. It is not the
lead's own run. It depends on a non-default environment: a .pth that sets
torch.version.cuda on CPU-only torch so that the `kernels` package picks its
Triton build. Its correctness results stand against their references, but it
is a named deviation and not a standard-path default result.
"""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
IMAGE='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
UUID='49964b17-e0b4-5bd2-dd40-7fc37b1205d3'

def main():
    with tarfile.open(BASE/'board-records.tgz') as t:
        raw={m.name:t.extractfile(m).read() for m in t if m.isfile()}
    get=lambda name:json.loads(raw[name])
    campaigns={}
    for name in ('s2-fp8-hub','s2-qwen-fp8'):
        report=get(f'out/{name}/report.json')
        assert report['status']=='completed-experiment'
        assert report['identity']['software_tree']=='files:c560dca3f1158106d11437948d09979d4d2b9ae181f3bb5ebaacc5fa6079290e'
        for step in report['steps']:
            assert step['status']=='completed'
            if step['kind']=='fpga':
                board=step['board']
                assert board['image_sha256']==IMAGE and board['image_uuid']==UUID and board['clock_hz']==100000000
                for p in ('probe_before','probe_after'):
                    assert board[p]['loaded_uuid']==UUID and board[p]['clock_counter_mhz'][0]==100
        campaigns[name]=dict(software_tree=report['identity']['software_tree'],steps=len(report['steps']))
    count=0
    for group in ('hub','hubv1'):
        report=get('out/s2-fp8-hub/'+group+'/report.json')
        assert report['backend']=='fpga-u280-multicore'
        assert all(r['status']=='pass' for r in report['cases'])
        count+=len(report['cases'])
    assert count==35
    linears=get('out/s2-qwen-fp8/linears/result.json')
    assert linears['passed']==linears['total']==14
    assert all(row['pass'] and row['vs_tnca_order_rederivation']['bitwise'] for row in linears['rows'])
    decode=get('out/s2-qwen-fp8/decode/result.json')
    assert decode['pass'] and decode['cpu_pair_reproduces_whole_model'] and len(decode['device'])==5
    assert all(row['ok'] for row in decode['summary'])
    for row in decode['device'][2:]:
        assert row['new_graphs']==0
        transfer=row['transfers_inside_step']
        assert transfer['host_read_calls']==transfer['host_write_calls']==0
    env=json.loads((BASE/'environment.json').read_text())
    assert env['status']=='passed' and env['versions']['torch']=='2.14.0+cpu' and not env['actual_cuda_available']
    result=dict(status='passed',scope='P8 only; two FP8 layers are not full BF16 Qwen3.5-2B',
                started_by=('agy/Gemini session (commit 63acb94), launched by the lead on the author\'s '
                            'instruction of 2026-10-26; not run by the lead itself'),
                deviation=('non-default environment: a .pth sets torch.version.cuda on CPU-only torch so '
                           'that the kernels package selects its Triton build; correctness results stand '
                           'against references; not a standard-path default result'),
                image_sha256=IMAGE,clock_hz=100000000,
                hub_cases_passed=count,linears_passed=14,decode_steps_including_prefill=5,
                steady_state_new_graphs=0,steady_state_host_transfer_calls=0,campaigns=campaigns)
    (BASE/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    index=dict(host='acf3019:/home/zzhouch/r41s2_board_86a3d929cf3d_fp8',
               collection='read-only JSON/JSONL, excluding caches/dumps/bundles/step caches',
               sha256=hashlib.sha256((BASE/'board-records.tgz').read_bytes()).hexdigest(),
               members={n:dict(size=len(d),sha256=hashlib.sha256(d).hexdigest()) for n,d in sorted(raw.items())})
    (BASE/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
