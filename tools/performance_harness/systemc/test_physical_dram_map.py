#!/usr/bin/env python3
"""Same real artifact across fast/slow paths at actual BO/PC boundaries."""
import argparse
import json
import os
from pathlib import Path
import struct
from unittest.mock import patch

from run import run, sha
from validate_chip import assemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--map',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text());mapping=audit['mapping']
    assert sha(args.map)==audit['map_sha256']
    # This frozen board anchor covers PC0-PC15 in one 4 GiB BO. The generic
    # parser/translator has separate non-arithmetic-ID and offset-BO tests.
    assert mapping['bo_base']==0 and mapping['bo_size']==1<<32
    local=mapping['local_limit'];pc_bytes=1<<28
    sources=[local,local+pc_bytes-64,local+pc_bytes,local+(1<<32)-64]
    outputs=[local+0x100000+i*64 for i in range(4)]
    assembly=[];commands=[]
    for i,(source,destination) in enumerate(zip(sources,outputs)):
        path=out/f'input-{i}.bin';path.write_bytes(struct.pack('<16I',*[i*100+j for j in range(16)]))
        commands.append(f'data {source} {path}')
        assembly.extend([f'.li s1, #{source}','v.ld.w v0, [s1, #0]',
                         f'.li s2, #{destination}','v.st.w [s2, #0], v0'])
    assembly.append('exit');image=out/'boundaries.hex';assemble('\n'.join(assembly)+'\n',image)
    script=out/'boundaries.commands'
    script.write_text('\n'.join([f'code {image}',*commands,'launch 1 0 0 1 1 1','fence',
        *[f'expect {address} {out/f"input-{i}.bin"}' for i,address in enumerate(outputs)]])+'\n')
    values=dict(line.split('#')[0].strip().split('=',1) for line in args.config.read_text().splitlines()
                if '=' in line.split('#')[0])
    values.update(memory_bytes=str(local+(1<<32)),dram_max_passes='8')
    config=out/'model.cfg';config.write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
    results=[]
    for path in ('fast','slow'):
        with patch.dict(os.environ,TNCA_DRAM_PATH=path,TNCA_DRAM_MAP=str(args.map.resolve())):
            result=run(args.binary,config,script,out/path,timeout=60,mode='chip')
        assert result['returncode']==0 and result['summary']['checked_bytes']==256,result
        summary=result['summary'];c=summary['counters']
        assert summary['address_map_domain']=='axi-bo-physical-pc' and summary['address_map_sha256']==audit['map_sha256'],summary
        assert summary['bank_row_mapping']=='unverified',summary
        if path=='slow':
            assert c['dram_replay_accepted_bursts']==c['dram_replay_completed_bursts']==2*c['dram_replay_requests'],c
            assert 'dram_replay_queue_refusals' in c and 'dram_replay_peak_rw_queued_bursts_per_pc' in c,c
        assert {k:v for k,v in c.items() if k.startswith('pc_') and k.endswith('_bytes')}=={
            'pc_0_bytes':640,'pc_1_bytes':64,'pc_15_bytes':64},c
        results.append(dict(path=path,status='numeric-and-physical-pc-map-pass',
            model_timing_status=result['model_timing_status'],visible_cycles=summary['visible_cycles']))
    report=dict(status='pass',artifact_sha256=sha(image),map_sha256=sha(args.map),source_image=audit['image_sha256'],
        logical_probe_addresses=sources,physical_pcs=[0,0,1,15],results=results,
        bank_row_mapping='unverified',scope='recorded B40 BO allocation; not new board execution or timing calibration')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
