"""Actual native host DMA maintenance, local split, and fit double-count refusal."""
import argparse
import json
import os
from pathlib import Path
import struct
from unittest.mock import patch

from tlm_backend import backend_class, _config_values
from validate_chip import assemble
from run import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--map',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    values=_config_values(args.config)
    values.update(host_launch_cycles='0',host_transfer_cycles='0',host_transfer_cycles_per_kib='0',memory_bytes=str(1<<24))
    profile=args.out/'device-service.cfg'
    profile.write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
    window=next(line.split() for line in args.map.read_text().splitlines() if line.startswith('window '))
    local=int(window[1],0);source=local+4096;destination=local+8192
    image=args.out/'copy.hex'
    assemble(f'.li s1, #{source}\nv.ld.w v0, [s1, #0]\n.li s2, #{destination}\nv.st.w [s2, #0], v0\nexit\n',image)
    code=b''.join(int(line.strip(),16).to_bytes(8,'little') for line in image.read_text().splitlines() if line.strip())
    data=struct.pack('<16I',*range(16));rows=[]
    for mode in ('coherent-direct','dma-maintenance'):
        with patch.dict(os.environ,TNCA_TLM_HOST_HANDOFF=mode,TNCA_DRAM_MAP=str(args.map.resolve()),TNCA_TLM_UNTIMED_COLD='0'):
            dev=backend_class()(binary=args.binary.resolve(),config=profile.resolve(),out=args.out/mode)
            try:
                dev.start();dev.upload(0,code);dev.upload(source,data)
                assert dev.launch(0,0).ok
                before=dev.counters_snapshot()
                dev.upload(local-8,b'\x00'*8)
                after_local=dev.counters_snapshot()
                assert after_local['host_dma_handoffs']==before['host_dma_handoffs']
                _,actual=dev.download(destination,len(data));assert actual==data
                after=dev.counters_snapshot()
                delta=after['host_dma_handoffs']-before['host_dma_handoffs']
                assert delta==(1 if mode=='dma-maintenance' else 0)
                maintained=sum(after.get(f'home_{i}_maintenance_completed',0)-before.get(f'home_{i}_maintenance_completed',0)
                               for i in range(dev.core_count))
                assert maintained==(dev.core_count if mode=='dma-maintenance' else 0)
                dev.upload(local-8,b'\x00'*16)
                assert dev.host_handoffs==after['host_dma_handoffs']+(1 if mode=='dma-maintenance' else 0)
                rows.append(dict(mode=mode,numeric_pass=True,local_only_no_handoff=True,
                                 straddling_transfer_handoff=True,download_maintenance_slices=maintained,
                                 before=before,after=after,identity=dev.identity))
            finally:dev.stop()
    refusals=[]
    for mode,config,mapping,expected in (
        ('invalid',profile,str(args.map.resolve()),'must be'),
        ('dma-maintenance',args.config,str(args.map.resolve()),'zero fitted host costs'),
        ('dma-maintenance',profile,'','explicit TNCA_DRAM_MAP')):
        with patch.dict(os.environ,TNCA_TLM_HOST_HANDOFF=mode,TNCA_DRAM_MAP=mapping):
            try:
                dev=backend_class()(binary=args.binary.resolve(),config=config.resolve(),out=args.out/'refusals')
                dev.start()
            except ValueError as error:
                assert expected in str(error),str(error)
                refusals.append(str(error))
            else:
                dev.stop();raise AssertionError('unsafe/ambiguous host policy was accepted')
    report=dict(status='pass',rows=rows,refusals=refusals,binary_sha256=sha(args.binary),
                map_sha256=sha(args.map),backend_sha256=sha(Path(__file__).with_name('tlm_backend.py')),
                scope='actual native model host ownership boundaries; no physical timing qualification')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',native_modes=len(rows),refusals=len(refusals)),indent=2))


if __name__=='__main__':main()
