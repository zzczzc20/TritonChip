"""Diagnostic only: replay complete captured local RAM, without Torch compilation."""
import json,struct,sys,os
from pathlib import Path
import triton
from tnca.runtime import Session
lat=int(os.environ.get('PROBE_MEM_LATENCY','1'))
if os.environ.get('TNCA_BACKEND')=='sim-file' and lat!=1:
 from tnca import device
 device.BENCH_PARAMS['core-vrf-sync-v5'] += ['-P',f'tnca_device.MEM_LATENCY={lat}']
 os.environ['TNCA_DEVICE']=device.prepare('core-vrf-sync-v5')
p=json.load(open('/tmp/r40_sdpa_failure_ram_snapshot.json'))
ram=Path('/tmp/r40_sdpa_failure_ram_snapshot.bin').read_bytes()
t=p['runtime_task'];variant=sys.argv[1]
backend=os.environ.get('TNCA_BACKEND','fpga-u280')
kw={} if backend=='sim-file' else dict(xclbin='/home/zzhouch/tnca_build/u280_r40hybrid/xo/tnca_u280.xclbin',image_profile='core-vrf-sync-v5',device_clock=True)
s=Session(profile='core-vrf-sync-v5',backend=backend,**kw)
reply=s.dev.upload(0,ram);assert reply.ok
reply=s.dev.upload(t['arg_addr'],bytes.fromhex(t['initial_block']));assert reply.ok
if variant.startswith('zero-gap'):
 reply=s.dev.upload(0x20080,bytes(128));assert reply.ok
elif variant=='zero-out':
 reply=s.dev.upload(0x20040,bytes(64));assert reply.ok
elif not variant.startswith('exact'):raise ValueError(variant)
reply=s.dev.launch(t['code_addr'],t['arg_addr'],role_mask=1,pid=(0,0,0),npid=(1,1,1))
assert reply.ok,str(reply)
reply,raw=s.dev.download(0x20040,64);assert reply.ok
values=list(struct.unpack('<32e',raw));err=max(abs(a-b) for a,b in zip(values,p['expected']))
report=dict(diagnostic_only=True,variant=variant,code_addr=t['code_addr'],arg_addr=t['arg_addr'],actual=values,max_abs_error=err,pass_=all(abs(a-b)<=.002+.002*abs(b) for a,b in zip(values,p['expected'])))
path=Path('/tmp/r40_sdpa_snapshot_replay_'+variant+'.json');assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='actual'}))
s.close()
