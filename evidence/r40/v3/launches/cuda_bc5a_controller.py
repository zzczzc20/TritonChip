from pathlib import Path
import json,os,subprocess,datetime,sys,traceback
h=Path.home();p=Path(__file__).parent;s=h/'r40v3_fixsrc'
sys.path[:0]=[str(s),str(s/'python'),str(s/'compiler/python'),str(s/'tools/asm'),str(s/'tools/r40')]
from archbench.provenance import software_tree
assert software_tree(refresh=True)=='files:bc5a778ebc5f4afebf6c1a9fbf3b2d8717cb02e118134ebbb73fb5a927fc8183'
env={k:v for k,v in os.environ.items() if not k.startswith('TNCA_')}
env.update(MAX_JOBS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TNCA_PROFILE='core-vrf-sync-v5',TNCA_HOST_TIMEOUT='1800',TNCA_DEADLINE='200000000',TORCH_EXTENSIONS_DIR=str(h/'.cache/tnca/r40v3_bc5a_extensions'),PYTHONPATH=os.pathsep.join(str(s/d) for d in ('','python','compiler/python','tools/asm')))
env.pop('CUDA_VISIBLE_DEVICES',None)
python=str(h/'tc_env/torchenv/bin/python')
config=dict(core_count=3,memory_ports=3,memory_bytes=8388608,maintenance=True,hbm_dma=True,board_top=True,noc_link_stages=2,dispatch_stages=2,home_stages=2,code_stages=2)
stages=[('storage-owner-unit',[python,str(s/'tools/r40/storage_owner_tests.py')],dict(env)),('signed-copy-range',[python,str(s/'tools/r40/copy_range_tests.py')],dict(env)),('native-recovery',[python,'-u',str(s/'tools/r40/recovery_tests.py'),'--multicore-dma'],dict(env,CUDA_VISIBLE_DEVICES='')),('cuda-device-conformance',[python,'-u',str(s/'tools/r40/device_tests.py')],dict(env,TNCA_BACKEND='fpga-multicore-sim',TNCA_BACKEND_ARGS=json.dumps(config)))]
state=dict(status='running',software_tree=software_tree(),completed=[],required_gate_promotions=[],scope='Complete repaired-source CUDA-host conformance and recovery on GB10; actual CUDA/TNCA coexistence, TNCA functional RTL only; no FPGA-host simulation or physical card')
def save():
 state['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=p/'state.tmp';q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p/'state.json')
try:
 (p/'commands.json').write_text(json.dumps([dict(name=n,command=c) for n,c,e in stages],indent=2)+'\n');save()
 for name,command,childenv in stages:
  state['current']=name;save()
  with (p/(name+'.log')).open('x') as f:
   q=subprocess.Popen(command,cwd=s,env=childenv,stdout=f,stderr=subprocess.STDOUT);state['child_pid']=q.pid;save();code=q.wait()
  state['completed'].append(dict(name=name,exit_code=code));save();assert code==0,name+' failed'
 state.update(status='complete-review-required',exit_code=0);save()
except Exception:
 state.update(status='failed-preserved',error=traceback.format_exc());save();raise
