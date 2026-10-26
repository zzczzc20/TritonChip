from pathlib import Path
import os,subprocess,json,sys,time,shlex
job=Path(__file__).resolve().parent;root=job/'source'
python=str(Path.home()/'tc_env/torchenv/bin/python')
env=dict(os.environ,R1_STRICT='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(root/x) for x in ('.','python','compiler/python','compiler/test','tools/asm')),TNCA_BACKEND='rtl',TNCA_PROFILE='core-vrf-sync-v5',TORCH_EXTENSIONS_DIR=str(job/'torch_extensions'))
commands=[('compiler-unit',[python,'compiler/test/run_tests.py','--no-rtl'])]
for dtype in ('float16','bfloat16'):
 commands.append(('gemm-'+dtype,[python,'tools/r40/gemm_tiles_tests.py','--columns','32','--depth','16','--dtype',dtype,'--dump-dir',str(job/('gemm-'+dtype))]))
state={'started':time.time(),'runs':[],'not_measured':['board performance','gate improvement']}
for name,cmd in commands:
 r={'name':name,'command':cmd,'status':'running'};state['runs'].append(r);(job/'batch-state.json').write_text(json.dumps(state,indent=2))
 with (job/(name+'.log')).open('w') as log:
  shell='source '+shlex.quote(str(Path.home()/'tc_env/env.sh'))+' && exec '+shlex.join(cmd)
  p=subprocess.run(['bash','-lc',shell],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 r.update(status='pass' if p.returncode==0 else 'fail',returncode=p.returncode)
 (job/'batch-state.json').write_text(json.dumps(state,indent=2))
 if p.returncode:break
state['finished']=time.time();(job/'batch-state.json').write_text(json.dumps(state,indent=2))
