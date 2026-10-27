from pathlib import Path
import subprocess,json,sys,os,time
h=Path.home();job=Path(__file__).resolve().parent;root=job/'source-v2'
python=str(h/'tc_env/torchenv/bin/python');env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(root/x) for x in ('.','python','compiler/python','compiler/test','tools/asm')),R1_STRICT='1',PYTHONDONTWRITEBYTECODE='1')
sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
from archbench.provenance import software_tree
identity=software_tree(root,refresh=True)
if identity!='files:60cf7062521a3e5d84677016836f48bf331270c389aaf4166f1f8d3d5818ef4c':raise RuntimeError('candidate identity mismatch: '+identity)
state={'software_tree':identity,'started':time.time(),'runs':[],'not_measured':['board performance','gate improvement']}
commands=[('unit',[python,'compiler/test/run_tests.py','--no-rtl']),('preflight',[python,'tools/r40/v3_compile_preflight.py','--source-root',str(root),'--software-tree',identity,'--out',str(job/'gemm-preflight-v2'),'--device','native-stub','--jobs','4','--items',str(job/'gemm-preflight-items.json'),'--torch-extensions-dir',str(job/'torch_extensions-v2')])]
for name,cmd in commands:
 r={'name':name,'command':cmd,'status':'running'};state['runs'].append(r);(job/'lowering-v2-state.json').write_text(json.dumps(state,indent=2))
 with (job/('lowering-v2-'+name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 r.update(status='pass' if p.returncode==0 else 'fail',returncode=p.returncode);(job/'lowering-v2-state.json').write_text(json.dumps(state,indent=2))
 if p.returncode:break
state['software_tree_after']=software_tree(root,refresh=True);state['finished']=time.time();(job/'lowering-v2-state.json').write_text(json.dumps(state,indent=2))
