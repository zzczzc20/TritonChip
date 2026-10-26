from pathlib import Path
import subprocess,json,os,sys,time
job=Path(__file__).resolve().parent; root=job/'source'
sys.path[:0]=[str(root),str(root/'python'),str(root/'compiler/python'),str(root/'compiler/test'),str(root/'tools/asm')]
from archbench.provenance import software_tree
# The extra regression file is part of the candidate; record it separately
# while comparing the baseline product with the authoritative delivered tree.
test=root/'compiler/test/test_fragment_residency.py'; testbytes=test.read_bytes(); test.unlink()
before=software_tree(root,refresh=True);test.write_bytes(testbytes)
state={'schema':'r40-followup-fragment/1','baseline_product':before,'started':time.time(),'not_measured':['board performance','gate improvement']}
def save(): (job/'state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
if before!='files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75':raise RuntimeError('baseline identity mismatch')
env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path[:5]),PYTHONDONTWRITEBYTECODE='1')
cmd=[sys.executable,'-m','unittest','discover','-s',str(root/'compiler/test'),'-p','test_fragment_residency.py','-v']
for phase in ('baseline','candidate'):
 if phase=='candidate':
  subprocess.run(['git','apply',str(job/'candidate.patch')],cwd=root,check=True)
  state['candidate_product']=software_tree(root,refresh=True)
 with (job/(phase+'.log')).open('w') as log:
  result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 state[phase+'_returncode']=result.returncode;save()
state['finished']=time.time();save()
