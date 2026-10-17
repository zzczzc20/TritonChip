import sys,subprocess
from pathlib import Path
root=Path.home()/'tritonchip-r40-bar-full'
sys.path.insert(0,str(root/'tools/r40'))
import hbm_tests as h
work=root/'sim/build/r40/ram_capacity_longhost'
env=h._env();env['TNCA_PROFILE']=h.HBM_PROFILE;env['TNCA_RTL_PARAMS']=h.RAM_CACHE;env['CUDA_VISIBLE_DEVICES']='';env['TNCA_HOST_TIMEOUT']='172800'
work.mkdir();(work/'child.py').write_text(h._CHILD)
print('prepare',h._prepare(h.RAM_CACHE),flush=True)
with (work/'full.log').open('w') as f:
    r=subprocess.run([h._torch_python(),str(work/'child.py'),'capacity'],cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=172800)
print('returncode',r.returncode,flush=True)
print((work/'full.log').read_text()[-2500:],flush=True)
