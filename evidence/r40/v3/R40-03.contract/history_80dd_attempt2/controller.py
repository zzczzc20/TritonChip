from pathlib import Path
import subprocess,os,sys,json,hashlib,datetime,traceback
h=Path.home();p=Path(__file__).parent;s=h/'r40_bucket_full_80dd_v2_source'
sys.path[:0]=[str(s),str(s/'python'),str(s/'compiler/python'),str(s/'tools/asm'),str(s/'tools/r40')]
from archbench.provenance import software_tree
import contract_v2
contract_v2.verify();tree=software_tree(refresh=True);assert tree=='files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca'
state=dict(status='running',software_tree=tree,contract_sha256=contract_v2.LOCK_SHA256,reference_sha256=hashlib.sha256((p/'reference.json').read_bytes()).hexdigest(),harness_sha256=hashlib.sha256((s/'tools/r40/contract_tests.py').read_bytes()).hexdigest(),required_gate_promotions=[],scope='Complete unchanged legacy 149-case inventory at O0/O1/O2, actual artifact byte comparison; GB10 compilation only, no FPGA or performance claim')
def save():
 state['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=p/'state.tmp';q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p/'state.json')
try:
 env={k:v for k,v in os.environ.items() if not k.startswith('TNCA_')};env.update(TILEBENCH_SOURCE=str(h/'tc_env/Tilebench'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='1',TNCA_BANK_ALLOC='0',TNCA_WINDOW_REUSE='0',TNCA_PACKED_WINDOW_UNPACK='0',PYTHONPATH=os.pathsep.join(str(s/d) for d in ('','python','compiler/python','tools/asm')))
 corpus=h/'tc_env/Tilebench'
 commit=subprocess.run(['git','-C',str(corpus),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
 assert commit=='6e2cf155dfe01f4660540d187c81b09eb55f394f'
 subprocess.run(['git','-C',str(corpus),'diff','--quiet','HEAD','--','benchmarks/operators'],check=True)
 state['tilebench_commit']=commit
 state['tilebench_source']=str(corpus)
 state['tilebench_sources_sha256']=hashlib.sha256(''.join(str(f.relative_to(corpus))+':'+hashlib.sha256(f.read_bytes()).hexdigest()+'\\n' for f in sorted((corpus/'benchmarks/operators').glob('*/impl_triton.py'))).encode()).hexdigest()
 cmd=[str(h/'tc_env/torchenv/bin/python'),'-u',str(s/'tools/r40/contract_tests.py'),'--reference',str(p/'reference.json')];(p/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');save()
 with (p/'contract.log').open('x') as log:
  q=subprocess.Popen(cmd,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT);state['child_pid']=q.pid;save();code=q.wait()
 state['exit_code']=code
 assert software_tree(refresh=True)==tree,'immutable product changed during contract comparison'
 log=(p/'contract.log').read_text();state['log_sha256']=hashlib.sha256(log.encode()).hexdigest()
 assert code==0 and log.count('r40-contract-tests PASS (447/447 byte-identical to B39)')==1 and not any(l.startswith('DIFF ') for l in log.splitlines())
 state['status']='complete-review-required';save()
except Exception:
 state.update(status='failed-preserved',error=traceback.format_exc());save();raise
