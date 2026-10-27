from pathlib import Path
import os,subprocess,json,time,hashlib,concurrent.futures as cf
h=Path.home();job=Path(__file__).resolve().parent
binary=h/'diag_gemm/tnca-chip';assert binary.is_file(),binary
python=str(h/'tc_env/torchenv/bin/python')
cases=[('fe1b-n16',h/'r40v3_capgate_fe1b','16,16,16'),('resident-n32',job/'source','16,32,16'),('lowering-n16',job/'source-v2','16,16,16'),('lowering-n32',job/'source-v2','16,32,16')]
state={'started':time.time(),'model_binary':str(binary),'model_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'evidence_level':'model-prediction-uncalibrated','not_measured':['board performance','gate improvement'],'runs':[]}
def one(case):
 name,root,tile=case;harness=h/'m2/src_fe1b/tools/performance_harness/systemc/run_graphs.py';config=h/'diag_gemm/final.cfg'
 cmd=[python,str(harness),'--source-root',str(root),'--binary',str(binary),'--config',str(config),'--out',str(job/('model-'+name)),'--point','op.linear/medium/fp16-m128-k320-n256','--backend','systemc-tlm-grid','--jobs','1','--memory-bytes','4294967296','--active-cores','1','--env','TNCA_GEMM_TILE='+tile,'--env','TNCA_PIPELINE=1','--torch-extensions-dir',str(job/('model-ext-'+name))]
 record={'name':name,'command':cmd,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'harness_sha256':hashlib.sha256(harness.read_bytes()).hexdigest()}
 with (job/('model-'+name+'.log')).open('w') as log:
  p=subprocess.run(cmd,cwd=job,stdout=log,stderr=subprocess.STDOUT)
 record['returncode']=p.returncode
 return record
with cf.ThreadPoolExecutor(max_workers=2) as pool:
 for record in pool.map(one,cases):
  state['runs'].append(record);(job/'model-m2-state.json').write_text(json.dumps(state,indent=2))
state['finished']=time.time();(job/'model-m2-state.json').write_text(json.dumps(state,indent=2))
