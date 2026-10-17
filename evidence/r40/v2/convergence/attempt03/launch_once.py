"""One-off final-attempt timer; invokes the existing immutable runner once."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,time,re,csv,os,traceback
home=Path.home();queue=home/'tnca_build/r40_attempt03_queue_20261017';source=home/'tnca_src_r40_attempt03_54249f7';build=home/'tnca_build/r40_attempt03_20261017'
plan=json.loads((queue/'queued.json').read_text());state=dict(plan,state='waiting-current-module')
def save():
 p=queue/'state.json';tmp=queue/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
save()
try:
 module=home/'tnca_build/r40_hit_owner_module_20261017';limit=time.monotonic()+5400
 while not (module/'runner.exit').exists():
  if (queue/'CANCEL').exists():raise RuntimeError('explicit queued-run cancellation marker')
  if time.monotonic()>limit:raise RuntimeError('observation window expired; inspect same module, never restart automatically')
  time.sleep(20)
 assert (module/'runner.exit').read_text().strip()=='0','current module failed; inspect exact terminal evidence before spending last full run'
 text=(module/'reports/timing.rpt').read_text();part=text[text.index('Design Timing Summary'):]
 m=re.search(r'(?m)^\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s',part);assert m,'accurate module min/max summary missing'
 wns,tns,whs,ths=map(float,[m[1],m[2],m[5],m[6]]);assert wns>=0 and whs>=0 and tns==0 and ths==0 and m[3]=='0' and m[7]=='0','module timing failed'
 assert re.search(r'(?m)^clk\s+\{[^}]+\}\s+10\.000\s+100\.000',text),'module clock identity mismatch'
 route=(module/'reports/route_status.rpt').read_text();assert re.search(r'# of nets with routing errors\.+\s*:\s*0\s',route)
 rows=list(csv.DictReader((module/'reports/tmu_state_locations.tsv').open(),delimiter='\t'));assert {r['replica'] for r in rows}=={'0','1','2'} and all(r['slr']=='SLR'+r['replica'] for r in rows),'actual local-state placement missing'
 tags=list(csv.DictReader((module/'reports/tag_mapping.tsv').open(),delimiter='\t'));assert sum(r['primitive']=='RAM32M16' for r in tags)==96,'tag root identity changed'
 cut=home/'tnca_build/r40_owner_boundary_synth_v2_20261017';assert (cut/'runner.exit').read_text().strip()=='0','final mapped cut incomplete'
 proof=(cut/'reports/owner_boundary.txt').read_text();assert re.search(r'direct_metadata_to_notice_paths 0\b',proof) and re.search(r'metadata_ff [1-9]\d* notice_ff [1-9]\d* owner_snapshot_ff [1-9]\d*',proof)
 for name in ['r40_tlw_predicate_module_20261017','r40_owner_boundary_synth_v2_20261017']:
  assert (home/'tnca_build'/name/'runner.exit').read_text().strip()=='0','existing owned vendor task not cleanly terminal'
 pre=json.loads((source/'evidence/r40/v2/convergence/attempt03/preflight.json').read_text());assert pre['source_commit']==plan['source_commit']
 for name,sha in pre['source_inputs'].items():assert digest(source/name)==sha,('immutable source changed',name)
 p=subprocess.run(['python3','tools/r40/contract_v2.py'],cwd=source,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
 config=source/'evidence/r40/v2/convergence/attempt03/image.tcl';assert digest(config)==plan['configuration_sha256']
 assert not build.exists() and not Path(str(build)+'.launch.json').exists(),'existing full attempt; never duplicate'
 evidence={str(p):digest(p) for p in [module/'reports/timing.rpt',module/'reports/route_status.rpt',module/'reports/tmu_state_locations.tsv',module/'reports/tag_mapping.tsv',cut/'reports/owner_boundary.txt',cut/'reports/optimized.dcp']}
 state.update(state='preflight-passed',preflight_utc=datetime.now(timezone.utc).isoformat(),module_setup_wns_ns=wns,module_hold_whs_ns=whs,proof_hashes=evidence);save()
 command=['bash',str(source/'flow/fpga/run_r40_attempt01.sh'),str(build),str(config)]
 manifest=dict(number=3,state='launched',source_commit=plan['source_commit'],source_archive_sha256=plan['source_archive_sha256'],source_root=str(source),build_root=str(build),command=command,configuration='attempt03/image.json',configuration_sha256=digest(config),clock_target_hz=100000000,contract_sha256=pre['contract_sha256'],hypothesis=pre['hypothesis'],started_utc=datetime.now(timezone.utc).isoformat(),preflight=state,image_sha256=None,timing_qualification='not-measured',cdc_qualification='not-measured',no_retry=True,ci_policy=pre['ci_policy'])
 reservation=home/'tnca_build/r40_full_attempt03.reservation.json'
 with reservation.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 manifestpath=Path(str(build)+'.launch.json');manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
 with Path(str(build)+'.detach.log').open('x') as log:p=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 manifest['runner_pid']=p.pid;manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
 out=home/'tnca_build/r40_attempt03_followup'
 observer=['bash','-c','source /tools/Xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1; exec python3 "$1" --build "$2" --runner-pid "$3" --out "$4"','r40-observer',str(source/'tools/r40/watch_implementation.py'),str(build),str(p.pid),str(out)]
 with Path(str(out)+'.detach.log').open('x') as log:w=subprocess.Popen(observer,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 manifest['report_observer']=dict(runner_pid=w.pid,out=str(out),scope='existing read-only observer, after terminal marker');manifestpath.write_text(json.dumps(manifest,indent=2)+'\n');reservation.write_text(json.dumps(manifest,indent=2)+'\n')
 state.update(state='launched',runner_pid=p.pid,observer_pid=w.pid,launch_record=str(manifestpath),remaining_full_attempts=0);save();print(json.dumps(state),flush=True)
except Exception as e:
 state.update(state='preflight-rejected-no-automatic-retry',reason=str(e),traceback=traceback.format_exc());save();print(json.dumps(state),flush=True);raise
