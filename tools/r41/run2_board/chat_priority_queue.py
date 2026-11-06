#!/usr/bin/env python3
import datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
h=Path.home();root=h/'r41run3/runtime_prepared';work=h/'r41run3';out=work/'chat_priority_20261031';py=h/'tc_env/torchenv/bin/python';sys.path.insert(0,str(root/'tools/r41/run2_board'));import auto
out.mkdir(exist_ok=False)
with (work/'candidate_queue.lock').open('a') as lock,(out/'queue.jsonl').open('x',buffering=1) as log:
 def record(kind,**kw):log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=auto.board_environment();image=work/'tnca_u280.xclbin'; expected='791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09';uuid='ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
 if auto.sha(image)!=expected:raise ValueError('candidate image changed')
 for name in ('chat-vl','chat-q35'):
  registration=root/'spec/r41/run3/stages'/(name+'.json');reg=json.loads(registration.read_text());assert reg['id']==name and reg['image_identity']=={'sha256':expected,'uuid':uuid}
  for item in reg['experiments']:
   if auto.sha(root/item['manifest'])!=item['sha256']:raise ValueError('chat manifest changed')
  while True:
   try:row=auto.probe(Path('/sys/bus/pci/devices'));ready=auto.idle(row) and auto.clock_mhz(Path('/sys/bus/pci/devices'))==100
   except (OSError,ValueError) as exc:record('probe-error',error=str(exc));time.sleep(5);continue
   if row['uuid']!=uuid:raise RuntimeError('candidate image changed; no programming attempted')
   if ready:break
   record('waiting-card',board=row);time.sleep(5)
  record('stage-start',stage=name,board=row,registration_sha256=auto.sha(registration))
  target=out/name;cmd=[str(py),str(root/'tools/r41/run2_board/stage.py'),'--root',str(root),'--out',str(target),'--python',str(py),'--registration',str(registration)]
  with (out/(name+'.log')).open('x') as stream:
   try:run=subprocess.run(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=11000);rc=run.returncode
   except subprocess.TimeoutExpired:rc=124;record('stage-timeout',stage=name)
  summary=target/'stage.json';record('stage-end',stage=name,returncode=rc,status=json.loads(summary.read_text()).get('status') if summary.exists() else None,summary_sha256=auto.sha(summary) if summary.exists() else None)
 record('completed',scope='both requested chat stages attempted; examine each result and failures')
