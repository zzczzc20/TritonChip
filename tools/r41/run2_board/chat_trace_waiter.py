#!/usr/bin/env python3
"""Retry the preserved VL stall with timed stacks after the two priority chats."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

home = Path.home()
root = home/'r41run3/runtime_prepared'
work = home/'r41run3'
inputs = work/'chat_trace_inputs'
out = work/'chat_trace_queue'
prior = work/'chat_priority_20261031'
pid = 1149434
sys.path.insert(0,str(root/'tools/r41/run2_board'))
import auto  # noqa: E402

manifest = inputs/'manifest.json'
wrapper = inputs/'trace_wrapper.py'
expected_manifest = '3b04738d8aca0aaf8943561a8277384822cdf09fad99a8d35bda413732ae4fb4'
expected_image = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
expected_uuid = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
if auto.sha(manifest) != expected_manifest:
    raise ValueError('VL diagnostic registration changed')
spec = json.loads(manifest.read_text())
if auto.sha(wrapper) != spec['registration']['trace_wrapper_sha256']:
    raise ValueError('VL diagnostic wrapper changed')
if out.exists():
    raise ValueError('VL diagnostic output already exists')
out.mkdir()
with (work/'chat_trace_waiter.lock').open('a') as exclusive,(out/'queue.jsonl').open('x',buffering=1) as log:
    fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def record(kind,**kw):
        log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
    record('waiting-priority-chats',queue=str(prior),pid=pid,manifest_sha256=expected_manifest)
    while True:
        events=[json.loads(x) for x in (prior/'queue.jsonl').read_text().splitlines() if x]
        if events and events[-1].get('kind') == 'completed':
            record('priority-chats-finished')
            break
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,ProcessLookupError):
            cmd=''
        if 'chat_priority_20261031.py' not in cmd:
            record('priority-queue-stopped',last_event=events[-1] if events else None)
            break
        time.sleep(10)
    while True:
        row=auto.probe(Path('/sys/bus/pci/devices'))
        if row['uuid'] != expected_uuid:
            raise RuntimeError('candidate image changed; no programming attempted')
        if not auto.idle(row) or auto.clock_mhz(Path('/sys/bus/pci/devices')) != 100:
            time.sleep(5)
            continue
        with (work/'candidate_queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                row=auto.probe(Path('/sys/bus/pci/devices'))
                if auto.idle(row) and row['uuid']==expected_uuid and auto.clock_mhz(Path('/sys/bus/pci/devices'))==100:
                    if (auto.sha(manifest)!=expected_manifest or auto.sha(wrapper)!=spec['registration']['trace_wrapper_sha256']
                            or auto.sha(work/'tnca_u280.xclbin')!=expected_image):
                        raise ValueError('diagnostic inputs changed')
                    record('stage-start',board=row)
                    py=home/'tc_env/torchenv/bin/python'
                    cmd=[str(py),str(root/'tools/performance_harness/experiment.py'),str(manifest),'--root',str(root),
                         '--out',str(out/'experiment'),'--python',str(py),'--budget-seconds','6000']
                    with (out/'experiment.log').open('x') as stream:
                        try:
                            run=subprocess.run(cmd,cwd=root,env=auto.board_environment(),stdout=stream,
                                               stderr=subprocess.STDOUT,timeout=6150)
                            rc=run.returncode
                        except subprocess.TimeoutExpired:
                            rc=124
                    report=out/'experiment/report.json'
                    record('stage-end',returncode=rc,report_sha256=auto.sha(report) if report.is_file() else None)
                    break
        time.sleep(5)
