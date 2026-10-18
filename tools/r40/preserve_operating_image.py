#!/usr/bin/env python3
"""Preserve a terminal implementation and serially audit an operating clock.

No implementation, bitstream regeneration, programming, clock programming or
acceptance promotion. Wait for the existing report observer before opening DCP.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--observer-state',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mhz',type=int,default=91)
    p.add_argument('--max-wait',type=float,default=86400)
    a=p.parse_args()
    if not 1<=a.mhz<=100 or a.max_wait<=0:p.error('positive wait and operating MHz in 1..100 required')
    build=a.build.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tnca-preserved-operating-image/1',scope=__doc__,build=str(build),
                requested_mhz=a.mhz,original_contract_gate='not-passed',conditional_gate='not-measured',
                tool_sha256=sha(Path(__file__)),artifacts=[],steps=[])
    def save(status):
        record.update(status=status,observed_utc=datetime.now(timezone.utc).isoformat())
        temporary=out/'state.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out/'state.json')
    save('waiting-existing-implementation-and-observer')
    deadline=time.monotonic()+a.max_wait
    while True:
        terminal=build/'build/attempt.exit'
        state=json.loads(a.observer_state.read_text()) if a.observer_state.exists() else {}
        observer_status=state.get('status','')
        if (terminal.exists() and terminal.read_text().strip()
                and observer_status and observer_status not in ('watching-existing-build','extracting-reports')):
            break
        if time.monotonic()>deadline:save('observation-timeout-no-retry');return 2
        time.sleep(15)
    # A terminal marker must not be used to overlap a surviving Vivado child.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes()
        except (OSError,PermissionError):continue
        if b'vivado' in cmd.split(b'\0',1)[0].lower() and str(build).encode() in cmd:
            save('vivado-still-active-no-checkpoint-open');return 2
    record['implementation_exit_code']=int(terminal.read_text().strip())
    record['original_observer_state']=state
    runs=build/'xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1'
    checkpoint=runs/'level0_wrapper_routed.dcp'
    files=sorted(set((build/'xo').glob('*.xclbin'))|set(runs.glob('*.bit')))
    for source in files:
        before=source.stat();destination=out/('preserved-'+source.name)
        with source.open('rb') as inp,destination.open('xb') as target:
            shutil.copyfileobj(inp,target,1<<20)
        digest=sha(destination)
        if before.st_size!=source.stat().st_size or digest!=sha(source):
            save('artifact-changed-during-preservation');return 2
        destination.chmod(0o444)
        record['artifacts'].append(dict(source=str(source),preserved=str(destination),
                                        bytes=before.st_size,sha256=digest))
    if not checkpoint.is_file():save('missing-routed-checkpoint-artifacts-preserved');return 2
    record['checkpoint']=dict(path=str(checkpoint),sha256=sha(checkpoint))
    save('artifacts-preserved-extracting-serial-clock-audit')
    def execute(label,command):
        with (out/(label+'.log')).open('x') as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code='report-timeout'
        record['steps'].append(dict(label=label,command=command,exit_code=code))
        save('artifacts-preserved-extracting-serial-clock-audit')
        return code==0
    xclbinutil=shutil.which('xclbinutil') or '/opt/xilinx/xrt/bin/xclbinutil'
    for item in record['artifacts']:
        if item['source'].endswith('.xclbin'):
            image=item['preserved']
            execute('xclbin-info',[xclbinutil,'--input',image,'--info'])
            execute('clock-topology',[xclbinutil,'--input',image,'--dump-section',
                                     'CLOCK_FREQ_TOPOLOGY:JSON:'+str(out/'clock_topology.json')])
    script=ROOT/'flow/fpga/audit_r40_operating_clock.tcl'
    record['clock_audit_script_sha256']=sha(script)
    ok=execute('operating-clock',['vivado','-mode','batch','-log',str(out/'operating.vivado.log'),
               '-journal',str(out/'operating.vivado.jou'),'-source',str(script),'-tclargs',
               str(checkpoint),str(out/'operating-clock-audit'),str(a.mhz),'placement'])
    save('preserved-reports-ready-review-required' if ok else 'preserved-report-failure-review-required')
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
