#!/usr/bin/env python3
"""Lead-started queue: require an approved loaded image and 15 minutes idle.

This script never programs or resets a card and never builds an image. It
runs only specified experiment manifests, with before/after image guards.
--check-only performs manifest/shell checks with no sysfs or card access.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]))
import experiment


def wait_idle(probe,expected,*,seconds=900,poll_seconds=10,clock=time.monotonic,sleep=time.sleep):
    idle_since=None
    while True:
        now=clock();row=probe()
        # Missing or malformed counters must never count as idle.
        idle=(row.get('contexts')==0 and row.get('outstanding_execs')==0 and not row.get('client_pid'))
        if idle:
            if idle_since is None:idle_since=now
            if now-idle_since>=seconds:
                reason=experiment.board_refusal(expected,row)
                if reason:raise RuntimeError(reason)
                return row
        else:idle_since=None
        sleep(poll_seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    a.root=a.root.resolve();a.out=a.out.resolve();a.python=a.python.resolve()
    specs=[]
    for name in a.manifest:
        path=name if name.is_absolute() else a.root/name
        spec=json.loads(path.read_text())
        if not spec.get('steps') or not any(s['kind']=='fpga' for s in spec['steps']):raise ValueError('board manifest required')
        board=spec['board']
        if board['image_sha256']!='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de' or board['clock_hz']!=100000000:
            raise ValueError('only already-approved run-1 image at 100 MHz is admitted')
        specs.append((path,spec))
    if a.check_only:
        print(json.dumps(dict(status='prepared-only',manifests=[str(p) for p,_ in specs],card_access=False)))
        return
    a.out.mkdir(parents=True,exist_ok=True)
    # Linux advisory lock prevents this queue from duplicating itself.
    import fcntl
    with (a.out/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for manifest,spec in specs:
            home=str(Path.home())
            def resolve(value):
                path=Path(value.format(root=a.root,home=home,out=a.out,python=a.python))
                return path if path.is_absolute() else a.root/path
            board=experiment.board_identity(spec['board'],resolve)
            # Probe uses the raw device/sysfs declarations plus verified identity.
            declared=dict(spec['board'],**board)
            wait_idle(lambda:experiment.sysfs_probe(declared),board)
            target=a.out/spec['id']
            command=[str(a.python),str(a.root/'tools/performance_harness/experiment.py'),str(manifest),
                     '--root',str(a.root),'--out',str(target),'--python',str(a.python)]
            with (a.out/(spec['id']+'.log')).open('a') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((target/'report.json').read_text())
            if result['status']!='completed-experiment':raise RuntimeError(f"{spec['id']}: {result['status']}")
        (a.out/'DONE').write_text('commands completed; per-point acceptance still required\n')

if __name__=='__main__':main()
