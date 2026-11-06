#!/usr/bin/env python3
"""Lead-started queue: require an approved loaded image and 15 minutes idle.

This script never programs or resets a card and never builds an image. It
runs only specified experiment manifests, with before/after image guards.
--check-only performs manifest/shell checks with no sysfs or card access.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]))
import experiment
sys.path.insert(0, str(HERE.parents[2]))
from tritonchip_fpga_harness.board_lease import board_lease


def absolute_path(path):
    return path.expanduser().absolute()


def wait_idle(probe, expected, *, seconds=900, poll_seconds=10, clock=time.monotonic, sleep=time.sleep, claim=None):
    idle_since = identity = last = None
    while True:
        now, row = clock(), probe()
        # total_execs detects work which started and finished between polls.
        idle = (row.get('contexts') == 0 and row.get('outstanding_execs') == 0
                and not row.get('client_pid') and isinstance(row.get('total_execs'), int))
        current = (row.get('loaded_uuid'), row.get('total_execs'),
                   tuple(row.get('clock_counter_mhz') or ()))
        if idle:
            if (idle_since is None or current != identity or last is None
                    or now < last or now - last > poll_seconds * 1.5):
                idle_since, identity = now, current
            if now - idle_since >= seconds:
                reason = experiment.board_refusal(expected, row)
                if reason:
                    raise RuntimeError(reason)
                # A busy cooperative lock does not invalidate an otherwise
                # continuous idle interval. Keep sampling and retrying; actual
                # activity or a changed image resets the interval above.
                if claim is None:
                    return row
                claimed = claim(row)
                if claimed is True:
                    return row
                if claimed is None:
                    # A recheck under the lease actually saw an interruption;
                    # only a merely busy lease preserves the idle interval.
                    idle_since = identity = None
        else:
            idle_since = identity = None
        last = now
        sleep(poll_seconds)


def queue_board(spec):
    """All FPGA steps must use the identity whose idle time and lease we check.

    experiment.run permits per-step board overrides. A queue that locks only
    the top-level device must reject an override targeting another device,
    image, management function, or clock before waiting or opening a card.
    Other per-step board fields (for example a memory window) remain allowed.
    """
    board = spec.get('board')
    steps = spec.get('steps')
    if not isinstance(board, dict) or not isinstance(steps, list):
        raise ValueError('board manifest required')
    missing = [k for k in experiment.BOARD_FIELDS if board.get(k) in (None, '', {})]
    if missing:
        raise ValueError(f'queue board identity lacks {missing}')
    fpga_steps = [step for step in steps if isinstance(step, dict) and step.get('kind') == 'fpga']
    if not fpga_steps:
        raise ValueError('board manifest required')

    def identity(value):
        device = value['device'].lower()
        mgmt = (value.get('mgmt_device') or device[:-1] + '0').lower()
        return (device, mgmt, value['image_sha256'].lower(),
                experiment.canonical_uuid(value['image_uuid']), value['clock_hz'])

    expected = identity(board)
    for step in fpga_steps:
        override = step.get('board') or {}
        if not isinstance(override, dict):
            raise ValueError('fpga step board override must be an object')
        if identity(dict(board, **override)) != expected:
            raise ValueError(f'fpga step {step.get("id", "<unnamed>")} changes the queued board identity')
    return board


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--budget-seconds',type=float,default=10800,help='total execution budget across manifests; idle waits excluded')
    a=p.parse_args()
    a.root=a.root.resolve();a.out=a.out.resolve();a.python=absolute_path(a.python)
    if not 0 < a.budget_seconds <= 10800:raise ValueError('queue budget must be positive and at most three hours')
    specs=[]
    for name in a.manifest:
        path=name if name.is_absolute() else a.root/name
        spec=json.loads(path.read_text())
        experiment.validate_spec(spec, host=None if a.check_only else socket.gethostname())
        board=queue_board(spec)
        approved={('30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de',100000000),
                  ('a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083',90000000),
                  ('791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09',100000000)}
        if (board['image_sha256'],board['clock_hz']) not in approved:
            raise ValueError('only approved B40, run-1, or corrected run-2 images are admitted')
        specs.append((path,spec))
    if a.check_only:
        print(json.dumps(dict(status='prepared-only',manifests=[str(p) for p,_ in specs],card_access=False)))
        return
    a.out.mkdir(parents=True,exist_ok=True)
    # Linux advisory lock prevents this queue from duplicating itself.
    import fcntl
    with (a.out/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        remaining=a.budget_seconds
        for manifest,spec in specs:
            if remaining<=0:raise RuntimeError('queue execution budget exhausted')
            home=str(Path.home())
            def resolve(value):
                path=Path(value.format(root=a.root,home=home,out=a.out,python=a.python))
                return path if path.is_absolute() else a.root/path
            board=experiment.board_identity(spec['board'],resolve)
            # Probe uses the raw device/sysfs declarations plus verified identity.
            declared=dict(spec['board'],**board)
            with ExitStack() as board_stack:
                def claim(ready):
                    # Take the same lease as interactive chat and candidates.
                    # Keep monitoring while another cooperative user holds it.
                    with ExitStack() as attempt:
                        try:
                            attempt.enter_context(board_lease(declared['device'], blocking=False))
                        except BlockingIOError:
                            return False
                        now = experiment.sysfs_probe(declared)
                        if (experiment.board_refusal(board, now) or
                                now.get('total_execs') != ready.get('total_execs')):
                            return None
                        board_stack.enter_context(attempt.pop_all())
                        return True

                wait_idle(lambda: experiment.sysfs_probe(declared), board, claim=claim)
                target=a.out/spec['id']
                command=[str(a.python),str(a.root/'tools/performance_harness/experiment.py'),str(manifest),
                         '--root',str(a.root),'--out',str(target),'--python',str(a.python),'--budget-seconds',str(remaining)]
                started=time.monotonic()
                with (a.out/(spec['id']+'.log')).open('a') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                remaining-=time.monotonic()-started
                result=json.loads((target/'report.json').read_text())
                if result['status']!='completed-experiment':raise RuntimeError(f"{spec['id']}: {result['status']}")
        (a.out/'DONE').write_text('commands completed; per-point acceptance still required\n')

if __name__=='__main__':main()
