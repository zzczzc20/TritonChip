#!/usr/bin/env python3
"""Run registered regression checks after the candidate campaign, then restore run 1."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CANDIDATE = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
OLD = '49964b17-e0b4-5bd2-dd40-7fc37b1205d3'
OLD_SHA = '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
TREES = {'candidate': 'files:e7b332afffab71a7559a42e3b22fe11572758814fb514c9027e22fd64045ca4f',
         'oldmem': 'files:9458e56b1fb26cab97edebc42db8b33cbeb290a92b7dcdb72839ad5b07e82f67',
         'oldoriginal': 'files:c560dca3f1158106d11437948d09979d4d2b9ae181f3bb5ebaacc5fa6079290e'}
ORDER = ('candidate_membound_default', 'candidate_original113', 'oldmem_membound_default',
         'oldoriginal_original113')


def verify_predecessor(queue, auto):
    events = [json.loads(x) for x in (queue/'queue.jsonl').read_text().splitlines() if x]
    if not events or events[-1].get('kind') != 'completed':
        return False
    ends = [x for x in events if x.get('kind') == 'stage-end']
    if [x.get('stage') for x in ends] != ['gather', 'attention', 'recurrence']:
        raise ValueError('upstream XL stage inventory differs')
    for event in ends:
        if event.get('returncode') != 0 or event.get('status') != 'pass':
            raise ValueError('upstream XL stage failed')
        path = queue/f"{event['stage']}-attempt-{event['attempt']:03d}"/'report.json'
        if auto.sha(path) != event.get('report_sha256') or json.loads(path.read_text()).get('status') != 'pass':
            raise ValueError('upstream XL report changed or failed')
    return True


def alive(pid, queue):
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'xl41_extra_waiter.py' in cmd and str(queue) in cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'oldmem', 'oldoriginal', 'plans', 'predecessor', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--predecessor-pid', type=int, required=True)
    p.add_argument('--poll', type=int, default=5)
    a = p.parse_args()
    if not 1 <= a.poll <= 10:
        raise ValueError('poll must be 1..10 seconds')
    root, work, oldmem, oldoriginal, plans, predecessor, out = (x.resolve() for x in
        (a.root, a.work, a.oldmem, a.oldoriginal, a.plans, a.predecessor, a.out))
    python = a.python.absolute()  # Preserve the virtualenv symlink.
    sys.path[:0] = [str(root/'tools/r41/run2_board'), str(root/'tools/performance_harness')]
    import auto
    import experiment
    registration = plans/'registration.json'
    plan = json.loads(registration.read_text())
    if (plan.get('schema') != 'tnca-r41-regression-recheck-registration/1'
            or tuple(plan.get('order', ())) != ORDER or set(plan.get('manifests', {})) != {x+'.json' for x in ORDER}):
        raise ValueError('invalid recheck registration')
    plan_sha = auto.sha(registration)
    image = Path.home()/'tnca_images/r41s2_run1_c9f25787494a/tnca_u280.xclbin'
    roots = {'candidate': root, 'oldmem': oldmem, 'oldoriginal': oldoriginal}
    def verify_inputs():
        if (auto.sha(registration) != plan_sha or auto.sha(image) != OLD_SHA
                or auto.sha(Path(__file__)) != plan.get('waiter_sha256')):
            raise ValueError('recheck registration, waiter or run-1 image changed')
        for name, entry in plan['manifests'].items():
            if auto.sha(plans/name) != entry['sha256']:
                raise ValueError('recheck manifest changed: '+name)
        for name, tree_root in roots.items():
            if experiment.software_tree(tree_root) != TREES[name]:
                raise ValueError('registered software tree changed: '+name)
    verify_inputs()
    if out.exists():
        raise ValueError('recheck queue output already exists')
    out.mkdir(parents=True)
    with (work/'perf_recheck_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
        record('waiting-upstream', queue=str(predecessor), pid=a.predecessor_pid, registration_sha256=plan_sha)
        while not verify_predecessor(predecessor, auto):
            if not alive(a.predecessor_pid, predecessor):
                record('upstream-stopped')
                raise RuntimeError('upstream queue stopped before completion')
            time.sleep(a.poll)
        record('upstream-verified')
        env = auto.board_environment()
        for i, name in enumerate(ORDER):
            arm = name.split('_')[0]
            target = out/name
            with (work/'candidate_queue.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                verify_inputs()
                if i == 2:
                    tracker = auto.IdleTracker(max_gap=max(15, a.poll*2.5))
                    record('waiting-15-minute-idle')
                    while True:
                        try:
                            row = auto.probe(Path('/sys/bus/pci/devices'))
                        except (OSError, ValueError) as error:
                            tracker = auto.IdleTracker(max_gap=max(15, a.poll*2.5))
                            record('idle-probe-error', error=str(error))
                            time.sleep(a.poll)
                            continue
                        if row['uuid'] != CANDIDATE:
                            raise RuntimeError('loaded image changed before run-1 restoration')
                        if tracker.observe(time.monotonic(), row):
                            break
                        time.sleep(a.poll)
                    for mode in ('--check', '--go'):
                        verify_inputs()
                        row = auto.probe(Path('/sys/bus/pci/devices'))
                        if (time.monotonic()-tracker.last > tracker.max_gap or not auto.idle(row)
                                or (row['uuid'], row['total execs']) != tracker.identity):
                            raise RuntimeError('board changed or became busy before load')
                        tracker.last = time.monotonic()
                        record('program', mode=mode, previous=row['uuid'])
                        with (out/'program.log').open('a') as stream:
                            subprocess.run(['bash',str(root/'flow/fpga/program.sh'),mode,str(image)],
                                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    with (out/'clock-verify.log').open('x') as stream:
                        subprocess.run([str(python),str(root/'tools/performance_harness/clock_qualify.py'),
                            'verify','--image',str(image),'--uuid',OLD,'--evidence',str(Path.home()/'tnca_build/r41s2_run1_c9f25787494a/r41s2-board/qualified_clock.json'),
                            '--out',str(out/'run1_clock_verify.json')], env=env, stdout=stream,
                            stderr=subprocess.STDOUT, check=True)
                row = auto.probe(Path('/sys/bus/pci/devices'))
                expected = CANDIDATE if i < 2 else OLD
                if not auto.idle(row) or row['uuid'] != expected or auto.clock_mhz(Path('/sys/bus/pci/devices')) != 100:
                    raise RuntimeError('board busy or image/clock differs before '+name)
                command = [str(python),str(roots[arm]/'tools/performance_harness/experiment.py'),
                           str(plans/(name+'.json')),'--root',str(roots[arm]),'--out',str(target),
                           '--python',str(python),'--budget-seconds','10800']
                record('stage-start',stage=name,board=row)
                with (out/(name+'.log')).open('x') as stream:
                    result = subprocess.run(command,cwd=roots[arm],env=env,stdout=stream,stderr=subprocess.STDOUT,
                                            timeout=10860)
                report = target/'report.json'
                batch = target/'recheck/batch.json'
                if result.returncode or not report.is_file() or not batch.is_file():
                    record('stage-end',stage=name,returncode=result.returncode,status='failed')
                    raise RuntimeError('recheck experiment failed; retain output')
                summary = json.loads(batch.read_text())
                if (json.loads(report.read_text()).get('status') != 'completed-experiment'
                        or summary.get('status') != 'complete' or summary.get('failed_configurations')
                        or summary.get('missing_formal')):
                    record('stage-end',stage=name,returncode=result.returncode,status='failed')
                    raise RuntimeError('recheck results incomplete; retain output')
                record('stage-end',stage=name,returncode=0,status='pass',report_sha256=auto.sha(report),
                       batch_sha256=auto.sha(batch))
        record('completed', scope='one registered recheck block; statistical acceptance remains separate')


if __name__ == '__main__':
    main()
