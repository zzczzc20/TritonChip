#!/usr/bin/env python3
"""Rerun the restored grid admission case after the registered run-1 rechecks."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

TREE = 'files:e7b332afffab71a7559a42e3b22fe11572758814fb514c9027e22fd64045ca4f'
UUID = '49964b17-e0b4-5bd2-dd40-7fc37b1205d3'
SHA = '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
SCRIPT = 'e1e4a2ff56bf7d72a0d85f121ae197f63d662696078f00f3a188e4a9c7d1bbf1'
MANIFEST = 'd267f35e5677ed7b273ce840c147539d868b72a02109eec3eb88ecf2829ed7d4'
ORDER = ('candidate_membound_default', 'candidate_original113', 'oldmem_membound_default',
         'oldoriginal_original113')


def upstream_done(queue, sha):
    events = [json.loads(x) for x in (queue/'queue.jsonl').read_text().splitlines() if x]
    if not events or events[-1].get('kind') != 'completed':
        return False
    ends = [x for x in events if x.get('kind') == 'stage-end']
    if tuple(x.get('stage') for x in ends) != ORDER:
        raise ValueError('recheck stage inventory differs')
    for event in ends:
        path = queue/event['stage']/'report.json'
        if event.get('returncode') != 0 or event.get('status') != 'pass' or sha(path) != event.get('report_sha256'):
            raise ValueError('recheck stage failed or evidence changed')
    clock = queue/'run1_clock_verify.json'
    clock_events = [x for x in events if x.get('kind') == 'run1-clock-verified']
    if (len(clock_events) != 1 or sha(clock) != clock_events[0].get('report_sha256')
            or json.loads(clock.read_text()).get('status') != 'pass'):
        raise ValueError('restored run-1 clock was not verified')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'manifest', 'predecessor', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--predecessor-pid', type=int, required=True)
    p.add_argument('--poll', type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('invalid polling interval')
    root, work, manifest, predecessor, out = (x.resolve() for x in
                                               (a.root, a.work, a.manifest, a.predecessor, a.out))
    python = a.python.absolute()
    sys.path[:0] = [str(root/'tools/r41/run2_board'), str(root/'tools/performance_harness')]
    import auto
    import experiment
    spec = json.loads(manifest.read_text())
    if (auto.sha(manifest) != MANIFEST or spec.get('id') != 'r41-grid-admission-rerun' or spec['board']['image_sha256'] != SHA
            or spec['board']['image_uuid'] != UUID or [s['id'] for s in spec['steps']] != ['grid']
            or spec['registration']['script_sha256'] != SCRIPT or spec['registration']['software_tree'] != TREE):
        raise ValueError('grid rerun registration differs')
    manifest_sha = auto.sha(manifest)
    def inputs_ok():
        if auto.sha(manifest) != manifest_sha or auto.sha(root/'tools/r40/grid_admission_case.py') != SCRIPT:
            raise ValueError('grid manifest or script changed')
        if experiment.software_tree(root) != TREE:
            raise ValueError('merged software tree changed')
    inputs_ok()
    if out.exists():
        raise ValueError('grid queue output already exists')
    out.mkdir(parents=True)
    with (work/'grid_rerun_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
        record('waiting-upstream', queue=str(predecessor), pid=a.predecessor_pid, manifest_sha256=manifest_sha)
        while not upstream_done(predecessor, auto.sha):
            try:
                cmd = Path(f'/proc/{a.predecessor_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            except (FileNotFoundError, ProcessLookupError):
                cmd = ''
            if 'perf_recheck_waiter.py' not in cmd or str(predecessor) not in cmd:
                record('upstream-stopped')
                raise RuntimeError('recheck queue stopped before completion')
            time.sleep(a.poll)
        record('upstream-verified')
        while True:
            row = auto.probe(Path('/sys/bus/pci/devices'))
            if not auto.idle(row):
                time.sleep(a.poll)
                continue
            with (work/'candidate_queue.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    inputs_ok()
                    row = auto.probe(Path('/sys/bus/pci/devices'))
                    if auto.idle(row):
                        if row['uuid'] != UUID or auto.clock_mhz(Path('/sys/bus/pci/devices')) != 100:
                            raise RuntimeError('run-1 image or clock changed before grid rerun')
                        target = out/'grid-experiment'
                        cmd = [str(python),str(root/'tools/performance_harness/experiment.py'),str(manifest),
                               '--root',str(root),'--out',str(target),'--python',str(python),'--budget-seconds','900']
                        record('stage-start', board=row)
                        with (out/'grid.log').open('x') as stream:
                            result = subprocess.run(cmd, cwd=root, env=auto.board_environment(), stdout=stream,
                                                    stderr=subprocess.STDOUT, timeout=960)
                        report = target/'grid/report.json'
                        value = json.loads(report.read_text()) if report.is_file() else {}
                        record('stage-end', returncode=result.returncode, status=value.get('status'),
                               report_sha256=auto.sha(report) if report.is_file() else None)
                        if result.returncode or value.get('status') != 'pass':
                            raise RuntimeError('grid rerun failed; original report retained')
                        record('completed')
                        break
            time.sleep(a.poll)


if __name__ == '__main__':
    main()
