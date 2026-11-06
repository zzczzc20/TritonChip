#!/usr/bin/env python3
"""Rebuild and rerun bounded board points from a cache-free source archive."""
import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def read_events(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def verify_extra(queue, sha):
    events = read_events(queue/'queue.jsonl')
    if not events or events[-1].get('kind') != 'completed':
        return False
    ends = [x for x in events if x.get('kind') == 'stage-end']
    if [x.get('stage') for x in ends] != ['gather', 'attention', 'recurrence']:
        raise ValueError('XL predecessor stage inventory differs')
    for event in ends:
        path = queue/f"{event['stage']}-attempt-{event['attempt']:03d}"/'report.json'
        if (event.get('returncode') != 0 or event.get('status') != 'pass'
                or sha(path) != event.get('report_sha256') or json.loads(path.read_text()).get('status') != 'pass'):
            raise ValueError('XL predecessor report differs or failed')
    return True


def pilot_budget(queue, sha, policy, expected_script):
    events = read_events(queue/'queue.jsonl')
    ends = [e for e in events if e.get('kind') == 'pilot-end' and e.get('returncode') == 0 and e.get('status') == 'pass']
    if len(ends) != 1:
        raise ValueError('successful Q41 pilot not found exactly once')
    event = ends[0]
    path = queue/f"attempt-{event['attempt']:03d}"/'report.json'
    if sha(path) != event.get('report_sha256'):
        raise ValueError('Q41 pilot report changed')
    report = json.loads(path.read_text())
    if (report.get('status') != 'pass' or report.get('script_sha256') != expected_script
            or report.get('point') != 'Q41-core' or len(report.get('steps', [])) != 2
            or not all(x.get('passed') is True for x in report['steps'])):
        raise ValueError('Q41 board pilot did not pass')
    prefill, decode = (float(x['wall_seconds']) for x in report['steps'])
    if not all(math.isfinite(x) and x > 0 for x in (prefill, decode)):
        raise ValueError('invalid Q41 pilot wall times')
    budget = max(policy['minimum_seconds'], math.ceil(policy['fixed_seconds']+
                 policy['prefill_multiplier']*prefill + policy['decode_multiplier']*decode*16))
    return budget, sha(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','archive','work','pilot','registration','q41-registration','q41-queue',
                 'predecessor','inputs','out','board-python','model-python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--predecessor-pid', type=int, required=True)
    p.add_argument('--poll', type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('invalid polling interval')
    root, archive, work, pilot, registration, q41_registration, q41_queue, predecessor, inputs, out = (
        x.resolve() for x in (a.root,a.archive,a.work,a.pilot,a.registration,a.q41_registration,
                              a.q41_queue,a.predecessor,a.inputs,a.out))
    board_python, model_python = a.board_python.absolute(), a.model_python.absolute()
    sys.path[:0] = [str(root/'tools/r41/run2_board'), str(root/'tools/performance_harness')]
    import auto
    import experiment
    plan = json.loads(registration.read_text())
    policy = dict(prefill_multiplier=2,decode_multiplier=2,fixed_seconds=1200,minimum_seconds=1800,
                  source='successful board pilot step wall times including state comparisons; bound calculated before each stage')
    if (plan.get('schema') != 'tnca-r41-clean-repro-queue/1' or plan.get('q41_point') != 'Q41-core'
            or plan.get('q41_steps') != 17 or plan.get('budget_rule') != policy
            or plan.get('bounded_stage_seconds') != 3600 or set(plan.get('stage_registrations', {})) != {'identity','hbm8'}
            or plan.get('image_sha256') != '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
            or plan.get('image_uuid') != 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3' or plan.get('clock_hz') != 100000000):
        raise ValueError('clean reproduction registration differs')
    plan_sha = auto.sha(registration)
    def verify_inputs():
        if (auto.sha(registration) != plan_sha or auto.sha(Path(__file__)) != plan['waiter_sha256']
                or auto.sha(archive) != plan['source_archive_sha256']
                or auto.sha(pilot) != plan['pilot_script_sha256']
                or auto.sha(q41_registration) != plan['q41_registration_sha256']
                or experiment.software_tree(root) != plan['software_tree']):
            raise ValueError('archive, source tree, pilot or registration changed')
        for name, digest in plan['stage_registrations'].items():
            if auto.sha(root/'spec/r41/run3/stages'/(name+'.json')) != digest:
                raise ValueError('board stage registration changed: '+name)
        for name, digest in plan['inputs'].items():
            if auto.sha(root/name) != digest:
                raise ValueError('Q41 frozen input changed: '+name)
    verify_inputs()
    if out.exists():
        raise ValueError('clean reproduction output already exists')
    out.mkdir(parents=True)
    with (work/'clean_repro_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
        record('waiting-upstream',queue=str(predecessor),pid=a.predecessor_pid,registration_sha256=plan_sha)
        while not verify_extra(predecessor, auto.sha):
            try:
                cmd = Path(f'/proc/{a.predecessor_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            except (FileNotFoundError,ProcessLookupError):
                cmd = ''
            if 'xl41_extra_waiter.py' not in cmd or str(predecessor) not in cmd:
                record('upstream-stopped')
                raise RuntimeError('XL predecessor stopped before completion')
            time.sleep(a.poll)
        budget, pilot_report_sha = pilot_budget(q41_queue,auto.sha,plan['budget_rule'],plan['pilot_script_sha256'])
        record('upstream-verified',pilot_report_sha256=pilot_report_sha,q41_budget_seconds=budget)
        for name in ('legacy','hbm8','q41'):
            verify_inputs()
            while True:
                try:
                    row = auto.probe(Path('/sys/bus/pci/devices'))
                    idle = auto.idle(row) and auto.clock_mhz(Path('/sys/bus/pci/devices')) == 100
                except (OSError, ValueError):
                    idle, row = False, None
                if row and row['uuid'] != plan['image_uuid']:
                    raise RuntimeError('candidate image changed before '+name)
                if idle:
                    break
                time.sleep(a.poll)
            env = auto.board_environment()
            caches = out/'fresh-caches'/name
            env.update(TRITON_CACHE_DIR=str(caches/'triton'),TORCHINDUCTOR_CACHE_DIR=str(caches/'inductor'),
                       TORCH_EXTENSIONS_DIR=str(caches/'extensions'),XDG_CACHE_HOME=str(caches/'xdg'))
            target = out/name
            if name == 'q41':
                env.update(TNCA_BACKEND='fpga-u280-multicore',TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',
                    TNCA_BACKEND_ARGS=json.dumps(dict(xclbin=str(work/'tnca_u280.xclbin'),
                        image_profile='core-vrf-sync-v5-x8-g2c4-dp',core_count=3,memory_ports=3,
                        hbm_dma=True,hbm_bytes=8*(1 << 30),device_clock=True)),
                    TNCA_OPERATING_CLOCK_EVIDENCE=str(work/'qualified_clock.json'),
                    TNCA_DEADLINE='200000000000',TNCA_HOST_TIMEOUT='3600',
                    PYTHONPATH=str(inputs/'transformers/src')+':'+str(root/'tools/models'))
                command=[str(model_python),str(pilot),'--root',str(root),'--snapshot',str(Path.home()/'r41s2_2b_inputs/snapshot'),
                         '--reference',str(Path.home()/'r41s2_2b_inputs/fp32_reference'),'--out',str(target),
                         '--device','tnca','--point','Q41-core','--steps','17']
                report = target/'report.json'
                timeout = budget
            else:
                stage = 'identity' if name == 'legacy' else 'hbm8'
                command=[str(board_python),str(root/'tools/r41/run2_board/stage.py'),'--root',str(root),
                         '--out',str(target),'--python',str(board_python),'--registration',str(root/'spec/r41/run3/stages'/(stage+'.json'))]
                report = target/'stage.json'
                timeout = plan['bounded_stage_seconds']
            record('stage-start',stage=name,board=row,timeout_seconds=timeout)
            with (out/(name+'.log')).open('x') as stream:
                if name == 'q41':
                    run = subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
                else:
                    with (work/'candidate_queue.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                        current = auto.probe(Path('/sys/bus/pci/devices'))
                        if not auto.idle(current) or current['uuid'] != plan['image_uuid']:
                            raise RuntimeError('card became busy before '+name)
                        run = subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            value = json.loads(report.read_text()) if report.is_file() else {}
            record('stage-end',stage=name,returncode=run.returncode,status=value.get('status'),
                   report_sha256=auto.sha(report) if report.is_file() else None)
            if run.returncode or value.get('status') != 'pass':
                raise RuntimeError('clean reproduction stage failed; preserve output')
            if name == 'q41' and (value.get('point') != 'Q41-core' or len(value.get('steps',[])) != 17
                                  or not all(x.get('passed') is True for x in value['steps'])):
                raise RuntimeError('clean full Q41 verification incomplete')
        record('completed',scope='clean source archive and independent session; review all evidence before acceptance')


if __name__ == '__main__':
    main()
