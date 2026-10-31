#!/usr/bin/env python3
"""Run registered XL matrix pilots after verified full Q41 sessions, without loading."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import q41_pilot_waiter as common

REQUIRED_INPUTS = {'spec/r41/stage2/xl_core.json', 'tools/models/qwen35_reference.py',
                   'tools/models/qwen_fp8_lib.py', 'tools/models/flux2_klein_run.py',
                   'tools/models/fp8_blockwise_ref.py', 'tools/models/fp8_checkpoint.py',
                   'tools/r41/run2_board/auto.py'}
STAGES = [dict(id='transpose', point='XL41-transpose', point_sha256='e09eb6a3c4419687f172a7c582476b20cd425791536ce00b9d277c95b2e7eb41'),
          dict(id='gemm', point='XL41-gemm', point_sha256='bfbea158bb1e5318c7d48fd1ed410276dd6bce51295e336ef30d1077799cf6ec'),
          dict(id='projection', point='XL41-projection', point_sha256='f399ffcb957670cde6a4624991a4143396073386a95a31e13db57926fab33799')]


def verify_q41_full(queue, events, full_plan, auto):
    if not events or events[-1].get('kind') != 'completed':
        raise ValueError('Q41 full queue is not completed')
    ends = [x for x in events if x.get('kind') == 'stage-end' and x.get('returncode') == 0]
    if [x.get('stage') for x in ends] != [x['id'] for x in full_plan['stages']]:
        raise ValueError('Q41 full stage inventory differs')
    for stage, end in zip(full_plan['stages'], ends):
        if end.get('status') != 'pass':
            raise ValueError('Q41 full stage did not pass')
        path = queue/f"{stage['id']}-attempt-{end['attempt']:03d}"/'report.json'
        if auto.sha(path) != end.get('report_sha256'):
            raise ValueError('Q41 full report changed')
        report = json.loads(path.read_text())
        if (report.get('status') != 'pass' or report.get('device') != 'tnca'
                or report.get('script_sha256') != full_plan['pilot_script_sha256']
                or report.get('point') != stage['point'] or report.get('steps_requested') != stage['steps']
                or report.get('board', {}).get('image_sha256') != common.IMAGE_SHA
                or report.get('board', {}).get('uuid') != common.IMAGE_UUID
                or report.get('board', {}).get('clock_hz') != 100_000_000
                or len(report.get('steps', [])) != stage['steps']
                or not all(x.get('passed') is True for x in report['steps'])):
            raise ValueError('Q41 full report identity or checks differ')
    return True


def queue_alive(pid, queue):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'q41_full_waiter.py' in command and str(queue) in command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'queue', 'pilot', 'guard', 'registration', 'q41-registration', 'inputs', 'cpu-evidence', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be within 1..60 seconds')
    root, work, queue, pilot, guard, registration, q41_registration, inputs, cpu_evidence, out = (
        x.resolve() for x in (a.root, a.work, a.queue, a.pilot, a.guard, a.registration,
                              a.q41_registration, a.inputs, a.cpu_evidence, a.out))
    sys.path.insert(0, str(root/'tools/r41/run2_board'))
    import auto
    plan = json.loads(registration.read_text())
    if (plan.get('schema') != 'tnca-r41-xl41-matrix-queue/1' or plan.get('stages') != STAGES
            or plan.get('timeout_seconds') != 3600 or plan.get('image_sha256') != common.IMAGE_SHA
            or plan.get('image_uuid') != common.IMAGE_UUID or plan.get('clock_hz') != 100_000_000
            or set(plan.get('inputs', {})) != REQUIRED_INPUTS):
        raise ValueError('invalid XL matrix registration')
    plan_sha = auto.sha(registration)
    dependency = Path(__file__).resolve().parent/'q41_pilot_waiter.py'
    if (auto.sha(pilot) != plan['pilot_script_sha256'] or auto.sha(guard) != plan['board_guard_sha256']
            or auto.sha(q41_registration) != plan['q41_full_registration_sha256']
            or auto.sha(cpu_evidence) != plan['cpu_preflight_sha256']
            or auto.sha(dependency) != plan['waiter_dependency_sha256']):
        raise ValueError('pilot, CPU evidence or predecessor registration differs')
    for name, digest in plan['inputs'].items():
        if auto.sha(root/name) != digest:
            raise ValueError('frozen XL input changed: '+name)
    full_plan = json.loads(q41_registration.read_text())
    if out.exists():
        raise ValueError('XL matrix queue output already exists')
    out.mkdir(parents=True)
    with (work/'xl41_matrix_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kwargs):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kwargs))+'\n')
        record('waiting-q41-full', queue=str(queue), pid=a.queue_pid, registration_sha256=plan_sha)
        while True:
            events = common.read_events(queue/'queue.jsonl')
            if events and events[-1].get('kind') == 'completed':
                verify_q41_full(queue, events, full_plan, auto)
                record('q41-full-verified')
                break
            if not queue_alive(a.queue_pid, queue):
                record('q41-full-stopped')
                raise RuntimeError('Q41 full queue stopped before completion')
            time.sleep(a.poll)
        for stage in STAGES:
            attempt, previous = 0, None
            while True:
                if (auto.sha(registration) != plan_sha or auto.sha(pilot) != plan['pilot_script_sha256']
                        or auto.sha(guard) != plan['board_guard_sha256']
                        or auto.sha(q41_registration) != plan['q41_full_registration_sha256']
                        or auto.sha(dependency) != plan['waiter_dependency_sha256']):
                    raise ValueError('XL registration or source changed while waiting')
                for name, digest in plan['inputs'].items():
                    if auto.sha(root/name) != digest:
                        raise ValueError('frozen XL input changed: '+name)
                try:
                    row = auto.probe(a.sysfs)
                    ready = auto.idle(row) and row['uuid'] == common.IMAGE_UUID and auto.clock_mhz(a.sysfs) == 100
                    with (work/'candidate_queue.lock').open('a') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            ready = False
                except (OSError, ValueError):
                    ready, row = False, None
                if not ready:
                    if previous != 'waiting-card':
                        record('waiting-card', stage=stage['id'], board=row)
                        previous = 'waiting-card'
                    time.sleep(a.poll)
                    continue
                attempt += 1
                target = out/f"{stage['id']}-attempt-{attempt:03d}"
                env = auto.board_environment()
                env.update(TNCA_BACKEND='fpga-u280-multicore', TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',
                           TNCA_BACKEND_ARGS=json.dumps(dict(xclbin=str(work/'tnca_u280.xclbin'),
                               image_profile='core-vrf-sync-v5-x8-g2c4-dp', core_count=3, memory_ports=3,
                               hbm_dma=True, hbm_bytes=8*(1 << 30), device_clock=True)),
                           TNCA_OPERATING_CLOCK_EVIDENCE=str(work/'qualified_clock.json'),
                           TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='3600',
                           PYTHONPATH=str(inputs/'transformers/src')+':'+str(root/'tools/models'))
                cmd = [str(a.python), str(pilot), '--root', str(root), '--out', str(target), '--device', 'tnca',
                       '--point', stage['point']]
                record('stage-start', stage=stage['id'], attempt=attempt, board=row)
                try:
                    with (out/f"{stage['id']}-attempt-{attempt:03d}.log").open('x') as stream:
                        run = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=plan['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    record('stage-timeout', stage=stage['id'], attempt=attempt)
                    raise RuntimeError('XL matrix pilot exceeded its registered one-hour limit')
                report_path = target/'report.json'
                summary = json.loads(report_path.read_text()) if report_path.is_file() else {}
                record('stage-end', stage=stage['id'], attempt=attempt, returncode=run.returncode,
                       status=summary.get('status'), report_sha256=auto.sha(report_path) if report_path.is_file() else None)
                error = summary.get('error', '')
                if run.returncode and (error.startswith('RuntimeError: card busy or candidate image/clock changed')
                                       or error.startswith('BlockingIOError:')):
                    previous = None
                    time.sleep(a.poll)
                    continue
                if (run.returncode or summary.get('status') != 'pass' or summary.get('point') != stage['point']
                        or summary.get('point_sha256') != stage['point_sha256']):
                    raise RuntimeError('XL matrix pilot failed; preserve output for diagnosis')
                break
        record('completed', scope='three bounded XL pilots only')


if __name__ == '__main__':
    main()
