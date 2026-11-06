#!/usr/bin/env python3
"""Wait for a verified Q41 pilot, then run the registered XL41 GEMV pilot."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import q41_pilot_waiter as prior

REQUIRED_INPUTS = {'spec/r41/stage2/xl_core.json', 'tools/models/qwen35_reference.py',
                   'tools/models/qwen_fp8_lib.py', 'tools/models/flux2_klein_run.py',
                   'tools/models/fp8_blockwise_ref.py', 'tools/models/fp8_checkpoint.py',
                   'tools/r41/run2_board/auto.py'}


def verify_q41(queue, events, q41_plan, auto):
    if not events or events[-1].get('kind') != 'completed':
        raise ValueError('Q41 pilot queue is not completed')
    ends = [x for x in events if x.get('kind') == 'pilot-end' and x.get('returncode') == 0]
    if len(ends) != 1 or ends[0].get('status') != 'pass':
        raise ValueError('Q41 pilot did not pass exactly once')
    end = ends[0]
    path = queue/f"attempt-{end['attempt']:03d}"/'report.json'
    if auto.sha(path) != end.get('report_sha256'):
        raise ValueError('Q41 pilot report changed')
    report = json.loads(path.read_text())
    if (report.get('status') != 'pass' or report.get('device') != 'tnca'
            or report.get('script_sha256') != q41_plan['pilot_script_sha256']
            or report.get('point') != 'Q41-core' or report.get('steps_requested') != 2
            or report.get('board', {}).get('image_sha256') != prior.IMAGE_SHA
            or report.get('board', {}).get('uuid') != prior.IMAGE_UUID
            or report.get('board', {}).get('clock_hz') != 100_000_000
            or len(report.get('steps', [])) != 2 or not all(x.get('passed') is True for x in report['steps'])):
        raise ValueError('Q41 pilot report identity or checks differ')
    return True


def queue_alive(pid, queue):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'q41_pilot_waiter.py' in command and str(queue) in command


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
    if (plan.get('schema') != 'tnca-r41-xl41-gemv-pilot/1' or plan.get('point') != 'XL41-gemv'
            or plan.get('point_sha256') != '4fc3f12b79ff75fc5cd70751b301317866fa38c9a82300d3d09fe7a6bc9d0b31'
            or plan.get('timeout_seconds') != 3600 or plan.get('image_sha256') != prior.IMAGE_SHA
            or plan.get('image_uuid') != prior.IMAGE_UUID or plan.get('clock_hz') != 100_000_000
            or set(plan.get('inputs', {})) != REQUIRED_INPUTS):
        raise ValueError('invalid XL pilot registration')
    plan_sha = auto.sha(registration)
    if (auto.sha(pilot) != plan['pilot_script_sha256'] or auto.sha(guard) != plan['board_guard_sha256']
            or auto.sha(q41_registration) != plan['q41_registration_sha256']
            or auto.sha(cpu_evidence) != plan['cpu_pilot_evidence_sha256']):
        raise ValueError('pilot source, predecessor registration or CPU evidence differs')
    for name, digest in plan['inputs'].items():
        if auto.sha(root/name) != digest:
            raise ValueError('frozen pilot input changed: '+name)
    q41_plan = json.loads(q41_registration.read_text())
    if out.exists():
        raise ValueError('XL pilot queue output already exists')
    out.mkdir(parents=True)
    with (work/'xl41_pilot_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kwargs):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kwargs))+'\n')
        record('waiting-q41', queue=str(queue), pid=a.queue_pid, registration_sha256=plan_sha)
        while True:
            events = prior.read_events(queue/'queue.jsonl')
            if events and events[-1].get('kind') == 'completed':
                verify_q41(queue, events, q41_plan, auto)
                record('q41-verified')
                break
            if not queue_alive(a.queue_pid, queue):
                record('q41-stopped')
                raise RuntimeError('Q41 pilot stopped before completion')
            time.sleep(a.poll)
        previous = None
        attempt = 0
        while True:
            if (auto.sha(registration) != plan_sha or auto.sha(pilot) != plan['pilot_script_sha256']
                    or auto.sha(guard) != plan['board_guard_sha256']):
                raise ValueError('XL pilot registration or source changed while waiting')
            try:
                row = auto.probe(a.sysfs)
                ready = auto.idle(row) and row['uuid'] == prior.IMAGE_UUID and auto.clock_mhz(a.sysfs) == 100
                with (work/'candidate_queue.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        ready = False
            except (OSError, ValueError):
                ready, row = False, None
            if not ready:
                if previous != 'waiting-card':
                    record('waiting-card', board=row)
                    previous = 'waiting-card'
                time.sleep(a.poll)
                continue
            attempt += 1
            target = out/f'attempt-{attempt:03d}'
            env = auto.board_environment()
            env.update(TNCA_BACKEND='fpga-u280-multicore', TNCA_PROFILE='core-vrf-sync-v5-x8-g2c4-dp',
                       TNCA_BACKEND_ARGS=json.dumps(dict(xclbin=str(work/'tnca_u280.xclbin'),
                           image_profile='core-vrf-sync-v5-x8-g2c4-dp', core_count=3, memory_ports=3,
                           hbm_dma=True, hbm_bytes=8*(1 << 30), device_clock=True)),
                       TNCA_OPERATING_CLOCK_EVIDENCE=str(work/'qualified_clock.json'),
                       TNCA_DEADLINE='200000000000', TNCA_HOST_TIMEOUT='3600',
                       PYTHONPATH=str(inputs/'transformers/src')+':'+str(root/'tools/models'))
            cmd = [str(a.python), str(pilot), '--root', str(root), '--out', str(target), '--device', 'tnca']
            record('pilot-start', attempt=attempt, board=row)
            try:
                with (out/f'attempt-{attempt:03d}.log').open('x') as stream:
                    run = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=plan['timeout_seconds'])
            except subprocess.TimeoutExpired:
                record('pilot-timeout', attempt=attempt)
                raise RuntimeError('XL pilot exceeded its registered one-hour limit')
            report_path = target/'report.json'
            summary = json.loads(report_path.read_text()) if report_path.is_file() else {}
            record('pilot-end', attempt=attempt, returncode=run.returncode, status=summary.get('status'),
                   report_sha256=auto.sha(report_path) if report_path.is_file() else None)
            error = summary.get('error', '')
            if run.returncode and (error.startswith('RuntimeError: card busy or candidate image/clock changed')
                                   or error.startswith('BlockingIOError:')):
                previous = None
                time.sleep(a.poll)
                continue
            if (run.returncode or summary.get('status') != 'pass' or summary.get('point') != plan['point']
                    or summary.get('point_sha256') != plan['point_sha256']):
                raise RuntimeError('XL pilot failed; preserve its output for diagnosis')
            record('completed', scope='bounded XL41 GEMV pilot only')
            break


if __name__ == '__main__':
    main()
