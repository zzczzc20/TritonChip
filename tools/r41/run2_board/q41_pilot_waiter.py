#!/usr/bin/env python3
"""Wait for the completed candidate campaign, then run the registered Q41 pilot.

Never loads an image. A failed upstream stage or a pilot execution failure
stops the waiter. An occupied card or lock is waited out without intervention.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_events(path):
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith('\n'):
        lines.pop()
    return [json.loads(line) for line in lines]


def verify_upstream(root, queue, events, auto):
    if not events or events[-1].get('kind') != 'completed':
        raise ValueError('upstream queue is not completed')
    done = [e for e in events if e.get('kind') in ('stage-end', 'stage-reused')]
    if [e.get('stage') for e in done] != list(auto.REQUIRED):
        raise ValueError('upstream stage inventory differs')
    for event in done:
        stage = event['stage']
        reg = json.loads((root/'spec/r41/run3/stages'/f'{stage}.json').read_text())
        if reg.get('id') != stage or reg.get('image_identity', {}).get('sha256') != IMAGE_SHA:
            raise ValueError('upstream registration identity differs: '+stage)
        if event['kind'] == 'stage-reused':
            summary = Path(event['prior'])
            if auto.sha(summary) != event.get('sha256'):
                raise ValueError('reused upstream summary changed: '+stage)
        else:
            if event.get('returncode') != 0:
                raise ValueError('upstream stage failed: '+stage)
            summary = queue/'results'/stage/'stage.json'
        result = json.loads(summary.read_text())
        rows, prior = reg['experiments'], result.get('experiments', [])
        if result.get('status') != 'pass' or len(rows) != len(prior):
            raise ValueError('upstream stage summary incomplete: '+stage)
        for row, got in zip(rows, prior):
            report = summary.parent/row['id']/'report.json'
            if (got.get('id') != row['id'] or got.get('manifest_sha256') != row['sha256']
                    or auto.sha(root/row['manifest']) != row['sha256']
                    or got.get('report_sha256') != auto.sha(report)
                    or len(got.get('checks', [])) != len(row['checks'])
                    or any(auto.sha(Path(c['path'])) != c['sha256'] for c in got['checks'])):
                raise ValueError('upstream stage evidence changed: '+stage)
    return True


IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
IMAGE_UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
REQUIRED_INPUTS = {
    'spec/r41/stage2/model_lock.json', 'spec/r41/stage2/model_points.json',
    'evidence/r41/stage1/kickoff/qwen-storage/inventory.json',
    'evidence/r41/stage1/kickoff/transformers-reference/sources.json',
    'evidence/r41/acceptance/p0/cross-validation.json',
    'tools/models/qwen35_reference.py', 'tools/models/qwen_fp8_lib.py',
    'tools/models/flux2_klein_run.py', 'tools/models/fp8_blockwise_ref.py',
    'tools/models/fp8_checkpoint.py', 'tools/r41/run2_board/auto.py',
}


def queue_alive(pid, queue):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'candidate_resume.py' in command and str(queue) in command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'queue', 'pilot', 'registration', 'inputs', 'cpu-evidence', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be within 1..60 seconds')
    root, work, queue, pilot, registration, inputs, cpu_evidence, out = (
        x.resolve() for x in (a.root, a.work, a.queue, a.pilot, a.registration, a.inputs, a.cpu_evidence, a.out))
    sys.path.insert(0, str(root/'tools/r41/run2_board'))
    import auto
    plan = json.loads(registration.read_text())
    if (plan.get('schema') != 'tnca-r41-q41-bf16-pilot/1' or plan.get('point') != 'Q41-core'
            or plan.get('steps') != 2 or plan.get('timeout_seconds') != 3600
            or plan.get('image_sha256') != IMAGE_SHA or plan.get('image_uuid') != IMAGE_UUID
            or plan.get('clock_hz') != 100_000_000 or set(plan.get('inputs', {})) != REQUIRED_INPUTS):
        raise ValueError('invalid pilot registration')
    plan_sha = auto.sha(registration)
    if auto.sha(pilot) != plan['pilot_script_sha256'] or auto.sha(cpu_evidence) != plan['cpu_pilot_evidence_sha256']:
        raise ValueError('pilot or CPU preflight evidence differs')
    for name, digest in plan['inputs'].items():
        if auto.sha(root/name) != digest:
            raise ValueError('frozen pilot input changed: '+name)
    for name in ('verification.json', 'reference_verification.json', 'transformers_verification.json'):
        if json.loads((inputs/name).read_text()).get('verified') is not True:
            raise ValueError('staged input was not verified: '+name)
    if out.exists():
        raise ValueError('pilot queue output already exists')
    out.mkdir(parents=True)
    with (work/'q41_pilot_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kwargs):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kwargs))+'\n')
        record('waiting-upstream', queue=str(queue), pid=a.queue_pid, registration_sha256=plan_sha)
        while True:
            events = read_events(queue/'queue.jsonl')
            if events and events[-1].get('kind') == 'completed':
                verify_upstream(root, queue, events, auto)
                record('upstream-verified')
                break
            failed = [e for e in events if e.get('kind') == 'stage-end' and e.get('returncode') != 0]
            if failed or not queue_alive(a.queue_pid, queue):
                record('upstream-stopped', failure=failed[-1] if failed else None)
                raise RuntimeError('upstream campaign stopped before completion')
            time.sleep(a.poll)
        previous = None
        attempt = 0
        while True:
            if auto.sha(registration) != plan_sha or auto.sha(pilot) != plan['pilot_script_sha256']:
                raise ValueError('pilot registration or source changed while waiting')
            try:
                row = auto.probe(a.sysfs)
                ready = auto.idle(row) and row['uuid'] == IMAGE_UUID and auto.clock_mhz(a.sysfs) == 100
                with (work/'candidate_queue.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        ready = False
            except (OSError, ValueError):
                ready = False
                row = None
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
            cmd = [str(a.python), str(pilot), '--root', str(root), '--snapshot', str(inputs/'snapshot'),
                   '--reference', str(inputs/'fp32_reference'), '--out', str(target), '--device', 'tnca',
                   '--point', plan['point'], '--steps', str(plan['steps'])]
            record('pilot-start', attempt=attempt, board=row)
            try:
                with (out/f'attempt-{attempt:03d}.log').open('x') as stream:
                    run = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=plan['timeout_seconds'])
            except subprocess.TimeoutExpired:
                record('pilot-timeout', attempt=attempt)
                raise RuntimeError('pilot exceeded its registered one-hour limit')
            summary = json.loads((target/'report.json').read_text()) if (target/'report.json').is_file() else {}
            record('pilot-end', attempt=attempt, returncode=run.returncode, status=summary.get('status'),
                   report_sha256=auto.sha(target/'report.json') if (target/'report.json').is_file() else None)
            error = summary.get('error', '')
            if run.returncode and (error.startswith('RuntimeError: card busy or candidate image/clock changed')
                                   or error.startswith('BlockingIOError:')):
                previous = None
                time.sleep(a.poll)
                continue
            if run.returncode or summary.get('status') != 'pass' or len(summary.get('steps', [])) != plan['steps']:
                raise RuntimeError('Q41 pilot failed; preserve its output for diagnosis')
            record('completed', scope='bounded Q41 pilot only')
            break


if __name__ == '__main__':
    main()
