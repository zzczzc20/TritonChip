#!/usr/bin/env python3
"""Run bounded gather, attention and recurrence pilots after verified XL matrix tests."""
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
EXPECTED = [('gather', 'XL41-gather', '2ba3f7430464280044813353d099bc1fe3f840310ea2df5ce9c0d76ac4d0adbe'),
            ('attention', 'XL41-attention', '750f069b5f2a36eaa715164d16e05416a5f877c5991778bab03d0cb89b649c3b'),
            ('recurrence', 'XL41-recurrence', 'dbbf36b4b43b8f503a0fed13c3c3e2be2ed50e15657762f4c63eb4344e3fda00')]
COUNTS = {'transpose': 268435456, 'gemm': 3145728, 'projection': 248320}


def verify_matrix(queue, events, plan, auto):
    if not events or events[-1].get('kind') != 'completed':
        raise ValueError('XL matrix queue is not completed')
    ends = [x for x in events if x.get('kind') == 'stage-end' and x.get('returncode') == 0]
    if [x.get('stage') for x in ends] != [x['id'] for x in plan['stages']]:
        raise ValueError('XL matrix stage inventory differs')
    for stage, end in zip(plan['stages'], ends):
        if end.get('status') != 'pass':
            raise ValueError('XL matrix stage did not pass')
        path = queue/f"{stage['id']}-attempt-{end['attempt']:03d}"/'report.json'
        if auto.sha(path) != end.get('report_sha256'):
            raise ValueError('XL matrix report changed')
        r = json.loads(path.read_text())
        checks = r.get('checks', {})
        if (r.get('status') != 'pass' or r.get('device') != 'tnca'
                or r.get('script_sha256') != plan['pilot_script_sha256']
                or r.get('point') != stage['point'] or r.get('point_sha256') != stage['point_sha256']
                or r.get('board', {}).get('image_sha256') != common.IMAGE_SHA
                or r.get('board', {}).get('uuid') != common.IMAGE_UUID
                or r.get('board', {}).get('clock_hz') != 100_000_000
                or checks.get('finite') is not True or checks.get('mismatches') != 0
                or checks.get('checked_elements') != COUNTS[stage['id']]):
            raise ValueError('XL matrix report identity or checks differ')
    return True


def queue_alive(pid, queue):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'xl41_matrix_waiter.py' in command and str(queue) in command


def valid_extra(stage, report):
    if stage == 'gather':
        rows = report.get('runs', [])
        return (report.get('formal_single_vector_unresolved') is True
                and report.get('formula_missing_required_rows') == [0, 32767, 65535]
                and [x.get('name') for x in rows] == ['formula', 'boundary']
                and [x.get('count') for x in rows] == [8193, 4]
                and all(x.get('passed') is True for x in rows))
    if stage == 'attention':
        return report.get('tail_zero') is True and report.get('checks', {}).get('mismatches') == 0
    if stage == 'recurrence':
        rows = report.get('updates', [])
        return (report.get('formal_update_rule_unresolved') is True and len(rows) == 4
                and all(x.get('passed') is True and x.get('chunks') == [32, 32, 1] for x in rows))
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'queue', 'pilot-dir', 'guard', 'matrix-helper', 'registration',
                 'matrix-registration', 'inputs', 'cpu-evidence-dir', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be within 1..60 seconds')
    root, work, queue, pilot_dir, guard, matrix_helper, registration, matrix_registration, inputs, cpu_dir, out = (
        x.resolve() for x in (a.root, a.work, a.queue, a.pilot_dir, a.guard, a.matrix_helper, a.registration,
                              a.matrix_registration, a.inputs, a.cpu_evidence_dir, a.out))
    sys.path.insert(0, str(root/'tools/r41/run2_board'))
    import auto
    plan = json.loads(registration.read_text())
    stages = plan.get('stages', [])
    if (plan.get('schema') != 'tnca-r41-xl41-extra-queue/1'
            or [(x.get('id'), x.get('point'), x.get('point_sha256')) for x in stages] != EXPECTED
            or plan.get('timeout_seconds') != 3600 or plan.get('image_sha256') != common.IMAGE_SHA
            or plan.get('image_uuid') != common.IMAGE_UUID or plan.get('clock_hz') != 100_000_000
            or set(plan.get('inputs', {})) != REQUIRED_INPUTS):
        raise ValueError('invalid XL extra registration')
    plan_sha = auto.sha(registration)
    dependency = Path(__file__).resolve().parent/'q41_pilot_waiter.py'
    if (auto.sha(guard) != plan['board_guard_sha256'] or auto.sha(matrix_helper) != plan['matrix_helper_sha256']
            or auto.sha(matrix_registration) != plan['matrix_registration_sha256']
            or auto.sha(dependency) != plan['waiter_dependency_sha256']):
        raise ValueError('guard, helper or predecessor registration differs')
    for stage in stages:
        if (auto.sha(pilot_dir/f"xl41_{stage['id']}_pilot.py") != stage['pilot_script_sha256']
                or auto.sha(cpu_dir/f"xl41_{stage['id']}_cpu_preflight.tgz") != stage['cpu_preflight_sha256']):
            raise ValueError('pilot or CPU evidence differs: '+stage['id'])
    for name, digest in plan['inputs'].items():
        if auto.sha(root/name) != digest:
            raise ValueError('frozen XL input changed: '+name)
    matrix_plan = json.loads(matrix_registration.read_text())
    if out.exists():
        raise ValueError('XL extra queue output already exists')
    out.mkdir(parents=True)
    with (work/'xl41_extra_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kwargs):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kwargs))+'\n')
        record('waiting-matrix', queue=str(queue), pid=a.queue_pid, registration_sha256=plan_sha)
        while True:
            events = common.read_events(queue/'queue.jsonl')
            if events and events[-1].get('kind') == 'completed':
                verify_matrix(queue, events, matrix_plan, auto)
                record('matrix-verified')
                break
            if not queue_alive(a.queue_pid, queue):
                record('matrix-stopped')
                raise RuntimeError('XL matrix queue stopped before completion')
            time.sleep(a.poll)
        for stage in stages:
            pilot = pilot_dir/f"xl41_{stage['id']}_pilot.py"
            attempt, previous = 0, None
            while True:
                if (auto.sha(registration) != plan_sha or auto.sha(pilot) != stage['pilot_script_sha256']
                        or auto.sha(guard) != plan['board_guard_sha256'] or auto.sha(matrix_helper) != plan['matrix_helper_sha256']
                        or auto.sha(matrix_registration) != plan['matrix_registration_sha256']
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
                cmd = [str(a.python), str(pilot), '--root', str(root), '--out', str(target), '--device', 'tnca']
                record('stage-start', stage=stage['id'], attempt=attempt, board=row)
                try:
                    with (out/f"{stage['id']}-attempt-{attempt:03d}.log").open('x') as stream:
                        run = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=plan['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    record('stage-timeout', stage=stage['id'], attempt=attempt)
                    raise RuntimeError('XL pilot exceeded its registered one-hour limit')
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
                        or summary.get('point_sha256') != stage['point_sha256'] or not valid_extra(stage['id'], summary)):
                    raise RuntimeError('XL pilot failed; preserve output for diagnosis')
                break
        record('completed', scope='bounded XL pilots; unresolved formal details remain')


if __name__ == '__main__':
    main()
