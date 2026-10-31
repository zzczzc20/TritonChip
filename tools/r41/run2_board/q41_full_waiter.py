#!/usr/bin/env python3
"""Continue Q41 core/tail after verified preregistered pilots, without loading."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import q41_pilot_waiter as q41
import xl41_pilot_waiter as xl


def verify_stream(queue, events, plan, auto):
    if not events or events[-1].get('kind') != 'completed':
        raise ValueError('stream pilot queue is not completed')
    ends = [e for e in events if e.get('kind') == 'pilot-end' and e.get('returncode') == 0]
    if len(ends) != 1 or ends[0].get('status') != 'pass':
        raise ValueError('stream pilot did not pass exactly once')
    end = ends[0]
    path = queue/f"attempt-{end['attempt']:03d}"/'report.json'
    if auto.sha(path) != end.get('report_sha256'):
        raise ValueError('stream report changed')
    report = json.loads(path.read_text())
    checks, board, ranges = report.get('checks', {}), report.get('board', {}), report.get('device_ranges', {})
    if (report.get('status') != 'pass' or report.get('device') != 'tnca'
            or report.get('script_sha256') != plan['pilot_script_sha256']
            or report.get('point') != 'XL41-stream' or report.get('point_sha256') != plan['point_sha256']
            or board.get('image_sha256') != q41.IMAGE_SHA or board.get('uuid') != q41.IMAGE_UUID
            or board.get('clock_hz') != 100_000_000 or checks.get('finite') is not True
            or checks.get('mismatches') != 0 or checks.get('checked_elements') != 1073741888
            or ranges.get('input_bytes') != 2147483776 or ranges.get('output_bytes') != 2147483776):
        raise ValueError('stream report identity or checks differ')
    return True


def budget_for(steps, pilot_report, policy):
    rows = pilot_report['steps']
    if len(rows) != 2:
        raise ValueError('pilot timing requires two steps')
    prefill, decode = (float(row['wall_seconds']) for row in rows)
    if not all(math.isfinite(x) and x > 0 for x in (prefill, decode)):
        raise ValueError('pilot timing is missing or invalid')
    return max(policy['minimum_seconds'], math.ceil(policy['fixed_seconds'] +
               policy['prefill_multiplier']*prefill + policy['decode_multiplier']*decode*(steps-1)))


def queue_alive(pid, queue):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return 'xl41_stream_waiter.py' in command and str(queue) in command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'stream-queue', 'q41-queue', 'pilot', 'registration',
                 'q41-registration', 'stream-registration', 'inputs', 'cpu-evidence', 'out', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stream-pid', type=int, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.poll <= 60:
        raise ValueError('poll must be within 1..60 seconds')
    root, work, stream_queue, q41_queue, pilot, registration, q41_registration, stream_registration, inputs, cpu_evidence, out = (
        x.resolve() for x in (a.root, a.work, a.stream_queue, a.q41_queue, a.pilot, a.registration,
                              a.q41_registration, a.stream_registration, a.inputs, a.cpu_evidence, a.out))
    sys.path.insert(0, str(root/'tools/r41/run2_board'))
    import auto
    plan = json.loads(registration.read_text())
    stages = [dict(id='core', point='Q41-core', steps=17), dict(id='tail', point='Q41-tail', steps=5),
              dict(id='independent-core', point='Q41-core', steps=17)]
    policy = dict(prefill_multiplier=2, decode_multiplier=2, fixed_seconds=1200, minimum_seconds=1800,
                  source='successful board pilot step wall times including state comparisons; bound calculated before each stage')
    if (plan.get('schema') != 'tnca-r41-q41-bf16-full-queue/1' or plan.get('stages') != stages
            or plan.get('budget_rule') != policy or plan.get('image_sha256') != q41.IMAGE_SHA
            or plan.get('image_uuid') != q41.IMAGE_UUID or plan.get('clock_hz') != 100_000_000
            or set(plan.get('inputs', {})) != q41.REQUIRED_INPUTS):
        raise ValueError('invalid Q41 full-run registration')
    plan_sha = auto.sha(registration)
    if (auto.sha(pilot) != plan['pilot_script_sha256']
            or auto.sha(q41_registration) != plan['q41_registration_sha256']
            or auto.sha(stream_registration) != plan['stream_registration_sha256']
            or auto.sha(cpu_evidence) != plan['cpu_full_preflight_sha256']):
        raise ValueError('pilot, CPU evidence or predecessor registration differs')
    if set(plan.get('waiter_dependencies', {})) != {'q41_pilot_waiter.py', 'xl41_pilot_waiter.py'}:
        raise ValueError('waiter dependencies differ')
    for name, digest in plan['waiter_dependencies'].items():
        if auto.sha(Path(__file__).resolve().parent/name) != digest:
            raise ValueError('waiter dependency changed: '+name)
    for name, digest in plan['inputs'].items():
        if auto.sha(root/name) != digest:
            raise ValueError('frozen Q41 input changed: '+name)
    for name in ('verification.json', 'reference_verification.json', 'transformers_verification.json'):
        if json.loads((inputs/name).read_text()).get('verified') is not True:
            raise ValueError('staged input was not verified: '+name)
    if out.exists():
        raise ValueError('full Q41 queue output already exists')
    out.mkdir(parents=True)
    with (work/'q41_full_waiter.lock').open('a') as exclusive, (out/'queue.jsonl').open('x', buffering=1) as log:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(kind, **kwargs):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **kwargs))+'\n')
        record('waiting-stream', queue=str(stream_queue), pid=a.stream_pid, registration_sha256=plan_sha)
        while True:
            events = q41.read_events(stream_queue/'queue.jsonl')
            if events and events[-1].get('kind') == 'completed':
                verify_stream(stream_queue, events, json.loads(stream_registration.read_text()), auto)
                qevents = q41.read_events(q41_queue/'queue.jsonl')
                qplan = json.loads(q41_registration.read_text())
                xl.verify_q41(q41_queue, qevents, qplan, auto)
                successful = next(e for e in qevents if e.get('kind') == 'pilot-end' and e.get('returncode') == 0)
                pilot_report_path = q41_queue/f"attempt-{successful['attempt']:03d}"/'report.json'
                pilot_report = json.loads(pilot_report_path.read_text())
                record('predecessors-verified', q41_report_sha256=auto.sha(pilot_report_path))
                break
            if not queue_alive(a.stream_pid, stream_queue):
                record('stream-stopped')
                raise RuntimeError('stream pilot stopped before completion')
            time.sleep(a.poll)
        for stage in stages:
            budget = budget_for(stage['steps'], pilot_report, policy)
            record('stage-budget', stage=stage['id'], seconds=budget, policy=policy,
                   pilot_step_seconds=[r['wall_seconds'] for r in pilot_report['steps']])
            attempt, previous = 0, None
            while True:
                if (auto.sha(registration) != plan_sha or auto.sha(pilot) != plan['pilot_script_sha256']
                        or auto.sha(q41_registration) != plan['q41_registration_sha256']
                        or auto.sha(stream_registration) != plan['stream_registration_sha256']):
                    raise ValueError('Q41 registration or source changed while waiting')
                for name, digest in plan['waiter_dependencies'].items():
                    if auto.sha(Path(__file__).resolve().parent/name) != digest:
                        raise ValueError('waiter dependency changed: '+name)
                for name, digest in plan['inputs'].items():
                    if auto.sha(root/name) != digest:
                        raise ValueError('frozen Q41 input changed: '+name)
                try:
                    row = auto.probe(a.sysfs)
                    ready = auto.idle(row) and row['uuid'] == q41.IMAGE_UUID and auto.clock_mhz(a.sysfs) == 100
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
                cmd = [str(a.python), str(pilot), '--root', str(root), '--snapshot', str(inputs/'snapshot'),
                       '--reference', str(inputs/'fp32_reference'), '--out', str(target), '--device', 'tnca',
                       '--point', stage['point'], '--steps', str(stage['steps'])]
                record('stage-start', stage=stage['id'], attempt=attempt, board=row)
                try:
                    with (out/f"{stage['id']}-attempt-{attempt:03d}.log").open('x') as stream:
                        run = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=budget)
                except subprocess.TimeoutExpired:
                    record('stage-timeout', stage=stage['id'], attempt=attempt)
                    raise RuntimeError('Q41 stage exceeded its registered pilot-derived budget')
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
                if (run.returncode or summary.get('status') != 'pass' or summary.get('device') != 'tnca'
                        or summary.get('point') != stage['point'] or len(summary.get('steps', [])) != stage['steps']
                        or not all(x.get('passed') is True for x in summary['steps'])):
                    raise RuntimeError('Q41 stage failed; preserve output for diagnosis')
                break
        record('completed', scope='core, tail and separate core process; not final acceptance')


if __name__ == '__main__':
    main()
