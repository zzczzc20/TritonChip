#!/usr/bin/env python3
"""Run a frozen, pre-registered two-model campaign without programming the card."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from consolidated_plan import IMAGE_SHA, IMAGE_UUID, MODEL_BUDGET_SECONDS, tree_digest


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def member(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('campaign path escapes its root')
    return root / path


def verify_inputs(root, overlay, plan):
    if plan.get('schema') != 'r41-consolidated-campaign/1' or plan.get('identity_checks') != 1 or \
            [item.get('model') for item in plan.get('models', [])] != ['vl', 'q35']:
        raise ValueError('campaign is not the registered two-model plan')
    if plan.get('status') != 'frozen' or not re.fullmatch(r'[0-9a-f]{40}', plan.get('frozen_commit') or ''):
        raise ValueError('campaign plan is provisional, not frozen')
    if (plan.get('image_sha256'), plan.get('image_uuid'), plan.get('clock_hz'),
            plan.get('not_before_utc'), plan.get('campaign_deadline_utc')) != \
            (IMAGE_SHA, IMAGE_UUID, 100000000, '2026-11-01T17:50:00Z', '2026-11-02T00:50:00Z'):
        raise ValueError('campaign image, clock or time window differs from the directive')
    if tree_digest(root) != plan.get('source_tree_sha256'):
        raise ValueError('frozen compiler, runtime or harness source changed')
    for path, expected in plan['source_sha256'].items():
        if sha(member(root, path)) != expected:
            raise ValueError(f'frozen source changed: {path}')
    for item in plan['models']:
        for key in ('manifest', 'reference'):
            if sha(member(root, item[key])) != item[key + '_sha256']:
                raise ValueError(f"{item['model']}: changed {key}")
        manifest = json.loads(member(root, item['manifest']).read_text())
        steps, board = manifest.get('steps') or [], manifest.get('board') or {}
        if len(steps) != 1 or steps[0].get('kind') != 'fpga' or \
                manifest.get('budget_seconds') != MODEL_BUDGET_SECONDS[item['model']] or \
                steps[0].get('timeout_seconds') != MODEL_BUDGET_SECONDS[item['model']] or \
                (board.get('image_sha256'), board.get('image_uuid'), board.get('clock_hz')) != \
                (IMAGE_SHA, IMAGE_UUID, 100000000):
            raise ValueError(f"{item['model']}: invalid single-session manifest")
        argv = steps[0].get('argv') or []
        if '{root}/tools/models/chat_quality.py' not in argv or '--warm-repeat' not in argv or \
                '--script' not in argv or '--script-checks' not in argv or \
                argv.count('--model') != 1 or argv[argv.index('--model') + 1] != item['model'] or \
                argv.count('--device') != 1 or argv[argv.index('--device') + 1] != 'tnca':
            raise ValueError(f"{item['model']}: manifest is not the registered chat session")
        required = {'--spec': '{root}/spec/r41/chat/quality-author-20261101.json',
                    '--reference': '{root}/' + item['reference'],
                    '--script': '{root}/spec/r41/chat/script-text.txt',
                    '--script-checks': '{root}/spec/r41/chat/script-author-20261101.json'}
        if any(argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value for flag, value in required.items()) or \
                any(flag in argv for flag in ('--prompts', '--layers', '--max-new-tokens')):
            raise ValueError(f"{item['model']}: quality, reference or script is not complete")
        registration = member(overlay, item['registration'])
        if sha(registration) != item['registration_sha256']:
            raise ValueError(f"{item['model']}: changed registration")
        reg = json.loads(registration.read_text())
        if len(reg.get('experiments', [])) != 1 or reg['experiments'][0].get('manifest') != item['manifest'] or \
                reg['experiments'][0].get('sha256') != item['manifest_sha256'] or \
                reg.get('max_experiment_seconds') != MODEL_BUDGET_SECONDS[item['model']] or \
                reg['experiments'][0].get('budget_seconds') != MODEL_BUDGET_SECONDS[item['model']] or \
                reg.get('image_identity') != dict(sha256=plan['image_sha256'], uuid=plan['image_uuid']):
            raise ValueError(f"{item['model']}: registration does not match plan")
        checks = reg['experiments'][0].get('checks') or []
        path = f"{item['model']}-quality/quality.json"
        count = 6 if item['model'] == 'vl' else 5
        required_checks = [dict(path=path, field=['status'], equals='pass'),
                           dict(path=path, field=['model_scope'], equals='full-checkpoint'),
                           dict(path=path, field=['reduced_proof'], equals=False),
                           dict(path=path, field=['prompts'], length_equals=count),
                           dict(path=path, field=['warm_repeat', 'passed'], equals=True),
                           dict(path=path, field=['scripted_chat', 'passed'], equals=True)]
        # JSON comparison preserves boolean versus integer distinctions.
        serialized = {json.dumps(check, sort_keys=True) for check in checks}
        if any(json.dumps(check, sort_keys=True) not in serialized for check in required_checks):
            raise ValueError(f"{item['model']}: required session checks are missing or changed")


def effective_registration(source, target, remaining):
    budget = int(remaining - 30)
    if budget < 1:
        return None
    reg = json.loads(source.read_text())
    old = reg['experiments'][0]['budget_seconds']
    reg['experiments'][0]['budget_seconds'] = min(old, budget)
    with target.open('x') as stream:
        json.dump(reg, stream, indent=2)
        stream.write('\n')
    return target


def utc(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def wait_idle(auto, sysfs, work, lock, deadline, expected_uuid, clock_hz, *, monitor=True, on_state=None):
    tracker = auto.IdleTracker(max_gap=15)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    last_state = None
    def report(state, row=None):
        nonlocal last_state
        if state != last_state and on_state is not None:
            on_state(state, row)
        last_state = state
    while now() < deadline:
        try:
            row = auto.probe(sysfs)
            stamp = time.monotonic()
            ready = tracker.observe(stamp, row)
            if monitor:
                try:
                    saved = json.loads((work / 'idle_monitor.json').read_text())
                    ready |= (saved['boot_id'] == boot and saved['identity'] == [row['uuid'], row['total execs']]
                              and saved['start'] is not None and saved['last'] is not None
                              and 0 <= stamp - saved['last'] <= 15 and stamp - saved['start'] >= 900
                              and auto.idle(row))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if row['uuid'] != expected_uuid:
                report('other-image', row)
            elif not auto.idle(row):
                report('busy', row)
            elif auto.clock_mhz(sysfs) * 1000000 != clock_hz:
                report('clock-mismatch', row)
            elif not ready:
                report('qualifying-idle', row)
            else:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    report('queue-locked', row)
                    time.sleep(10)
                    continue
                current = auto.probe(sysfs)
                if current == row and auto.idle(current) and auto.clock_mhz(sysfs) * 1000000 == clock_hz:
                    report('acquired', current)
                    return current
                fcntl.flock(lock, fcntl.LOCK_UN)
                report('changed-during-acquisition', current)
        except (OSError, ValueError):
            tracker = auto.IdleTracker(max_gap=15)
            report('probe-error')
        time.sleep(10)
    return None


def run(root, overlay, output, plan):
    root, overlay, output = map(Path, (root, overlay, output))
    verify_inputs(root, overlay, plan)
    if now() < utc(plan['not_before_utc']):
        raise ValueError('campaign cannot start before the author freeze time')
    deadline = utc(plan['campaign_deadline_utc'])
    if now() >= deadline:
        raise ValueError('campaign window has ended')
    output.mkdir(parents=True, exist_ok=False)
    def record(event, **fields):
        with (output / 'queue.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=now().isoformat(), event=event, **fields)) + '\n')
    sys.path.insert(0, str(root / 'tools/r41/run2_board'))
    import auto
    work = Path.home() / 'r41run3'
    sysfs = Path('/sys/bus/pci/devices')
    py = Path.home() / 'tc_env/torchenv/bin/python'
    record('waiting-card', image_uuid=plan['image_uuid'])
    def board_state(state, board):
        record('card-state', state=state, board=board)
        print('card-state:', state, flush=True)
    with (work / 'candidate_queue.lock').open('a') as lock:
        row = wait_idle(auto, sysfs, work, lock, deadline, plan['image_uuid'], plan['clock_hz'],
                        on_state=board_state)
        if row is None:
            record('deadline-waiting-card')
            return
        if sha(work / 'tnca_u280.xclbin') != plan['image_sha256']:
            raise ValueError('approved image file changed')
        record('identity-clock', board=row, clock_hz=plan['clock_hz'], image_sha256=plan['image_sha256'])
        for item in plan['models']:
            model = item['model']
            current = auto.probe(sysfs)
            if not auto.idle(current) or current['uuid'] != plan['image_uuid'] or \
                    auto.clock_mhz(sysfs) * 1000000 != plan['clock_hz']:
                fcntl.flock(lock, fcntl.LOCK_UN)
                record('waiting-card-handoff', model=model, board=current)
                current = wait_idle(auto, sysfs, work, lock, deadline, plan['image_uuid'], plan['clock_hz'],
                                    on_state=board_state)
                if current is None:
                    record('deadline-waiting-card', model=model)
                    break
            remaining = (deadline - now()).total_seconds()
            registration = effective_registration(member(overlay, item['registration']),
                                                  output / f'{model}-effective-registration.json', remaining)
            if registration is None:
                record('deadline-before-model', model=model)
                break
            verify_inputs(root, overlay, plan)
            record('model-start', model=model, board=current, registration_sha256=sha(registration))
            command = [str(py), str(root / 'tools/r41/run2_board/stage.py'), '--root', str(root),
                       '--out', str(output / model), '--python', str(py), '--registration', str(registration)]
            with (output / f'{model}.log').open('x') as stream:
                result = subprocess.run(command, cwd=root, env=auto.board_environment(),
                                        stdout=stream, stderr=subprocess.STDOUT)
            summary = output / model / 'stage.json'
            record('model-end', model=model, returncode=result.returncode,
                   stage_sha256=sha(summary) if summary.is_file() else None)
        record('finished', scope='one attempt per model; inspect failures before one targeted retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'overlay', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.overlay, args.output, json.loads((args.overlay / 'plan.json').read_text()))


if __name__ == '__main__':
    main()
