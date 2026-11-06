#!/usr/bin/env python3
"""Read-only worktree sensor and narrowly targeted Codex terminal adapter."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
import uuid

try:
    from .activity import Novelty, code_lines, fingerprint, watched_code
    from .watchdog import DEFAULTS, WatchClock
except ImportError:
    from activity import Novelty, code_lines, fingerprint, watched_code
    from watchdog import DEFAULTS, WatchClock

STATE = Path('/home/ubuntu/.guard')
COMMON_REPO = Path('/home/ubuntu/TritonChip')
THREAD_DB = Path('/home/ubuntu/.codex/thread_history_1.sqlite')
GOAL_DB = Path('/home/ubuntu/.codex/goals_1.sqlite')
SESSION_DB = Path('/home/ubuntu/.codex/state_5.sqlite')
NONCODE = ('evidence/', 'doc/', 'spec/')
CODE_CACHE = {}


def command(args, cwd=None, input=None):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    return subprocess.run(args, cwd=cwd, input=input, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env, check=True).stdout


def git(repo, *args):
    return command(['git', '-C', str(repo), *args])


def worktrees(prefix):
    rows = git(COMMON_REPO, 'worktree', 'list', '--porcelain').decode().splitlines()
    return [Path(row[9:]) for row in rows if row.startswith('worktree ')
            and row[9:].startswith(prefix)]


def file_digest(path):
    if not path.is_file() or path.is_symlink():
        return 'missing'
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def thread_worktrees(targets):
    result = []
    with closing(sqlite3.connect(f'file:{SESSION_DB}?mode=ro', uri=True, timeout=1)) as db:
        for target in targets:
            row = db.execute('SELECT cwd FROM threads WHERE id=?', (target['thread_id'],)).fetchone()
            if not row:
                raise RuntimeError('target thread has no recorded cwd')
            result.append(Path(git(Path(row[0]), 'rev-parse', '--show-toplevel').decode().strip()))
    return result


def sample(prefix, last_heads, extra_worktrees=()):
    lines, dirty, heads = set(), {}, {}
    committed_evidence = 0
    paths = list(dict.fromkeys(worktrees(prefix) + list(extra_worktrees)))
    if not paths:
        raise RuntimeError('no configured worktrees found')
    for root in paths:
        head = git(root, 'rev-parse', 'HEAD').decode().strip()
        heads[str(root)] = head
        old = last_heads.get(str(root))
        if old and old != head:
            changes = git(root, 'diff', '--numstat', '-z', '--no-renames', old, head)
            for row in changes.split(b'\0'):
                if row:
                    add, delete, path = row.split(b'\t', 2)
                    if os.fsdecode(path).startswith(NONCODE):
                        committed_evidence += sum(int(x) for x in (add, delete) if x.isdigit()) or 1
        tracked = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
        for raw in tracked.split(b'\0'):
            if not raw:
                continue
            name = os.fsdecode(raw)
            if watched_code(name):
                file = root / name
                if file.is_symlink() or not file.is_file():
                    continue
                if file.stat().st_size > 4_000_000:
                    raise RuntimeError(f'code file too large to inspect: {file}')
                body = file.read_bytes()
                key = (name == 'Makefile', Path(name).suffix.lower(), hashlib.sha256(body).digest())
                cached = CODE_CACHE.get(key)
                if cached is None:
                    cached = frozenset(map(fingerprint, code_lines(name, body)))
                    if len(CODE_CACHE) >= 30_000:
                        CODE_CACHE.clear()
                    CODE_CACHE[key] = cached
                lines.update(cached)
        changed = git(root, 'ls-files', '-z', '-m', '-d', '--others', '--exclude-standard', '--', *NONCODE)
        staged = git(root, 'diff', '--cached', '--name-only', '-z', '--no-renames', '--', *NONCODE)
        for raw in set(changed.split(b'\0') + staged.split(b'\0')):
            if raw:
                name = os.fsdecode(raw)
                dirty[(str(root), name)] = file_digest(root / name)
    return lines, dirty, heads, committed_evidence


def goal_status(thread_id):
    with closing(sqlite3.connect(f'file:{GOAL_DB}?mode=ro', uri=True, timeout=1)) as db:
        row = db.execute('SELECT status FROM thread_goals WHERE thread_id=?', (thread_id,)).fetchone()
    if not row:
        raise RuntimeError(f'goal not found for {thread_id}')
    return row[0]


def pane_for(target):
    # Resolve an exact session name and pin its pane and Codex thread identity.
    session, pane, tty = command(['tmux', 'display-message', '-p', '-t', '=' + target['tmux'] + ':',
                                  '#{session_name}\t#{pane_id}\t#{pane_tty}']).decode().strip().split('\t')
    if session != target['tmux']:
        raise RuntimeError('tmux resolved the wrong session')
    tty = tty.removeprefix('/dev/')
    processes = command(['ps', '-t', tty, '-o', 'args=']).decode(errors='replace').splitlines()
    expected = target['thread_id']
    if not any(re.search(r'\bcodex\s+resume\s+' + re.escape(expected) + r'(?:\s|$)', x)
               for x in processes):
        raise RuntimeError('target pane does not contain the expected Codex thread')
    return pane


def log(path, text):
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE / path).open('a') as out:
        out.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {text}\n')


def alert(text):
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    (STATE / 'ALERT.txt').write_text(text + '\n')


def screen(pane):
    return command(['tmux', 'capture-pane', '-p', '-t', pane]).decode(errors='replace')


def composer_empty(pane):
    rows = screen(pane).splitlines()
    prompts = [match.group(1).strip() for line in rows
               if (match := re.match(r'^\s*›\s*(.*)$', line))]
    return bool(prompts) and prompts[-1] in ('', 'Ask Codex to do anything')


def wait_idle(thread_id, seconds=10):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        with closing(sqlite3.connect(f'file:{THREAD_DB}?mode=ro', uri=True, timeout=1)) as db:
            row = db.execute('SELECT status FROM thread_turns WHERE thread_id=? '
                             'ORDER BY rollout_ordinal DESC LIMIT 1', (thread_id,)).fetchone()
        if not row or row[0] != 'inProgress':
            return True
        time.sleep(0.25)
    return False


def deliver(target, message, *, interrupt=False, queued=True, retries=3):
    """Only send keys to the verified pane. Confirm the marker on its screen."""
    pane = pane_for(target)
    marker = '守护' + uuid.uuid4().hex[:8]
    text = marker + ' ' + message
    if interrupt:
        command(['tmux', 'send-keys', '-t', pane, 'Escape'])
        if not wait_idle(target['thread_id']):
            log('warning.log', f'{target["tmux"]} {marker} interruption not confirmed')
            alert(f'watchdog could not confirm interruption of tmux {target["tmux"]}')
            return False
    for attempt in range(1, retries + 1):
        if marker not in screen(pane):
            if not composer_empty(pane):
                log('warning.log', f'{target["tmux"]} {marker} attempt={attempt} composer is not empty')
                time.sleep(0.4)
                continue
            command(['tmux', 'send-keys', '-t', pane, '-l', '--', text])
            time.sleep(0.25)
        if marker not in screen(pane):
            log('warning.log', f'{target["tmux"]} {marker} attempt={attempt} not visible in input')
            continue
        command(['tmux', 'send-keys', '-t', pane, 'Tab' if queued else 'Enter'])
        time.sleep(0.4)
        if marker in screen(pane):
            log('warning.log', f'{target["tmux"]} {marker} attempt={attempt} visible after delivery')
            return True
        log('warning.log', f'{target["tmux"]} {marker} attempt={attempt} not visible after delivery')
    alert(f'watchdog could not confirm delivery to tmux {target["tmux"]}: {marker}')
    return False


def pause(target, reason):
    pane = pane_for(target)
    command(['tmux', 'send-keys', '-t', pane, 'Escape'])
    confirmed = False
    if wait_idle(target['thread_id']) and composer_empty(pane):
        command(['tmux', 'send-keys', '-t', pane, '-l', '--', '/goal pause'])
        command(['tmux', 'send-keys', '-t', pane, 'Enter'])
        for _ in range(20):
            if goal_status(target['thread_id']) == 'paused':
                confirmed = True
                break
            time.sleep(0.25)
    suffix = 'goal paused' if confirmed else 'goal pause NOT confirmed'
    log('stop.log', f'{target["tmux"]} {reason}; {suffix}')
    alert(f'tmux {target["tmux"]}: {reason}; {suffix}')
    return confirmed


def find_assessment(thread_id, marker, after):
    with closing(sqlite3.connect(f'file:{THREAD_DB}?mode=ro', uri=True, timeout=1)) as db:
        rows = db.execute('SELECT item_json FROM thread_items WHERE thread_id=? AND item_type=? '
                          'AND created_at_ms>=? ORDER BY rollout_ordinal',
                          (thread_id, 'agentMessage', int(after * 1000))).fetchall()
    for (raw,) in rows:
        text = json.loads(raw).get('text', '')
        for line in text.splitlines():
            if marker in line and len(line.split(marker, 1)[1].strip(' ：:')) >= 8:
                return line.strip()
    return None


def code_digest(lines):
    return hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()


def run_validation(result, args, prefix='/home/ubuntu/TritonChip'):
    """Run a test/TLM command and record a new result bound to the code content."""
    if not args or not re.search(r'(?i)(test|pytest|tlm|compare)', ' '.join(args)):
        raise ValueError('validation command must identify a test or TLM comparison')
    result = Path(result).resolve()
    before = file_digest(result)
    config_path = STATE / 'targets.json'
    targets = json.loads(config_path.read_text()).get('targets', []) if config_path.exists() else []
    extras = thread_worktrees(targets) if targets else []
    initial, _, _, _ = sample(prefix, {}, extras)
    code_before = code_digest(initial)
    proc = subprocess.run(args, check=False)
    after = file_digest(result)
    if proc.returncode == 0 and after != 'missing' and after != before:
        final, _, _, _ = sample(prefix, {}, extras)
        if code_digest(final) == code_before:
            log('validations.log', json.dumps({'time': time.time(), 'code': code_before,
                'result': str(result), 'before': before, 'after': after,
                'runner': Path(args[0]).name, 'exit': 0}, ensure_ascii=False))
    return proc.returncode


def completed_validation(lines, events, after):
    """Only count a successful wrapper record whose new result still matches."""
    try:
        records = (STATE / 'validations.log').read_text().splitlines()
    except FileNotFoundError:
        return False
    current = code_digest(lines)
    for line in records:
        try:
            record = json.loads(line.split(' ', 1)[1])
            if (record['time'] >= after and record['exit'] == 0 and record['code'] == current
                    and record['before'] != record['after']
                    and file_digest(Path(record['result'])) == record['after']
                    and any(e[1] > 0 and e[0] <= record['time'] for e in events)):
                return True
        except (KeyError, ValueError, IndexError, OSError):
            continue
    return False


def completed_command_validation(thread, since, now, old_dirty, new_dirty, heads, events, code_activity):
    """Credit observed successful test commands with a changed evidence result.

    Ignored files and outputs outside watched evidence need the explicit wrapper,
    because their previous content cannot be established reliably.
    """
    if code_activity:
        return False
    with closing(sqlite3.connect(f'file:{THREAD_DB}?mode=ro', uri=True, timeout=1)) as db:
        rows = db.execute('SELECT item_json, completed_at_ms FROM thread_items WHERE thread_id=? '
                          'AND item_type=? AND completed_at_ms>? AND completed_at_ms<=?',
                          (thread, 'commandExecution', int(since * 1000), int(now * 1000))).fetchall()
    output_arg = re.compile(r'(?:--(?:json|out|output|log)(?:=|\s+)|(?:^|\s)>\s*)([^\s;&|]+)')
    for raw, completed_ms in rows:
        record = json.loads(raw)
        cmd = record.get('command') or ''
        finished = completed_ms / 1000
        if (record.get('status') != 'completed' or record.get('exitCode') != 0 or
                not re.search(r'(?i)\b(?:test\w*|pytest|unittest|tlm|compar\w*)\b', cmd) or
                not any(e[1] > 0 and e[0] <= finished for e in events)):
            continue
        cwd = Path(record.get('cwd') or '/')
        for match in output_arg.finditer(cmd):
            name = match.group(1).strip("'\"")
            if '$' in name or '`' in name:
                continue
            path = (cwd / name).resolve()
            for root, head in heads.items():
                try:
                    relative = str(path.relative_to(Path(root).resolve()))
                except ValueError:
                    continue
                if not relative.startswith(NONCODE):
                    continue
                key = (root, relative)
                before = old_dirty.get(key)
                if before is None:
                    old = subprocess.run(['git', '-C', root, 'show', f'{head}:{relative}'],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    if old.returncode == 0:
                        before = hashlib.sha256(old.stdout).hexdigest()
                    elif subprocess.run(['git', '-C', root, 'check-ignore', '-q', '--', relative],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                        before = 'missing'
                    else:
                        continue
                after = new_dirty.get(key, file_digest(path))
                if after != 'missing' and after != before and file_digest(path) == after:
                    return True
    return False


def check_hooks(prefix='/home/ubuntu/TritonChip'):
    common = Path(git(COMMON_REPO, 'rev-parse', '--path-format=absolute', '--git-common-dir').decode().strip())
    dest = common / 'guard'
    problems = []
    for name in ('guard.py', 'activity.py', 'monitor.py', 'watchdog.py', 'supervisor.sh',
                 'githooks/pre-commit', 'githooks/commit-msg', 'githooks/post-commit', 'githooks/pre-push'):
        expected = git(COMMON_REPO, 'show', 'origin/r41:tools/guard/' + name)
        file = dest / name
        executable = name.startswith('githooks/') or name == 'supervisor.sh'
        if (not file.is_file() or file.is_symlink() or file.read_bytes() != expected
                or (executable and not os.access(file, os.X_OK))):
            file.parent.mkdir(parents=True, exist_ok=True)
            temp = file.with_name(file.name + '.restore')
            temp.write_bytes(expected)
            temp.chmod(0o755 if executable else 0o644)
            os.replace(temp, file)
            problems.append('restored ' + name)
    expected_path = str(dest / 'githooks')
    common_config = common / 'config'
    try:
        value = command(['git', 'config', '--file', str(common_config), '--get', 'core.hooksPath']).decode().strip()
    except subprocess.CalledProcessError:
        value = ''
    if value != expected_path:
        command(['git', 'config', '--file', str(common_config), 'core.hooksPath', expected_path])
        problems.append('restored common core.hooksPath')
    for root in worktrees(prefix):
        try:
            actual = git(root, 'config', '--get', 'core.hooksPath').decode().strip()
        except subprocess.CalledProcessError:
            actual = ''
        if actual != expected_path:
            git(root, 'config', '--worktree', 'core.hooksPath', expected_path)
            problems.append(f'restored worktree Git configuration: {root}')
    if problems:
        alert('; '.join(problems))
    return problems


def save_runtime(machines, prior_status, interrupted, assessments, events, meter, heads, dirty):
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = {
        'version': 1,
        'machines': {key: {'last_productive': m.last_productive, 'last_feedback': m.last_feedback,
                           'last_assessment': m.last_assessment, 'warned': sorted(m.warned),
                           'stopped': m.stopped} for key, m in machines.items()},
        'prior_status': prior_status, 'interrupted': sorted(interrupted),
        'assessments': assessments, 'events': list(events), 'heads': heads,
        'dirty': [[a, b, v] for (a, b), v in dirty.items()],
        'meter': None if meter is None else {'previous': sorted(meter.previous),
                    'baseline': sorted(meter.baseline), 'seen': sorted(meter.seen),
                    'removed': sorted(meter.removed)},
    }
    temp = STATE / 'runtime.json.tmp'
    temp.write_text(json.dumps(value, separators=(',', ':')))
    os.replace(temp, STATE / 'runtime.json')


def load_runtime(machines):
    path = STATE / 'runtime.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value['version'] != 1 or set(value['machines']) != set(machines):
        raise ValueError('watchdog runtime state is incompatible')
    for key, state in value['machines'].items():
        m = machines[key]
        m.last_productive, m.last_feedback = state['last_productive'], state['last_feedback']
        m.last_assessment, m.warned = state['last_assessment'], set(state['warned'])
        m.stopped = state['stopped']
    saved = value['meter']
    meter = None
    if saved is not None:
        meter = Novelty(saved['baseline'])
        meter.previous, meter.seen, meter.removed = map(set, (saved['previous'], saved['seen'], saved['removed']))
    return (value['prior_status'], set(value['interrupted']), value['assessments'],
            deque(value['events']), meter, value['heads'],
            {(a, b): digest for a, b, digest in value['dirty']})


def validate_config(cfg):
    positive = ('measure_seconds', 'feedback_seconds', 'minimum_lines', 'stop_seconds',
                'assessment_seconds', 'assessment_timeout_seconds')
    if any(not isinstance(cfg.get(key), int) or cfg[key] <= 0 for key in positive):
        raise ValueError('watchdog intervals and minimum_lines must be positive integers')
    if not isinstance(cfg.get('resume_grace_seconds'), int) or cfg['resume_grace_seconds'] < 0:
        raise ValueError('resume grace must be a nonnegative integer')
    warnings = cfg.get('warning_seconds')
    if (not isinstance(warnings, list) or not warnings or
            any(not isinstance(x, int) or x <= 0 or x >= cfg['stop_seconds'] for x in warnings) or
            sorted(set(warnings)) != warnings):
        raise ValueError('warning times must be unique ascending seconds before the stop')
    targets = cfg.get('targets')
    if (not isinstance(targets, list) or not targets or
            any(not isinstance(t, dict) or not isinstance(t.get('tmux'), str) or not t['tmux'] or
                not isinstance(t.get('thread_id'), str) or not t['thread_id'] for t in targets) or
            len({t['thread_id'] for t in targets}) != len(targets) or
            len({t['tmux'] for t in targets}) != len(targets)):
        raise ValueError('targets must name distinct tmux sessions and Codex threads')


def run(config):
    cfg = {**DEFAULTS, **config}
    validate_config(cfg)
    targets = cfg['targets']
    machines = {t['thread_id']: WatchClock(time.time(), cfg) for t in targets}
    prior_status = {t['thread_id']: None for t in targets}
    interrupted = set()
    assessments = {}
    events = deque()
    last_heads = {}
    dirty = {}
    meter = None
    last_sample = last_hooks = last_save = 0
    try:
        restored = load_runtime(machines)
        if restored is not None:
            prior_status, interrupted, assessments, events, meter, last_heads, dirty = restored
    except (OSError, ValueError, KeyError, TypeError) as exc:
        alert(f'watchdog runtime state is invalid; refusing to control any session: {exc}')
        raise RuntimeError('invalid watchdog runtime state') from exc
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    validation_after = time.time()
    while True:
        now = time.time()
        state_changed = False
        if now - last_hooks >= cfg['measure_seconds']:
            try:
                check_hooks()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                alert(f'hook integrity check failed: {exc}')
            last_hooks = now
        if future is None and now - last_sample >= cfg['measure_seconds']:
            try:
                extras = thread_worktrees(targets)
                future = executor.submit(sample, cfg.get('worktree_prefix', '/home/ubuntu/TritonChip'),
                                         last_heads.copy(), extras)
            except (OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as exc:
                meter = None
                alert(f'cannot locate target worktree: {exc}')
            last_sample = now
        validation_productive = False
        if future is not None and future.done():
            state_changed = True
            try:
                current, next_dirty, heads, committed = future.result()
                if meter is None:
                    meter = Novelty(current)
                    changed, code_activity = 0, False
                else:
                    code_activity = current != meter.previous
                    changed = meter.update(current)
                evidence_activity = next_dirty != dirty or committed > 0
                events.append((now, changed, bool(evidence_activity and not code_activity),
                               bool(evidence_activity or code_activity)))
                validation_productive = completed_validation(current, events, validation_after)
                for target in targets:
                    validation_productive |= completed_command_validation(
                        target['thread_id'], validation_after, now, dirty, next_dirty,
                        last_heads, events, code_activity)
                dirty, last_heads = next_dirty, heads
                validation_after = now
            except Exception as exc:
                meter = None  # Fail closed: never stop a target on a failed measurement.
                alert(f'worktree measurement failed: {exc}')
            future = None
        while events and events[0][0] < now - cfg['stop_seconds']:
            events.popleft()
        lines30 = sum(e[1] for e in events)
        activity_windows = sum(e[3] for e in events)
        share = round(100 * sum(e[2] for e in events) / activity_windows) if activity_windows else 0
        for target in targets:
            thread = target['thread_id']
            try:
                status = goal_status(thread)
                if status != prior_status[thread]:
                    state_changed = True
                if status == 'active' and prior_status[thread] != 'active':
                    interrupted.discard(thread)
                    machines[thread].resume(now)
                if status != 'active' and thread not in interrupted:
                    prior_status[thread] = status
                    continue
                prior_status[thread] = status
                if meter is None:
                    continue
                current_event = events[-1] if events else None
                productive = bool(current_event and current_event[0] == now and
                                  (current_event[1] >= cfg['minimum_lines'] or validation_productive))
                if productive:
                    state_changed = True
                actions = machines[thread].tick(now, productive)
                has_warning = any(action.startswith('warning') for action, _ in actions)
                for action, value in actions:
                    state_changed = True
                    if action.startswith('warning'):
                        remaining = round(value / 60)
                        idle_minutes = round((cfg['stop_seconds'] - value) / 60)
                        msg = (f'已无产出{idle_minutes}分钟，距终止还剩{remaining}分钟；'
                               f'过去30分钟有效代码{lines30}行，证据空转占比{share}%；'
                               '请立刻写代码或验证，并用一句话自评。')
                        interrupt = action == 'warning_interrupt'
                        if interrupt:
                            interrupted.add(thread)
                        deliver(target, msg, interrupt=interrupt, queued=not interrupt)
                    elif action == 'stop':
                        pause(target, f'连续{cfg["stop_seconds"] // 60}分钟没有显著代码贡献')
                        interrupted.discard(thread)
                    elif action == 'recovered':
                        deliver(target, '已恢复：检测到有效代码贡献，计时已重置。')
                        if status == 'paused':
                            interrupted.discard(thread)
                    elif action == 'feedback':
                        if not has_warning:
                            deliver(target, f'过去30分钟有效代码{lines30}行，证据空转占比{share}%，距上次有效产出{int(value // 60)}分钟。')
                    elif action == 'assessment':
                        marker = '自评' + uuid.uuid4().hex[:8]
                        if deliver(target, f'请用一行回答 {marker}：过去一小时改了什么，推进哪个闭环项？'):
                            assessments[thread] = (marker, now, machines[thread].last_productive)
                if thread in assessments:
                    marker, at, prior_productivity = assessments[thread]
                    answer = find_assessment(thread, marker, at)
                    if answer:
                        log('significance.log', f'{thread} {answer}')
                        del assessments[thread]
                        state_changed = True
                    elif now - at >= cfg['assessment_timeout_seconds']:
                        log('significance.log', f'{thread} {marker} 未在10分钟内回答')
                        machines[thread].last_productive = min(machines[thread].last_productive, prior_productivity)
                        del assessments[thread]
                        state_changed = True
            except (OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError, ValueError) as exc:
                alert(f'watchdog target {target["tmux"]} failed safely: {exc}')
        if state_changed or now - last_save >= cfg['measure_seconds']:
            save_runtime(machines, prior_status, interrupted, assessments, events, meter, last_heads, dirty)
            last_save = now
        time.sleep(1)
