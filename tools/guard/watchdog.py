#!/usr/bin/env python3
"""Activity watchdog; dry-run history and tests before enabling any target."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

try:
    from .activity import Novelty, code_lines, fingerprint, watched_code
except ImportError:
    from activity import Novelty, code_lines, fingerprint, watched_code

DEFAULTS = dict(measure_seconds=120, feedback_seconds=600, minimum_lines=30,
                warning_seconds=[600, 1200, 1500], stop_seconds=1800,
                resume_grace_seconds=900, assessment_seconds=3600,
                assessment_timeout_seconds=600)


class WatchClock:
    """Pure state machine; the caller executes the returned actions."""
    def __init__(self, start, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.last_productive = start
        self.last_feedback = start
        self.last_assessment = start
        self.warned = set()
        self.stopped = False

    def resume(self, now):
        self.last_productive = now + self.cfg['resume_grace_seconds']
        self.last_feedback = now
        self.last_assessment = now
        self.warned.clear()
        self.stopped = False

    def tick(self, now, productive=False):
        actions = []
        if self.stopped:
            return actions
        if productive:
            if self.warned:
                actions.append(('recovered', 0))
            self.last_productive = max(now, self.last_productive)
            self.warned.clear()
        idle = max(0, now - self.last_productive)
        if idle >= self.cfg['stop_seconds']:
            self.stopped = True
            return actions + [('stop', 0)]
        for threshold in self.cfg['warning_seconds']:
            if idle >= threshold and threshold not in self.warned:
                self.warned.add(threshold)
                actions.append(('warning_interrupt' if threshold == max(self.cfg['warning_seconds'])
                                else 'warning', self.cfg['stop_seconds'] - threshold))
        if now - self.last_feedback >= self.cfg['feedback_seconds']:
            self.last_feedback = now
            actions.append(('feedback', idle))
        if now - self.last_assessment >= self.cfg['assessment_seconds']:
            self.last_assessment = now
            actions.append(('assessment', 0))
        return actions


def run_git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd)


def history_baseline(commit):
    """Seed known lines from all existing product code, so copies do not count."""
    process = subprocess.Popen(['git', 'archive', commit, '--', 'python', 'compiler',
                                'rtl', 'tb', 'tools', 'flow', 'fpga', 'archbench', 'Makefile'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for item in archive:
                if item.isfile() and item.size <= 4_000_000 and watched_code(item.name):
                    body = archive.extractfile(item).read()
                    lines.update(map(fingerprint, code_lines(item.name, body)))
    finally:
        process.stdout.close()
    error = process.stderr.read().decode(errors='replace')
    if process.wait():
        raise RuntimeError(error)
    return lines


def commit_changes(commit, known, removed, baseline):
    parent = run_git('rev-parse', commit + '^').decode().strip()
    data = run_git('diff', '--numstat', '-z', '--no-renames', parent, commit).split(b'\0')
    additions = deletions = 0
    evidence_lines = 0
    for row in data:
        if not row:
            continue
        added, deleted, raw = row.split(b'\t', 2)
        path = raw.decode('utf-8', 'surrogateescape')
        if path.startswith(('evidence/', 'doc/', 'spec/')):
            evidence_lines += (int(added) if added.isdigit() else 0) + (int(deleted) if deleted.isdigit() else 0)
        if not watched_code(path):
            continue
        before = subprocess.run(['git', 'show', f'{parent}:{path}'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL).stdout
        after = subprocess.run(['git', 'show', f'{commit}:{path}'], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL).stdout
        old = set(map(fingerprint, code_lines(path, before)))
        new = set(map(fingerprint, code_lines(path, after)))
        fresh = new - old - known
        first_removed = (old - new) & baseline - removed
        additions += len(fresh)
        deletions += len(first_removed)
        known.update(new)
        removed.update(first_removed)
    return max(additions, deletions), evidence_lines


def replay_history(ref, hours, now=None):
    now = time.time() if now is None else now
    start = now - hours * 3600
    since = datetime.fromtimestamp(start, timezone.utc).isoformat()
    data = run_git('log', '--reverse', '--no-merges', f'--since={since}',
                   '--format=%ct %H', ref).decode().splitlines()
    events = [(int(row.split()[0]), row.split()[1]) for row in data]
    if not events:
        return dict(commits=0, productive_windows=0, stops=[], note='No commits in requested period.')
    baseline = run_git('rev-parse', events[0][1] + '^').decode().strip()
    initial = history_baseline(baseline)
    known, removed = set(initial), set()
    readings = []
    for stamp, commit in events:
        lines, evidence = commit_changes(commit, known, removed, initial)
        readings.append((stamp, lines, evidence))
    readings.sort()
    clock = WatchClock(start)
    stops, productive_windows = [], 0
    pending_lines = 0
    index = 0
    for minute in range(1, int((now - start) // 60) + 1):
        tick = start + minute * 60
        while index < len(readings) and readings[index][0] <= tick:
            _, lines, evidence = readings[index]
            pending_lines += lines
            index += 1
        productive = False
        if minute * 60 % DEFAULTS['measure_seconds'] == 0:
            productive = pending_lines >= DEFAULTS['minimum_lines']
            productive_windows += bool(productive)
            pending_lines = 0
        for action, _ in clock.tick(tick, productive):
            if action == 'stop':
                stops.append(datetime.fromtimestamp(tick, timezone.utc).isoformat())
                # Independent simulated periods after each stop, not real resumes.
                clock = WatchClock(tick)
    return dict(commits=len(events), productive_windows=productive_windows, stops=stops,
                note='Commit-only lower bound: uncommitted work and validation runs are not reconstructible.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--history', default=None, metavar='REF')
    parser.add_argument('--hours', type=float, default=30)
    parser.add_argument('--sample', action='store_true', help='read worktrees once; never send keys')
    parser.add_argument('--live', action='store_true', help='run the separately configured watchdog')
    parser.add_argument('--config', type=Path, default=Path('/home/ubuntu/.guard/targets.json'))
    parser.add_argument('--validate-result', type=Path, help='run a test and bind its new result to code')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.history and args.dry_run:
        print(json.dumps(replay_history(args.history, args.hours), indent=2))
        return 0
    if args.sample and args.dry_run:
        try:
            from .monitor import sample
        except ImportError:
            from monitor import sample
        start = time.monotonic()
        lines, dirty, heads, _ = sample('/home/ubuntu/TritonChip', {})
        print(json.dumps({'worktrees': len(heads), 'unique_code_lines': len(lines),
                          'dirty_evidence_paths': len(dirty),
                          'seconds': round(time.monotonic() - start, 2)}))
        return 0
    if args.validate_result:
        try:
            from .monitor import run_validation
        except ImportError:
            from monitor import run_validation
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        return run_validation(args.validate_result, command)
    if args.live and not args.dry_run:
        try:
            from .monitor import run
        except ImportError:
            from monitor import run
        config = json.loads(args.config.read_text())
        if not config.get('enabled'):
            parser.error('live monitoring is disabled in the configuration')
        run(config)
        return 0
    parser.error('choose --dry-run --history REF, --dry-run --sample, --validate-result, or --live')


if __name__ == '__main__':
    raise SystemExit(main())
