#!/usr/bin/env python3
"""Local Git policy. Installed in the common Git directory, not a worktree."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

STATE = Path('/home/ubuntu/.guard')
ZERO = '0' * 40
NONCODE = ('.md', '.rst', '.txt', '.json', '.jsonl', '.yaml', '.yml',
           '.toml', '.csv', '.tsv', '.log', '.tgz', '.tar.gz', '.zip',
           '.pdf', '.png', '.jpg', '.jpeg', '.svg', '.lock')
ARCHIVE = ('.tgz', '.tar.gz', '.zip', '.log')
CODE_DIRS = ('python/', 'compiler/', 'rtl/', 'tb/', 'tools/', 'flow/',
             'fpga/', 'archbench/')
ITEMS = re.compile(r'(?i)(tokens?\s*/\s*s|tokens? per second|chat quality|'
                   r'no[- ]cpu[- ]fallback|image bring[- ]up|pr closure)')


def git(*args, input=None, check=True):
    p = subprocess.run(['git', *args], input=input, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    if check and p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace').strip())
    return p.stdout


def blob(ref, path):
    p = subprocess.run(['git', 'cat-file', 'blob', f'{ref}:{path}'],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout if p.returncode == 0 else b''


def exists(ref, path):
    return subprocess.run(['git', 'cat-file', '-e', f'{ref}:{path}'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def is_code(path):
    lower = path.lower()
    if lower.startswith(('evidence/', 'doc/', 'spec/', 'tools/guard/')):
        return False
    if lower.endswith(NONCODE):
        return False
    return (path == 'Makefile' or path.startswith(CODE_DIRS)
            or bool(re.search(r'(^|/)(test[^/]*|[^/]*_test[^/]*)$', lower)))


def added_bytes(old, new):
    if b'\0' in old or b'\0' in new:
        return len(new) if old != new else 0
    before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=True)
    return sum(sum(map(len, after[j:k])) for op, _, _, j, k in matcher.get_opcodes()
               if op in ('insert', 'replace'))


def changes(old, new, staged=False):
    args = ['diff', '--numstat', '-z', '--no-renames']
    args += ['--cached'] if staged else [old, new]
    records = git(*args).split(b'\0')
    entries = []
    for record in records:
        if not record:
            continue
        added, deleted, rawpath = record.split(b'\t', 2)
        path = os.fsdecode(rawpath)
        target = '' if staged else new
        before, after = blob(old, path), blob(target, path)
        entries.append((path, added, before, after, exists(old, path), exists(target, path)))
    return entries


def policy_errors(entries):
    errors = []
    # Compare code tokens, not just paths: comments and formatting cannot make
    # an evidence-only commit appear to contain product code.
    try:
        from .activity import code_lines
    except ImportError:
        from activity import code_lines
    if not any(is_code(path) and code_lines(path, before) != code_lines(path, after)
               for path, _, before, after, _, _ in entries):
        errors.append('commit must change a code path outside tools/guard/')
    evidence_bytes = evidence_lines = evidence_new = 0
    for path, added, before, after, old_exists, new_exists in entries:
        if path.startswith('evidence/'):
            evidence_bytes += added_bytes(before, after)
            evidence_lines += int(added) if added != b'-' else 0
            if not old_exists and new_exists:
                evidence_new += 1
        if path.lower().endswith(ARCHIVE) and len(after) > 100_000 and after != before:
            errors.append(f'{path}: archive or log exceeds 100 KB')
    if evidence_bytes > 200_000:
        errors.append(f'evidence adds {evidence_bytes} bytes (limit 200000)')
    if evidence_lines > 2_000:
        errors.append(f'evidence adds {evidence_lines} lines (limit 2000)')
    if evidence_new > 5:
        errors.append(f'evidence adds {evidence_new} files (limit 5)')
    return errors


def message_errors(message, ancestors='HEAD'):
    trailers = re.findall(r'^Significance:\s*(.*?)\s*$', message, re.M | re.I)
    if len(trailers) != 1:
        return ['exactly one Significance trailer is required']
    sentence = trailers[0].strip()
    words = re.findall(r"[\w]+(?:[-/'\w]*[\w])?", sentence, re.UNICODE)
    errors = []
    if len(words) < 8 or not ITEMS.search(sentence):
        errors.append('Significance needs at least eight words and a named closure item')
    if re.search(r'(?i)\b(?:updat\w* (?:the )?evidence|record\w* (?:the )?results|'
                 r'preserv\w* (?:the )?evidence|register\w* (?:the )?results)\b', sentence):
        errors.append('generic evidence bookkeeping is not a significance sentence')
    if len(re.findall(r'[.!?](?:\s|$)', sentence)) > 1 or '\n' in sentence:
        errors.append('Significance must be one sentence')
    history = git('log', '--format=%B%x00', ancestors, check=False).decode(errors='replace')
    normalized = ' '.join(sentence.casefold().split())
    if any(' '.join(s.casefold().split()) == normalized
           for s in re.findall(r'^Significance:\s*(.*?)\s*$', history, re.M | re.I)):
        errors.append('Significance repeats an earlier sentence')
    return errors


def signature(tree, parent, message):
    return {'tree': tree, 'parent': parent,
            'message_sha256': hashlib.sha256(message.encode()).hexdigest()}


def pending_signature():
    tree = git('write-tree').decode().strip()
    parent = git('rev-parse', '--verify', 'HEAD', check=False).decode().strip() or ZERO
    return tree, parent


def take_token(tree, parent):
    token = STATE / 'author_override'
    try:
        # Rename makes a token claim atomic among concurrent commits.
        claimed = STATE / f'claimed-{os.getpid()}-{time.time_ns()}'
        token.rename(claimed)
    except FileNotFoundError:
        return False
    try:
        nonce = claimed.read_text().strip()
        if not nonce:
            raise RuntimeError('empty author override token')
        STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
        pending = {'tree': tree, 'parent': parent,
                   'nonce_sha256': hashlib.sha256(nonce.encode()).hexdigest()}
        (STATE / f'pending-{tree}-{parent}').write_text(json.dumps(pending))
        with (STATE / 'override.log').open('a') as out:
            out.write(json.dumps({**pending, 'event': 'claimed', 'time': time.time()}, sort_keys=True) + '\n')
        return True
    finally:
        claimed.unlink(missing_ok=True)


def pending(tree, parent):
    return STATE / f'pending-{tree}-{parent}'


def finish_override(tree, parent, message):
    path = pending(tree, parent)
    if not path.exists() and not take_token(tree, parent):
        return False
    info = json.loads(path.read_text())
    info['approved'] = True
    path.write_text(json.dumps(info, sort_keys=True))
    return True


def post_commit():
    commit = git('rev-parse', 'HEAD').decode().strip()
    tree = git('rev-parse', 'HEAD^{tree}').decode().strip()
    parent = git('rev-parse', 'HEAD^', check=False).decode().strip() or ZERO
    path = pending(tree, parent)
    if not path.exists():
        return 0
    info = json.loads(path.read_text())
    if not info.get('approved'):
        return 0
    message = git('log', '-1', '--format=%B', 'HEAD').decode().rstrip('\n') + '\n'
    info.update(signature(tree, parent, message))
    with (STATE / 'override.log').open('a') as out:
        out.write(json.dumps({**info, 'event': 'commit', 'commit': commit, 'time': time.time()}, sort_keys=True) + '\n')
    path.unlink()
    return 0


def overridden(commit):
    try:
        lines = (STATE / 'override.log').read_text().splitlines()
        return any(row.get('event') == 'commit' and row.get('commit') == commit
                   for row in map(json.loads, lines))
    except (OSError, ValueError):
        return False


def reject(errors):
    for error in errors:
        print(f'guard: {error}', file=sys.stderr)
    return 1 if errors else 0


def pre_commit():
    tree, parent = pending_signature()
    errors = policy_errors(changes('HEAD', '', staged=True))
    claimed = take_token(tree, parent)
    if errors and not claimed and not pending(tree, parent).exists():
        return reject(errors)
    return 0


def commit_msg(path):
    message = Path(path).read_text()
    tree, parent = pending_signature()
    errors = message_errors(message)
    if errors or pending(tree, parent).exists():
        if finish_override(tree, parent, message):
            return 0
        return reject(errors)
    return 0


def pre_push(lines):
    for line in lines:
        local_ref, local, remote_ref, remote = line.split()
        if local == ZERO:
            return reject([f'{remote_ref}: deleting a remote ref is not a fast-forward'])
        if remote != ZERO and subprocess.run(
                ['git', 'merge-base', '--is-ancestor', remote, local],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            return reject([f'{remote_ref}: not a verifiable fast-forward'])
        revs = [local] + ([f'^{remote}'] if remote != ZERO else [])
        for commit in git('rev-list', '--reverse', *revs).decode().splitlines():
            parent = git('rev-parse', f'{commit}^', check=False).decode().strip() or ZERO
            tree = git('rev-parse', f'{commit}^{{tree}}').decode().strip()
            message = git('log', '-1', '--format=%B', commit).decode().rstrip('\n') + '\n'
            if overridden(commit):
                continue
            if parent == ZERO:
                return reject([f'{commit}: root commit cannot be verified'])
            errors = policy_errors(changes(parent, commit)) + message_errors(message, parent)
            if errors:
                return reject([f'{commit}: {e}' for e in errors])
    return 0


def main():
    try:
        command = sys.argv[1]
        if command == 'pre-commit':
            return pre_commit()
        if command == 'commit-msg':
            return commit_msg(sys.argv[2])
        if command == 'post-commit':
            return post_commit()
        if command == 'pre-push':
            return pre_push(sys.stdin)
        raise RuntimeError(f'unknown hook {command}')
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return reject([str(exc)])


if __name__ == '__main__':
    raise SystemExit(main())
