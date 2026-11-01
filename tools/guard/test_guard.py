#!/usr/bin/env python3
"""Exercise installed hooks in isolated repositories and a local bare remote."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
VALID = 'Significance: Advance PR closure by keeping reviewable code and evidence changes together.'


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.remote = self.root / 'remote.git'
        self.state = self.root / 'state'
        self.repo.mkdir()
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'Test Author')
        self.run_git('config', 'user.email', 'test@example.invalid')
        self.write('Makefile', 'all:\n\t@true\n')
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'baseline')
        subprocess.run(['git', 'init', '--bare', '-q', str(self.remote)], check=True)
        self.run_git('remote', 'add', 'origin', str(self.remote))
        self.run_git('push', '-q', 'origin', 'HEAD:refs/heads/main')
        self.hooks = self.root / 'installed'
        (self.hooks / 'githooks').mkdir(parents=True)
        source = (HERE / 'guard.py').read_text().replace(
            "Path('/home/ubuntu/.guard')", f'Path({str(self.state)!r})')
        (self.hooks / 'guard.py').write_text(source)
        shutil.copy2(HERE / 'activity.py', self.hooks / 'activity.py')
        for hook in ('pre-commit', 'commit-msg', 'post-commit', 'pre-push'):
            shutil.copy2(HERE / 'githooks' / hook, self.hooks / 'githooks' / hook)
            (self.hooks / 'githooks' / hook).chmod(0o755)
        self.run_git('config', 'core.hooksPath', str(self.hooks / 'githooks'))

    def run_git(self, *args, ok=True):
        p = subprocess.run(['git', *args], cwd=self.repo, text=True, capture_output=True)
        if ok and p.returncode:
            self.fail(f'git {args}: {p.stdout} {p.stderr}')
        return p

    def write(self, path, content):
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    def stage(self, path, content):
        self.write(path, content)
        self.run_git('add', path)

    def commit(self, msg=VALID, ok=True, *args):
        return self.run_git('commit', '-qm', 'A useful change\n\n' + msg, *args, ok=ok)

    def code(self):
        self.stage('python/useful.py', 'answer = 42\n')

    def test_code_and_evidence_can_commit(self):
        self.code()
        self.stage('evidence/result.json', '{}\n')
        self.commit()

    def test_noncode_and_guard_only_are_rejected(self):
        for path in ('doc/note.md', 'spec/a.json', 'evidence/a.json',
                     'tools/guard/not_product.py', 'tools/other.json', 'python/comments.py'):
            with self.subTest(path=path):
                self.run_git('reset', '--hard', '-q', 'HEAD')
                self.stage(path, 'hello\n')
                if path == 'python/comments.py':
                    self.stage(path, '# comment only\n')
                self.assertNotEqual(self.commit(ok=False).returncode, 0)

    def test_comment_or_whitespace_only_code_does_not_qualify(self):
        self.code()
        self.commit()
        self.stage('python/useful.py', 'answer=42  # comment only\n')
        self.stage('evidence/result.json', '{}\n')
        self.assertNotEqual(self.commit(VALID + ' Different wording.', ok=False).returncode, 0)

    def test_evidence_file_count_bytes_and_lines(self):
        for mode in ('count', 'bytes', 'lines'):
            with self.subTest(mode=mode):
                self.run_git('reset', '--hard', '-q', 'HEAD')
                self.code()
                if mode == 'count':
                    for i in range(6):
                        self.stage(f'evidence/f{i}', '')
                elif mode == 'bytes':
                    self.stage('evidence/large', 'x' * 200001)
                else:
                    self.stage('evidence/lines', 'x\n' * 2001)
                self.assertNotEqual(self.commit(ok=False).returncode, 0)

    def test_archive_limit_everywhere(self):
        for ext in ('tgz', 'tar.gz', 'zip', 'log'):
            with self.subTest(ext=ext):
                self.run_git('reset', '--hard', '-q', 'HEAD')
                self.code()
                self.stage('elsewhere/big.' + ext, 'x' * 100001)
                self.assertNotEqual(self.commit(ok=False).returncode, 0)

    def test_message_required_specific_and_unique(self):
        for bad in ('', 'Significance: update evidence',
                    'Significance: This carefully updates the evidence and records results for PR closure.'):
            with self.subTest(bad=bad):
                self.run_git('reset', '--hard', '-q', 'HEAD')
                self.code()
                self.assertNotEqual(self.commit(bad, ok=False).returncode, 0)
        self.commit()
        self.stage('python/useful.py', 'answer = 43\n')
        self.assertNotEqual(self.commit(ok=False).returncode, 0)

    def test_pre_push_catches_no_verify_and_non_fast_forward(self):
        self.stage('doc/a.md', 'paper\n')
        self.commit('', True, '--no-verify')
        p = self.run_git('push', 'origin', 'HEAD:refs/heads/main', ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('guard:', p.stderr)
        self.run_git('push', '-q', '--no-verify', 'origin', 'HEAD:refs/heads/main')
        self.run_git('reset', '--hard', '-q', 'HEAD^')
        self.code()
        self.commit()
        p = self.run_git('push', '--force', 'origin', 'HEAD:refs/heads/main', ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('fast-forward', p.stderr)

    def test_token_applies_once_and_survives_push_validation(self):
        self.state.mkdir()
        (self.state / 'author_override').write_text('author nonce\n')
        self.stage('doc/a.md', 'author exception\n')
        self.commit('')
        self.assertFalse((self.state / 'author_override').exists())
        records = [json.loads(line) for line in (self.state / 'override.log').read_text().splitlines()]
        self.assertEqual([row['event'] for row in records], ['claimed', 'commit'])
        self.assertEqual(records[-1]['commit'], self.run_git('rev-parse', 'HEAD').stdout.strip())
        self.run_git('push', '-q', 'origin', 'HEAD:refs/heads/main')
        self.stage('doc/b.md', 'another exception\n')
        self.assertNotEqual(self.commit('', ok=False).returncode, 0)


class InstallTests(unittest.TestCase):
    def test_shared_path_protects_worktree_without_tracked_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, old = root / 'repo', root / 'old'
            repo.mkdir()
            def git(cwd, *args, check=True):
                return subprocess.run(['git', '-C', str(cwd), *args], check=check,
                                      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git(repo, 'init', '-q')
            git(repo, 'config', 'user.name', 'Test Author')
            git(repo, 'config', 'user.email', 'test@example.invalid')
            (repo / 'Makefile').write_text('all:\n\t@true\n')
            git(repo, 'add', 'Makefile')
            git(repo, 'commit', '-qm', 'baseline')
            git(repo, 'worktree', 'add', '-q', '--detach', str(old), 'HEAD')
            shutil.copytree(HERE, repo / 'tools/guard', ignore=shutil.ignore_patterns('__pycache__'))
            policy = repo / 'tools/guard/guard.py'
            policy.write_text(policy.read_text().replace("Path('/home/ubuntu/.guard')", f'Path({str(root / "state")!r})'))
            subprocess.run(['sh', str(repo / 'tools/guard/install.sh')], cwd=repo, check=True)
            shared = git(repo, 'config', '--get', 'core.hooksPath').stdout.strip()
            self.assertEqual(git(old, 'config', '--get', 'core.hooksPath').stdout.strip(), shared)
            self.assertTrue(Path(shared).is_absolute())
            (old / 'doc').mkdir()
            (old / 'doc/note.md').write_text('no product code\n')
            git(old, 'add', 'doc/note.md')
            failed = git(old, 'commit', '-qm', 'note\n\n' + VALID, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('guard:', failed.stderr)


if __name__ == '__main__':
    unittest.main()
