"""Synthetic watchdog windows, with no real tmux target."""
import tempfile
import shutil
import subprocess
import unittest
import os
import sqlite3
import json
import time
from pathlib import Path
from unittest.mock import patch

from tools.guard.activity import Novelty, code_lines, fingerprint
from tools.guard.watchdog import WatchClock, commit_changes
from tools.guard import monitor


class ClockTests(unittest.TestCase):
    def test_warnings_and_stop_at_ten_twenty_twenty_five_thirty_minutes(self):
        clock = WatchClock(0)
        self.assertIn(('warning', 1200), clock.tick(600))
        self.assertIn(('warning', 600), clock.tick(1200))
        self.assertIn(('warning_interrupt', 300), clock.tick(1500))
        self.assertIn(('stop', 0), clock.tick(1800))
        self.assertEqual(clock.tick(1801), [])

    def test_productivity_prevents_warning_and_resets_it(self):
        clock = WatchClock(0)
        self.assertEqual(clock.tick(599), [])
        self.assertEqual(clock.tick(600, productive=True), [('feedback', 0)])
        self.assertIn(('warning', 1200), clock.tick(1200))
        self.assertIn(('recovered', 0), clock.tick(1250, productive=True))
        self.assertNotIn(('warning', 1200), clock.tick(1849))
        self.assertIn(('warning', 1200), clock.tick(1850))
        self.assertIn(('warning', 600), clock.tick(2450))
        self.assertIn(('warning_interrupt', 300), clock.tick(2750))
        self.assertIn(('stop', 0), clock.tick(3050))

    def test_resume_grace(self):
        clock = WatchClock(0)
        clock.tick(1800)
        clock.resume(2000)
        self.assertFalse(any(a[0].startswith('warning') for a in clock.tick(3499)))
        self.assertIn(('warning', 1200), clock.tick(3500))
        clock.resume(4000)
        clock.tick(4060, productive=True)
        self.assertFalse(any(a[0].startswith('warning') for a in clock.tick(5499)))
        self.assertIn(('warning', 1200), clock.tick(5500))


class ActivityTests(unittest.TestCase):
    def test_comments_whitespace_docstrings_and_generated_are_ignored(self):
        before = b'"""doc"""\nx = 1 # comment\n'
        after = b'"""changed doc"""\n\nx=1 # different\n'
        self.assertEqual(code_lines('python/x.py', before), code_lines('python/x.py', after))
        self.assertEqual(code_lines('rtl/x.v', b'/* comment */\n x = 1; // old\n'),
                         code_lines('rtl/x.v', b'// newer\nx=1;\n'))
        self.assertEqual(code_lines('rtl/gen/x.v', b'wire x;\n'), [])
        self.assertEqual(code_lines('tools/guard/watchdog.py', b'x=1\n'), [])

    def test_copy_and_toggle_do_not_count_twice(self):
        original = {fingerprint('old'), fingerprint('existing elsewhere')}
        meter = Novelty(original)
        first = {fingerprint('new'), fingerprint('existing elsewhere')}
        self.assertEqual(meter.update(first), 1)
        self.assertEqual(meter.update(original), 0)
        self.assertEqual(meter.update(first), 0)

    def test_history_does_not_credit_reverting_new_lines(self):
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            try:
                os.chdir(d)
                subprocess.run(['git', 'init', '-q'], check=True)
                subprocess.run(['git', 'config', 'user.name', 'Test'], check=True)
                subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], check=True)
                Path('python').mkdir()
                Path('python/a.py').write_text('x = 1\n')
                subprocess.run(['git', 'add', '.'], check=True)
                subprocess.run(['git', 'commit', '-qm', 'base'], check=True)
                base = {fingerprint(line) for line in code_lines('python/a.py', b'x = 1\n')}
                Path('python/a.py').write_text('x = 1\ny = 2\n')
                subprocess.run(['git', 'commit', '-qam', 'add'], check=True)
                added = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
                Path('python/a.py').write_text('x = 1\n')
                subprocess.run(['git', 'commit', '-qam', 'revert'], check=True)
                reverted = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
                known, removed = set(base), set()
                self.assertEqual(commit_changes(added, known, removed, base)[0], 1)
                self.assertEqual(commit_changes(reverted, known, removed, base)[0], 0)
            finally:
                os.chdir(old_cwd)


class MonitorTests(unittest.TestCase):
    def test_pane_resolution_pins_exact_session_and_thread(self):
        target = {'tmux': '10', 'thread_id': 'abc-123'}
        with patch.object(monitor, 'command', side_effect=[
                b'10\t%10\t/dev/pts/19\n', b'bash\ncodex resume abc-123 --sandbox danger-full-access\n']) as command:
            self.assertEqual(monitor.pane_for(target), '%10')
            self.assertIn('=10:', command.call_args_list[0].args[0])
        with patch.object(monitor, 'command', side_effect=[b'11\t%11\t/dev/pts/20\n']):
            with self.assertRaises(RuntimeError):
                monitor.pane_for(target)

    def test_delivery_interrupts_then_submits_and_checks_screen(self):
        target = {'tmux': 'test', 'thread_id': 'thread'}
        state = {'input': '', 'queue': '', 'calls': []}
        def fake_command(args, **kwargs):
            state['calls'].append(args)
            if '-l' in args:
                state['input'] = args[-1]
            elif args[-1] in ('Tab', 'Enter'):
                state['queue'], state['input'] = state['input'], ''
            return b''
        def fake_screen(pane):
            return ('› ' + (state['input'] or 'Ask Codex to do anything') + '\n' + state['queue'])
        with patch.object(monitor, 'pane_for', return_value='%test'), \
             patch.object(monitor, 'command', side_effect=fake_command), \
             patch.object(monitor, 'screen', side_effect=fake_screen), \
             patch.object(monitor, 'wait_idle', return_value=True), \
             patch.object(monitor, 'log') as log, patch.object(monitor, 'alert'), \
             patch.object(monitor.time, 'sleep'):
            self.assertTrue(monitor.deliver(target, '还剩5分钟，请写代码并自评。',
                                            interrupt=True, queued=False))
            self.assertEqual(state['calls'][0][-1], 'Escape')
            self.assertEqual(state['calls'][-1][-1], 'Enter')
            self.assertIn('还剩5分钟', state['queue'])
            self.assertIn('visible after delivery', log.call_args.args[1])

    def test_delivery_retries_and_does_not_overwrite_existing_input(self):
        target = {'tmux': 'test', 'thread_id': 'thread'}
        with patch.object(monitor, 'pane_for', return_value='%test'), \
             patch.object(monitor, 'screen', return_value='› text typed by author'), \
             patch.object(monitor, 'command') as command, patch.object(monitor, 'log') as log, \
             patch.object(monitor, 'alert') as alert, patch.object(monitor.time, 'sleep'):
            self.assertFalse(monitor.deliver(target, 'warning', retries=3))
            command.assert_not_called()
            self.assertEqual(log.call_count, 3)
            alert.assert_called_once()

    def test_delivery_retries_if_message_disappears(self):
        target = {'tmux': 'test', 'thread_id': 'thread'}
        state = {'input': '', 'queue': '', 'submissions': 0}
        def fake_command(args, **kwargs):
            if '-l' in args:
                state['input'] = args[-1]
            elif args[-1] == 'Tab':
                state['submissions'] += 1
                state['queue'] = state['input'] if state['submissions'] > 1 else ''
                state['input'] = ''
            return b''
        def fake_screen(pane):
            return '› ' + (state['input'] or 'Ask Codex to do anything') + '\n' + state['queue']
        with patch.object(monitor, 'pane_for', return_value='%test'), \
             patch.object(monitor, 'command', side_effect=fake_command), \
             patch.object(monitor, 'screen', side_effect=fake_screen), \
             patch.object(monitor, 'log') as log, patch.object(monitor, 'alert'), \
             patch.object(monitor.time, 'sleep'):
            self.assertTrue(monitor.deliver(target, '还剩20分钟，请写代码并自评。'))
            self.assertEqual(state['submissions'], 2)
            self.assertEqual(log.call_count, 2)

    def test_pause_uses_only_targeted_tmux_keys(self):
        target = {'tmux': 'test', 'thread_id': 'thread'}
        with patch.object(monitor, 'pane_for', return_value='%test'), \
             patch.object(monitor, 'command', return_value=b'') as command, \
             patch.object(monitor, 'wait_idle', return_value=True), \
             patch.object(monitor, 'composer_empty', return_value=True), \
             patch.object(monitor, 'goal_status', return_value='paused'), \
             patch.object(monitor, 'log'), patch.object(monitor, 'alert'), \
             patch.object(monitor.time, 'sleep'):
            self.assertTrue(monitor.pause(target, 'test'))
        args = [call.args[0] for call in command.call_args_list]
        self.assertEqual([x[-1] for x in args], ['Escape', '/goal pause', 'Enter'])
        self.assertTrue(all(x[:4] == ['tmux', 'send-keys', '-t', '%test'] for x in args))

    def test_tampered_installed_hooks_and_common_config_are_restored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'repo'
            root.mkdir()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'Test'], check=True)
            subprocess.run(['git', '-C', str(root), 'config', 'user.email', 'test@example.invalid'], check=True)
            shutil.copytree(Path(__file__).parent, root / 'tools/guard', ignore=shutil.ignore_patterns('__pycache__'))
            subprocess.run(['git', '-C', str(root), 'add', 'tools/guard'], check=True)
            subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'base'], check=True)
            subprocess.run(['git', '-C', str(root), 'update-ref', 'refs/remotes/origin/r41', 'HEAD'], check=True)
            dest = root / '.git/guard'
            (dest / 'githooks').mkdir(parents=True)
            (dest / 'githooks/pre-commit').write_text('tampered\n')
            with patch.object(monitor, 'COMMON_REPO', root), patch.object(monitor, 'STATE', Path(d) / 'state'):
                problems = monitor.check_hooks()
                self.assertTrue(any('pre-commit' in p for p in problems))
                self.assertTrue(any('hooksPath' in p for p in problems))
                self.assertEqual((dest / 'githooks/pre-commit').read_bytes(),
                                 (root / 'tools/guard/githooks/pre-commit').read_bytes())
                self.assertEqual(monitor.check_hooks(), [])
                (dest / 'githooks/pre-commit').chmod(0o644)
                self.assertTrue(any('pre-commit' in p for p in monitor.check_hooks()))
                self.assertTrue(os.access(dest / 'githooks/pre-commit', os.X_OK))
                git = ['git', '-C', str(root)]
                linked = Path(str(root) + '-linked')
                subprocess.run(git + ['config', 'extensions.worktreeConfig', 'true'], check=True)
                subprocess.run(git + ['worktree', 'add', '-q', '--detach', str(linked), 'HEAD'], check=True)
                subprocess.run(['git', '-C', str(linked), 'config', '--worktree', 'core.hooksPath', '/dev/null'], check=True)
                self.assertTrue(any('worktree Git configuration' in p for p in monitor.check_hooks(str(root))))
                actual = subprocess.check_output(['git', '-C', str(linked), 'config', '--get', 'core.hooksPath'], text=True).strip()
                self.assertEqual(actual, str(dest / 'githooks'))

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as d, patch.object(monitor, 'STATE', Path(d)):
            clock = WatchClock(0)
            clock.tick(600)
            meter = Novelty({'a', 'b'})
            meter.update({'b', 'c'})
            monitor.save_runtime({'id': clock}, {'id': 'active'}, set(), {},
                                 [(600, 1, False, True)], meter, {'repo': 'head'}, {})
            restored_clock = WatchClock(1000)
            restored = monitor.load_runtime({'id': restored_clock})
            self.assertEqual(restored_clock.last_productive, 0)
            self.assertEqual(restored_clock.warned, {600})
            self.assertEqual(restored[4].update({'a', 'b'}), 0)

    def test_validation_requires_new_result_and_linked_code(self):
        with tempfile.TemporaryDirectory() as d, patch.object(monitor, 'STATE', Path(d)):
            result = Path(d) / 'result.json'
            snapshot = ({'a', 'b'}, {}, {}, 0)
            with patch.object(monitor, 'sample', return_value=snapshot):
                rc = monitor.run_validation(result, ['python3', '-c',
                    'from pathlib import Path; Path(' + repr(str(result)) + ').write_text("test")'])
            self.assertEqual(rc, 0)
            self.assertTrue(monitor.completed_validation({'a', 'b'}, [(0, 1, False, True)], 0))
            self.assertFalse(monitor.completed_validation({'a', 'b'}, [], 0))
            result.write_text('changed')
            self.assertFalse(monitor.completed_validation({'a', 'b'}, [(0, 1, False, True)], 0))

    def test_automatically_credits_successful_test_with_changed_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'repo'
            root.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            (root/'evidence').mkdir()
            result = root/'evidence/result.json'
            result.write_text('old')
            git('add', '.')
            git('commit', '-qm', 'base')
            head = git('rev-parse', 'HEAD')
            result.write_text('new')
            db_path = Path(d) / 'history.db'
            now = time.time()
            item = {'status': 'completed', 'exitCode': 0, 'cwd': str(root),
                    'command': 'python test_example.py --json evidence/result.json'}
            with sqlite3.connect(db_path) as db:
                db.execute('create table thread_items(thread_id text, item_type text, item_json text, completed_at_ms integer)')
                db.execute('insert into thread_items values (?, ?, ?, ?)',
                           ('thread', 'commandExecution', json.dumps(item), int((now - 1) * 1000)))
            db.close()
            key = (str(root), 'evidence/result.json')
            kwargs = dict(thread='thread', since=now-20, now=now, old_dirty={},
                          new_dirty={key: monitor.file_digest(result)}, heads={str(root): head},
                          events=[(now-10, 1, False, True)], code_activity=False)
            with patch.object(monitor, 'THREAD_DB', db_path):
                self.assertTrue(monitor.completed_command_validation(**kwargs))
                kwargs['code_activity'] = True
                self.assertFalse(monitor.completed_command_validation(**kwargs))


if __name__ == '__main__':
    unittest.main()
