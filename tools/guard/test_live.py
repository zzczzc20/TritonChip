"""End-to-end test against a private fake TUI; never targets an existing session."""
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from tools.guard import monitor

FAKE_TUI = r'''
import codecs, json, os, select, sqlite3, sys, tty
from pathlib import Path
db_path, events_path = sys.argv[1:]
tty.setraw(0)
decoder = codecs.getincrementaldecoder('utf-8')()
current = ''
queue = []
def draw():
    prompt = current or 'Ask Codex to do anything'
    sys.stdout.write('\x1b[2J\x1b[H› ' + prompt + '\r\n' + '\r\n'.join(queue[-4:]) + '\r\n')
    sys.stdout.flush()
def event(kind, text=''):
    with open(events_path, 'a') as out: out.write(json.dumps([kind,text], ensure_ascii=False)+'\n')
draw()
while True:
    if not select.select([0], [], [], 10)[0]: continue
    data = os.read(0, 4096)
    if not data: break
    for ch in decoder.decode(data):
        if ch == '\x1b': event('escape')
        elif ch == '\x15': current = ''
        elif ch in ('\t', '\r', '\n'):
            if current == 'EXIT': sys.exit(0)
            if current == '/goal pause':
                with sqlite3.connect(db_path) as db: db.execute('update thread_goals set status=?', ('paused',))
            event('tab' if ch == '\t' else 'enter', current)
            if current: queue.append(current)
            current = ''
        else: current += ch
    draw()
'''


@unittest.skipUnless(shutil.which('tmux'), 'tmux is required for the isolated integration test')
class LiveIntegrationTests(unittest.TestCase):
    def test_fake_session_receives_warnings_and_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            shutil.copytree(Path(__file__).parent, repo / 'tools/guard',
                            ignore=shutil.ignore_patterns('__pycache__'))
            (repo / 'Makefile').write_text('all:\n\t@true\n')
            git('add', '.')
            git('commit', '-qm', 'fixture')
            git('update-ref', 'refs/remotes/origin/r41', 'HEAD')
            token = uuid.uuid4().hex[:10]
            session, thread = 'guardtest' + token, 'fixture-' + token
            state, goals, history, sessions = (root/'state', root/'goals.db', root/'history.db', root/'sessions.db')
            with sqlite3.connect(goals) as db:
                db.execute('create table thread_goals(thread_id text, status text)')
                db.execute('insert into thread_goals values (?, ?)', (thread, 'active'))
            db.close()
            with sqlite3.connect(history) as db:
                db.execute('create table thread_turns(thread_id text, status text, rollout_ordinal integer)')
                db.execute('create table thread_items(thread_id text, item_type text, item_json text, created_at_ms integer, completed_at_ms integer, rollout_ordinal integer)')
            db.close()
            with sqlite3.connect(sessions) as db:
                db.execute('create table threads(id text, cwd text)')
                db.execute('insert into threads values (?, ?)', (thread, str(repo)))
            db.close()
            fake = root / 'fake_tui.py'
            fake.write_text(FAKE_TUI)
            events = root / 'events.jsonl'
            fake_command = ('exec -a ' + shlex.quote('codex resume ' + thread) + ' python3 ' +
                            ' '.join(map(shlex.quote, (str(fake), str(goals), str(events)))))
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', '-c', fake_command], check=True)
            pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t', '=' + session + ':',
                                            '#{pane_id}']).decode().strip()
            proc = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if 'Ask Codex' in monitor.screen(pane):
                        break
                    time.sleep(.1)
                else:
                    self.fail('fake TUI failed to start')
                target = {'tmux': session, 'thread_id': thread}
                config = dict(targets=[target], worktree_prefix=str(repo), measure_seconds=2,
                              feedback_seconds=100, minimum_lines=30, warning_seconds=[3, 6, 9],
                              stop_seconds=12, resume_grace_seconds=0, assessment_seconds=100,
                              assessment_timeout_seconds=10)
                ctx = multiprocessing.get_context('fork')
                with patch.object(monitor, 'STATE', state), patch.object(monitor, 'COMMON_REPO', repo), \
                     patch.object(monitor, 'GOAL_DB', goals), patch.object(monitor, 'THREAD_DB', history), \
                     patch.object(monitor, 'SESSION_DB', sessions):
                    proc = ctx.Process(target=monitor.run, args=(config,))
                    proc.start()
                    deadline = time.monotonic() + 25
                    while time.monotonic() < deadline:
                        if monitor.goal_status(thread) == 'paused':
                            break
                        if not proc.is_alive():
                            self.fail('watchdog exited before pausing the fake goal')
                        time.sleep(.2)
                    else:
                        self.fail('watchdog did not pause the fake goal')
                rows = [json.loads(line) for line in events.read_text().splitlines()]
                warnings = [text for kind, text in rows if '距终止还剩' in text]
                self.assertEqual(len(warnings), 3, rows)
                self.assertTrue(all('有效代码' in x and '证据空转占比' in x and '一句话自评' in x for x in warnings))
                self.assertEqual(sum(kind == 'escape' for kind, _ in rows), 2)
                self.assertIn(('enter', '/goal pause'), [tuple(row) for row in rows])
                self.assertTrue((state/'stop.log').exists())
            finally:
                if proc is not None and proc.is_alive():
                    proc.terminate()  # This test's own watchdog process only.
                    proc.join(5)
                subprocess.run(['tmux', 'send-keys', '-t', pane, 'C-u'], check=False)
                subprocess.run(['tmux', 'send-keys', '-t', pane, '-l', '--', 'EXIT'], check=False)
                subprocess.run(['tmux', 'send-keys', '-t', pane, 'Enter'], check=False)


if __name__ == '__main__':
    unittest.main()
