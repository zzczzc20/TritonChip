#!/usr/bin/env python3
"""Device launch lifecycle checks with real processes, without RTL tools."""
import concurrent.futures
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tnca._child_exec import child_argv


class ChildExecTests(unittest.TestCase):
    def test_arguments_and_exit_status(self):
        value = 'space $literal 中文'
        result = subprocess.run(child_argv([sys.executable, '-c',
            'import sys; print(sys.argv[1]); sys.exit(7)', value]),
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout.strip(), value)

    def test_start_from_concurrent_threads(self):
        def run(i):
            return subprocess.run(child_argv([sys.executable, '-c',
                f'print({i})']), capture_output=True, text=True, timeout=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for i, result in enumerate(pool.map(run, range(32))):
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), str(i))

    def test_owner_changed_before_start_refuses_exec(self):
        command = child_argv([sys.executable, '-c', 'raise AssertionError("executed")'])
        command[2] = str(os.getpid() + 1000000)
        result = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 125)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux parent-death signal')
    def test_child_terminates_when_owner_is_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / 'ready'
            code = ('import subprocess, sys, time; '
                    'from tnca._child_exec import child_argv; '
                    'p=subprocess.Popen(child_argv([sys.executable,"-c",'
                    '"import os,time; from pathlib import Path; '
                    f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"]),'
                    'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,'
                    'start_new_session=True); time.sleep(60)')
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'python'))
            owner = subprocess.Popen([sys.executable, '-c', code], env=env)
            child = None
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists(), 'child did not start')
                child = int(ready.read_text())
                owner.kill()
                owner.wait(timeout=10)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    stat = Path(f'/proc/{child}/stat')
                    if not stat.exists() or stat.read_text().split(') ')[1][0] == 'Z':
                        break
                    time.sleep(.02)
                else:
                    self.fail('device process survived owner SIGKILL')
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=10)
                if child:
                    try:
                        os.kill(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
