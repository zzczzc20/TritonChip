"""Timeouts must stop the real graph worker and its model process."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from run_graphs import bounded_child, complete_run


class GraphWorkerTests(unittest.TestCase):
    def test_partial_success_record_cannot_hide_a_failed_worker(self):
        self.assertFalse(complete_run(dict(status='pass', returncode='timeout')))
        self.assertFalse(complete_run(dict(status='pass', returncode=7)))
        self.assertFalse(complete_run(dict(status='pass', returncode=0, fallbacks=['cpu'])))
        self.assertTrue(complete_run(dict(status='pass', returncode=0, fallbacks=[])))

    def test_success_keeps_stdout(self):
        stdout, code = bounded_child([sys.executable, '-c', 'print("complete")'],
                                      cwd='.', env=dict(os.environ), stderr=subprocess.DEVNULL, timeout=3)
        self.assertEqual((stdout, code), ('complete\n', 0))

    def test_timeout_kills_worker_and_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'descendant'
            program = ('import subprocess,sys,time; '
                       'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                       f'open({str(pidfile)!r},"w").write(str(p.pid)); '
                       'print("started",flush=True); time.sleep(30)')
            stdout, code = bounded_child([sys.executable, '-c', program], cwd=directory,
                                          env=dict(os.environ), stderr=subprocess.DEVNULL, timeout=.5)
            self.assertEqual(code, 'timeout')
            self.assertEqual(stdout, 'started\n')
            status = Path('/proc', pidfile.read_text(), 'stat')
            for _ in range(50):
                try:
                    state = status.read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    break
                if state == 'Z':
                    break
                time.sleep(.01)
            else:
                self.fail('model/compile descendant survived the graph timeout')


if __name__ == '__main__':
    unittest.main()
