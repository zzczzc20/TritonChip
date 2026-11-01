"""The session worker must wake promptly and never lose queued launches."""
import threading
import unittest

from tnca.runtime import Session


class QueueWakeTests(unittest.TestCase):
    def test_idle_to_busy_repeatedly_and_shutdown(self):
        session = object.__new__(Session)
        session._lock = threading.RLock()
        session._queue = []
        session._stop = threading.Event()
        session._queue_wake = threading.Event()
        done = threading.Event()
        observed = []
        def run_one(task):
            observed.append(task)
            done.set()
        session._run_one = run_one
        worker = threading.Thread(target=session._run_queue, daemon=True)
        worker.start()
        try:
            for task in range(200):
                done.clear()
                with session._lock:
                    session._queue.append(task)
                    session._queue_wake.set()
                self.assertTrue(done.wait(2), f'launch {task} was not processed')
            self.assertEqual(observed, list(range(200)))
        finally:
            session._stop.set()
            session._queue_wake.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())

if __name__ == '__main__':
    unittest.main()
