"""Resident ABI blocks must be reuploaded after reuse, faults or writable scratch."""
from collections import OrderedDict
import threading
import types
import unittest

from tnca.runtime import Session, SUCCEEDED, FAILED, _ArgumentUploadCache


class Device:
    def __init__(self):
        self.memory = {}
        self.uploads = []

    def upload(self, address, data):
        self.memory[address] = bytes(data)
        self.uploads.append((address, bytes(data)))
        return types.SimpleNamespace(ok=True)


class Heap:
    def alloc(self, size):
        return 256, size

    def release(self, address, size):
        pass


class Event:
    def __init__(self, number):
        self.job_id, self.terminal = number, False

    def _to(self, state, error):
        self.terminal = True


class ArgumentUploadTests(unittest.TestCase):
    def setUp(self):
        self.session = object.__new__(Session)
        self.session._lock = threading.RLock()
        self.session._argument_upload_cache = _ArgumentUploadCache()
        self.session._argument_uploads_avoided = 0
        self.session._argument_upload_bytes_avoided = 0
        self.session._argument_uploads_attempted = 0
        self.session._argument_upload_bytes_attempted = 0
        self.session._active_events = OrderedDict()
        self.session._buffers = OrderedDict()
        self.session._next_buffer = 0
        self.session._heap = Heap()
        self.session.closed, self.session.poisoned = False, False
        self.session.id, self.session.epoch = 1, 0
        self.session.dev = Device()
        self.number = 0

    def task(self, data=b'argument', cacheable=True):
        self.number += 1
        return dict(event=Event(self.number), arg_addr=256, block=bytearray(data),
                    argument_cacheable=cacheable, allocation=(256, len(data)), leased=[])

    def run_task(self, task, state=SUCCEEDED):
        self.session._upload_argument_block(task)
        self.session._finish(task, state)

    def test_reuse_then_buffer_reallocation_requires_upload(self):
        self.run_task(self.task())
        self.run_task(self.task())
        self.assertEqual(len(self.session.dev.uploads), 1)
        self.assertEqual(self.session.argument_upload_stats['uploads_avoided'], 1)
        self.session.allocate(8)
        self.session.dev.memory[256] = b'changed!'
        self.run_task(self.task())
        self.assertEqual(len(self.session.dev.uploads), 2)
        self.assertEqual(self.session.dev.memory[256], b'argument')
        self.assertEqual(self.session.argument_upload_stats['uploads_attempted'], 2)

    def test_fault_and_writable_argument_blocks_invalidate(self):
        self.run_task(self.task())
        self.run_task(self.task(), FAILED)
        self.session.dev.memory[256] = b'changed!'
        self.run_task(self.task())
        self.assertEqual(len(self.session.dev.uploads), 2)
        self.run_task(self.task(cacheable=False))
        self.run_task(self.task())
        self.assertEqual(len(self.session.dev.uploads), 4)

    def test_partial_overlap_and_bounded_entries(self):
        cache = _ArgumentUploadCache(limit=2)
        cache.remember(32, b'abcdefgh')
        cache.invalidate(36, 4)
        self.assertFalse(cache.matches(32, b'abcdefgh'))
        cache.remember(32, b'a')
        cache.remember(64, b'b')
        cache.remember(96, b'c')
        self.assertFalse(cache.matches(32, b'a'))
        self.assertTrue(cache.matches(96, b'c'))

    def test_duplicate_finish_cannot_resurrect_an_invalidated_entry(self):
        task = self.task()
        self.run_task(task)
        self.session.allocate(8)
        self.session.dev.memory[256] = b'changed!'
        self.session._finish(task, SUCCEEDED)
        self.run_task(self.task())
        self.assertEqual(len(self.session.dev.uploads), 2)
        self.assertEqual(self.session.dev.memory[256], b'argument')


if __name__ == '__main__':
    unittest.main()
