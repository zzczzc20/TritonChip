"""CPU checks of the optional device EOS guard's physical ABI layout."""
import struct
import threading
import unittest
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

from tnca.eos_guard import HEADER_BYTES, argument_table, guard_code
from tnca.runtime import Buffer, Session, _Allocator, _build_argument_block


class GuardLayoutTests(unittest.TestCase):
    def test_header_preserves_core_local_descriptors_and_scratch(self):
        cores, stride, base = 3, 1024, 0x40000
        params = [('value', 'u64'), ('map', 'device_ptr64')]
        resolved = {'value': 17, 'map': 0}
        maps = [('map', bytes(range(256)))]
        raw = _build_argument_block(params, resolved, maps, arg_addr=base + HEADER_BYTES,
                                    arg_stride=stride, cores=cores,
                                    map_start=256, scratch=16, scratch_start=512)
        guarded = argument_table(raw, stride, cores, 0x1200, 0x9000)
        self.assertEqual(len(guarded), stride * cores)
        for core in range(cores):
            offset = core * stride
            self.assertEqual(struct.unpack_from('<QQ', guarded, offset + 240), (0x1200, 0x9000))
            self.assertEqual(guarded[offset + HEADER_BYTES:offset + stride], raw[offset:offset + stride - HEADER_BYTES])
            self.assertEqual(struct.unpack_from('<Q', guarded, offset + HEADER_BYTES + 8)[0],
                             base + HEADER_BYTES + offset + 256)
            self.assertEqual(struct.unpack_from('<Q', guarded, offset + HEADER_BYTES + 16)[0],
                             base + HEADER_BYTES + offset + 512)
        self.assertEqual(len(guard_code()), 56)
        with self.assertRaises(ValueError):
            argument_table(raw, stride, cores, 0x1201, 0x9000)

    def test_guarded_submission_keeps_flag_leased_and_restores_normal_launches(self):
        session = Session.__new__(Session)
        session._check_usable = lambda: None
        session._lock = threading.RLock()
        session._measurement_owner = None
        session.id, session.epoch = 9, 1
        session.physical_grid, session.large_grid = True, False
        session._physical_core_count, session._next_job = 3, 0
        session._heap = _Allocator(0x10000, 0x50000)
        session._code = _Allocator(0x1000, 0x10000)
        session._events, session._active_events = weakref.WeakValueDictionary(), OrderedDict()
        session._last_event, session._queue = None, []
        session._queue_wake = mock.Mock()
        session._stop_guard_local, session._stop_guard_pc = threading.local(), None
        session.dev = SimpleNamespace(upload=mock.Mock(return_value=SimpleNamespace(ok=True)),
                                      physical_grid_capability=lambda: (3, 4096))
        flag = Buffer(1, 0x9000, 8, session.id, session.epoch)
        module = SimpleNamespace(epoch=1, alive=True, name='unit', params=[], code_addr=0x3000,
                                 manifest={'identity': {'kernel_name': 'unit'}, 'resources': {}, 'parameters': {},
                                           'execution': {'program_to_role': {'0': ['role0']}}})
        reserved = session.prepare_device_stop_guard()
        self.assertEqual(session.prepare_device_stop_guard(), reserved)
        session.dev.upload.assert_called_once()
        with session.device_stop_guard(flag):
            event = session.submit(module, {})
            task = event._task
            self.assertEqual(flag.leases, 1)
            self.assertEqual(task['arg_addr'] - task['upload_addr'], HEADER_BYTES)
            self.assertEqual(task['guarded_pc'], session._stop_guard_pc)
            self.assertEqual(struct.unpack_from('<QQ', task['block'], 240), (module.code_addr, flag.addr))
            session._finish(task, 'SUCCEEDED')
            session._queue.clear()
        self.assertEqual(flag.leases, 0)
        ordinary = session.submit(module, {})
        self.assertIsNone(ordinary._task['guarded_pc'])
        self.assertEqual(ordinary._task['upload_addr'], ordinary._task['arg_addr'])


if __name__ == '__main__':
    unittest.main()
