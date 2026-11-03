"""CPU checks of the optional device EOS guard's physical ABI layout."""
import struct
import threading
import unittest
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

from tnca.eos_guard import (HEADER_BYTES, PROBE_BUSY_MIN_CYCLES, argument_table,
                            guard_code, stop_probe_code, stop_probe_indicates_stop)
from tnca.runtime import Buffer, Session, _Allocator, _build_argument_block


class GuardLayoutTests(unittest.TestCase):
    def test_completion_probe_is_conservative_without_a_tensor_read(self):
        self.assertGreater(len(stop_probe_code()), 0)
        self.assertTrue(stop_probe_indicates_stop(107))
        for ambiguous in (0, None, True, PROBE_BUSY_MIN_CYCLES, 1 << 40):
            self.assertFalse(stop_probe_indicates_stop(ambiguous))

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

    def test_probe_calibration_requires_a_long_unguarded_completion(self):
        session = Session.__new__(Session)
        session._check_usable = lambda: None
        session._lock = threading.RLock()
        session._stop_guard_local = threading.local()
        session._stop_probe_module = session._stop_probe_calibration = None
        session._stop_probe_preparing = False
        session._active_events = OrderedDict()
        session._modules = []
        session._code = _Allocator(0x1000, 0x10000)
        session.physical_grid, session.epoch, session.poisoned = True, 1, False
        session.dev = SimpleNamespace(upload=lambda *args: SimpleNamespace(ok=True))
        with mock.patch.object(session, 'submit', return_value=SimpleNamespace(ok=True, device_cycles=100)), \
             mock.patch.object(session, 'wait'):
            module, cycles = session.prepare_device_stop_probe()
        self.assertIsNone(module)
        self.assertEqual(cycles, 100)
        self.assertEqual(session._code.used, 0)
        with mock.patch.object(session, 'submit', return_value=SimpleNamespace(ok=True, device_cycles=60000)), \
             mock.patch.object(session, 'wait'):
            module, cycles = session.prepare_device_stop_probe()
            self.assertEqual(session.prepare_device_stop_probe(), (module, cycles))
        self.assertEqual(cycles, 60000)
        self.assertTrue(module.alive)


if __name__ == '__main__':
    unittest.main()
