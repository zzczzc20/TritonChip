"""CPU unit checks for launch argument layout and queue ownership."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest import TestCase, main, mock
import os
import random
import struct
import threading

from tnca import abi
from tnca.errors import BufferError as TncaBufferError, QueueFull
from tnca.runtime import (Session, _Allocator, _build_argument_block,
                          _encode_resolved_arguments, _first_undeclared_alias,
                          _requested_queue_depth)
from tnca.torch_device.queue import submit_queued


class ArgumentLayoutTests(TestCase):
    def test_fast_encoding_matches_abi_for_boundaries(self):
        params = [('f', 'f32'), ('a', 'i32'), ('b', 'u32'),
                  ('c', 'i64'), ('d', 'u64'), ('p', 'device_ptr64')]
        rng = random.Random(41)
        floats = [0.0, -0.0, float('inf'), float('-inf'), float('nan')]
        floats += [rng.uniform(-1e30, 1e30) for _ in range(200)]
        for f in floats:
            values = {'f': f, 'a': rng.randrange(-(1 << 31), 1 << 31),
                      'b': rng.randrange(1 << 32),
                      'c': rng.randrange(-(1 << 63), 1 << 63),
                      'd': rng.randrange(1 << 64), 'p': rng.randrange(1 << 64)}
            self.assertEqual(_encode_resolved_arguments(params, values),
                             abi.encode_block(params, values))

    def test_physical_layout_matches_original_per_core_encoding(self):
        params = [('value', 'i32'), ('map1', 'device_ptr64'),
                  ('map2', 'device_ptr64'), ('value2', 'u64')]
        addr = 0x20000
        for map_count in (0, 1, 2):
            for scratch in (0, 64):
                for cores in (0, 1, 3, 8):
                    maps = [(f'map{i+1}', bytes([i + 1]) * 256)
                            for i in range(map_count)]
                    resolved = {'value': -7, 'map1': 0x40000,
                                'map2': 0x50000, 'value2': (1 << 64) - 1}
                    slots = abi.block_size(len(params)) + (8 if scratch else 0)
                    map_start = (slots + 255) // 256 * 256
                    scratch_start = map_start + 256 * map_count
                    size = max(scratch_start + scratch, slots, 8)
                    stride = (size + 255) // 256 * 256 if cores else 0
                    reference = bytearray(scratch_start + scratch if scratch
                                          else (scratch_start if maps else slots))
                    for i, (name, data) in enumerate(maps):
                        resolved[name] = addr + map_start + 256 * i
                        reference[map_start + 256 * i:map_start + 256 * (i + 1)] = data
                    reference[:abi.block_size(len(params))] = abi.encode_block(params, resolved)
                    if scratch:
                        struct.pack_into('<Q', reference, abi.block_size(len(params)), addr + scratch_start)
                    if cores:
                        table = bytearray(stride * cores)
                        for core in range(cores):
                            core_args, block = dict(resolved), bytearray(reference)
                            for i, (name, _) in enumerate(maps):
                                core_args[name] = addr + core * stride + map_start + 256 * i
                            block[:abi.block_size(len(params))] = abi.encode_block(params, core_args)
                            if scratch:
                                struct.pack_into('<Q', block, abi.block_size(len(params)),
                                                 addr + core * stride + scratch_start)
                            table[core * stride:core * stride + len(block)] = block
                        reference = table
                    result = _build_argument_block(params, dict(resolved), maps, arg_addr=addr,
                              arg_stride=stride, cores=cores, map_start=map_start,
                              scratch=scratch, scratch_start=scratch_start)
                    self.assertEqual(result, reference, (map_count, scratch, cores))

    def test_alias_sweep_preserves_first_conflict(self):
        rng = random.Random(42)
        for _ in range(2000):
            spans = [(str(i), start, start + rng.randrange(1, 10), None)
                     for i in range(rng.randrange(31)) for start in [rng.randrange(50)]]
            allowed = {tuple(sorted((spans[i][0], spans[j][0])))
                       for i in range(len(spans)) for j in range(i + 1, len(spans))
                       if rng.random() < .1}
            expected = None
            for i, (name, lo, hi, _) in enumerate(spans):
                for other, lo2, hi2, _ in spans[i + 1:]:
                    if lo < hi2 and lo2 < hi and tuple(sorted((name, other))) not in allowed:
                        expected = (name, other)
                        break
                if expected:
                    break
            self.assertEqual(_first_undeclared_alias(spans, allowed), expected)


class QueueOwnershipTests(TestCase):
    def test_depth_is_bounded_and_defaults_to_eight(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_requested_queue_depth(), 8)
            for value in ('8', '32', '128'):
                os.environ['TNCA_QUEUE_DEPTH'] = value
                self.assertEqual(_requested_queue_depth(), int(value))
            for value in ('7', '129', 'invalid'):
                os.environ['TNCA_QUEUE_DEPTH'] = value
                with self.assertRaises(ValueError):
                    _requested_queue_depth()

    def test_queue_waits_and_retries_after_transient_memory_pressure(self):
        predecessor = SimpleNamespace(ok=True)
        session = mock.Mock(max_unfinished=32, unfinished_count=1,
                            oldest_unfinished=predecessor, latest_event=predecessor)
        admitted = object()
        session.submit.side_effect = [TncaBufferError('out of device memory: temporary'), admitted]
        self.assertIs(submit_queued(session, object(), {}, (1, 1, 1)), admitted)
        session.wait.assert_called_once_with(predecessor)
        self.assertEqual(session.submit.call_count, 2)

    def test_queue_full_race_retries_without_predecessor(self):
        session = mock.Mock(max_unfinished=8, unfinished_count=0,
                            oldest_unfinished=None, latest_event=None)
        admitted = object()
        session.submit.side_effect = [QueueFull('race'), admitted]
        self.assertIs(submit_queued(session, object(), {}, (1, 1, 1)), admitted)
        session.wait.assert_not_called()

    def test_publication_failure_releases_reservation_and_leases(self):
        session = Session.__new__(Session)
        session._check_usable = lambda: None
        session._lock = threading.RLock()
        session._measurement_owner = None
        session.epoch = 1
        session.physical_grid = False
        session.large_grid = False
        session._next_job = 0
        session._heap = _Allocator(0x20000, 0x30000)
        session._events = OrderedDict()
        session._active_events = OrderedDict()
        session._queue = []
        session._queue_wake = mock.Mock()
        session._queue_wake.set.side_effect = MemoryError('publication failed')
        buffer = SimpleNamespace(leases=0)
        session._resolve = lambda module, args: ({}, [buffer])
        module = SimpleNamespace(epoch=1, alive=True, name='unit', params=[],
                                 manifest={'identity': {'kernel_name': 'unit'}, 'resources': {}})
        with self.assertRaises(MemoryError):
            session.submit(module, {})
        self.assertEqual(buffer.leases, 0)
        self.assertEqual(session._heap.used, 0)
        self.assertEqual((session._events, session._active_events, session._queue),
                         (OrderedDict(), OrderedDict(), []))


if __name__ == '__main__':
    main()
