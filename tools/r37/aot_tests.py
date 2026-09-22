#!/usr/bin/env python3
"""AOT descriptor ABI, scratch ownership and typed cache regressions."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
import triton
import triton.language as tl
from tnca import aot, device
from tnca.cache import Cache
from tnca.errors import ArgumentError, BufferBusy, BufferError, DeviceLost
from tnca.runtime import Session, TensorDescriptor


@triton.jit
def descriptor_add(x, y, N, TYPE: tl.constexpr):
    v = x.load([0, 0]).to(TYPE)
    for _ in range(0, N):
        v = v + 1
    out = tl.make_tensor_descriptor(y, [2, 16], [16, 1], [2, 16])
    out.store([0, 0], v)


@triton.jit
def typed_load(x, y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(y+i, tl.load(x+i).to(tl.float32))


class AotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        device.prepare('core-sim', verbose=False)
        cls.directory = tempfile.TemporaryDirectory(prefix='tnca-aot-tests-')
        cls.path = Path(cls.directory.name)
        cls.art = aot.compile_aot(
            descriptor_add, signature={'x': 'tensordesc<fp32[2,16]>',
                                       'y': '*fp32', 'N': 'i32', 'TYPE': 'constexpr'},
            constexprs={'TYPE': tl.float32}, options={'opt_level': 2},
            output=str(cls.path/'descriptor.tnca'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_device_build_publishes_atomically(self):
        target = self.path/'atomic-device.vvp'
        target.write_bytes(b'old executable')
        def build(argv, **kwargs):
            output = Path(argv[argv.index('-o')+1])
            self.assertNotEqual(output, target)
            output.write_bytes(b'partial')
            self.assertEqual(target.read_bytes(), b'old executable')
            output.write_bytes(b'complete executable')
            return SimpleNamespace(returncode=0, stderr='', stdout='')
        with mock.patch.object(device, 'expected_path', return_value=str(target)), \
             mock.patch.object(device, 'prune'), \
             mock.patch.object(device.subprocess, 'run', side_effect=build):
            device.prepare(force=True, verbose=False)
        self.assertEqual(target.read_bytes(), b'complete executable')
        self.assertFalse(list(self.path.glob('.building-*')))
        with mock.patch.object(device, 'expected_path', return_value=str(target)), \
             mock.patch.object(device, 'prune'), \
             mock.patch.object(device.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='failed', stdout='')):
            with self.assertRaises(DeviceLost):
                device.prepare(force=True, verbose=False)
        self.assertEqual(target.read_bytes(), b'complete executable')
        self.assertFalse(list(self.path.glob('.building-*')))

    def test_launch_alignment_after_buffer_reuse(self):
        from tnca.runtime import _Allocator
        allocator = _Allocator(0x20000, 0x30000)
        allocator.alloc(64)
        addr, size = allocator.alloc(1024)
        self.assertNotEqual(addr % 256, 0)
        allocator.release(addr, size)
        aligned, reserved = allocator.alloc(256, align=256)
        self.assertEqual(aligned % 256, 0)
        self.assertGreaterEqual(aligned, addr)
        self.assertLessEqual(aligned+reserved, addr+size)

    def test_descriptor_layout_and_bounds(self):
        desc = TensorDescriptor(None, [2, 16], [16, 1], [1, 16], 'fp16')
        raw = desc.pack(0x12300, 64)
        self.assertEqual(struct.unpack_from('<3Q', raw), (0x12300, 2, 16))
        self.assertEqual(struct.unpack_from('<2q', raw, 48), (32, 2))
        self.assertEqual(struct.unpack_from('<2I', raw, 88), (1, 16))
        self.assertEqual(raw[108:112], bytes([2, 0, 0, 0]))
        with self.assertRaises(ArgumentError):
            desc.pack(0x12300, 63)
        with self.assertRaises(ArgumentError):
            TensorDescriptor(None, (2, 16), (1,), (1, 16), 'fp16')

    def test_queue_scratch_isolation_and_backing_buffer_leases(self):
        self.assertGreater(self.art.manifest['resources']['scratch_bytes'], 0)
        with Session() as session:
            module = session.load(self.art.path)
            x, y, z = (session.allocate(128) for _ in range(3))
            session.upload(x, struct.pack('<32f', *range(32)))
            session.upload(y, bytes(128))
            session.upload(z, bytes(128))
            desc = lambda buf: TensorDescriptor(buf, (2, 16), (16, 1), (2, 16), 'fp32')
            # Hold the queue exactly as the runtime contract tests do, so
            # this tests two live launches rather than depending on timing.
            session._stop.set()
            session._worker.join(timeout=10)
            one = session.submit(module, {'x': desc(x), 'y': y, 'N': 2})
            two = session.submit(module, {'x': desc(y), 'y': z, 'N': 2}, depends_on=(one,))
            task1, task2 = session.task_of(one), session.task_of(two)
            self.assertNotEqual(task1['allocation'][0], task2['allocation'][0])
            self.assertNotEqual(struct.unpack_from('<Q', task1['block'], 24)[0],
                                struct.unpack_from('<Q', task2['block'], 24)[0])
            with self.assertRaises(BufferBusy):
                session.free(x)
            with self.assertRaises(BufferBusy):
                session.upload(y, bytes(128))
            session._start_worker()
            session.wait(two)
            self.assertTrue(one.ok and two.ok, (one.error, two.error))
            self.assertEqual(struct.unpack('<32f', session.download(z)), tuple(float(i+4) for i in range(32)))
            self.assertNotIn('allocation', task1)
            self.assertNotIn('allocation', task2)
            session.free(x)
            with self.assertRaises(ArgumentError):
                session.submit(module, {'x': desc(x), 'y': y, 'N': 2})

    def test_buffer_bounds_refused_before_submission(self):
        with Session() as session:
            module = session.load(self.art.path)
            x, y = session.allocate(8), session.allocate(128)
            desc = lambda buf: TensorDescriptor(buf, (2, 16), (16, 1), (2, 16), 'fp32')
            with self.assertRaises(ArgumentError):
                session.submit(module, {'x': desc(x), 'y': y, 'N': 2})
            self.assertEqual(len(session.unfinished), 0)

    def test_code_cannot_overwrite_heap(self):
        with Session(heap_base=64) as session:
            with self.assertRaises(BufferError):
                session.load(self.art.path)

    def test_cache_distinguishes_pointer_element_types(self):
        cache = Cache(str(self.path/'cache'))
        artifacts = []
        for dtype in ('fp16', 'fp32', 'fp16'):
            artifacts.append(aot.compile_aot(
                typed_load, signature={'x': '*'+dtype, 'y': '*fp32', 'BLOCK': 'constexpr'},
                constexprs={'BLOCK': 16}, cache=cache, output=str(self.path/f'{dtype}.tnca')))
        self.assertFalse(artifacts[1].from_cache)
        self.assertTrue(artifacts[2].from_cache)
        self.assertNotEqual(artifacts[0].cache_key, artifacts[1].cache_key)

    def test_dtype_constexpr_is_recorded(self):
        self.assertEqual(self.art.manifest['identity']['constexprs']['TYPE'], {'triton_dtype': 'fp32'})


if __name__ == '__main__':
    unittest.main()
