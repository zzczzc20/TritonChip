#!/usr/bin/env python3
"""Reproduce the original large-model TMU alias without allocating host RAM."""
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.runtime import _Allocator, DEVICE_RESERVED_RANGES
from tnca.errors import BufferError


class ReservedMemoryTests(unittest.TestCase):
    def test_original_large_model_layout_and_mutation(self):
        report = json.loads((ROOT / 'evidence/r40/v2/convergence/attempt03/'
                             'large_native_mmio_collision_6bd07b7.json').read_text())
        buffers = sorted(report['parameter_buffers'], key=lambda b: b['address'])
        old = _Allocator(0x20000, (4 << 30) + 0x20000)
        fixed = _Allocator(old.base, old.limit, reserved=DEVICE_RESERVED_RANGES)
        lo, hi = DEVICE_RESERVED_RANGES[0]
        conflicts = []
        for b in buffers:
            address, _ = old.alloc(b['bytes'])
            self.assertEqual(address, b['address'], b['name'])
            if address < hi and address + b['bytes'] > lo:
                conflicts.append(b['name'])
            address, size = fixed.alloc(b['bytes'])
            self.assertFalse(address < hi and address + size > lo, b['name'])
        self.assertEqual(conflicts, ['out.weight'])
        self.assertEqual(fixed.used, old.used)
        self.assertEqual(fixed.capacity, old.capacity - 4096)

    def test_prefix_reuse_release_alignment_and_oom_atomicity(self):
        a = _Allocator(0, 1024, reserved=((256, 320), (640, 768)))
        x = a.alloc(192)
        y = a.alloc(256)
        self.assertEqual(y, (320, 256))
        self.assertEqual(a.alloc(64), (192, 64))
        self.assertEqual(a.alloc(128, align=256), (768, 256))
        state = (a.next, a.used, list(a.free_list))
        with self.assertRaises(BufferError):
            a.alloc(192)
        self.assertEqual((a.next, a.used, a.free_list), state)
        for block in (x, y, (192, 64), (768, 256)):
            a.release(*block)
        self.assertEqual((a.next, a.used, a.free_list), (0, 0, []))

    def test_randomized_live_ranges_accounting_and_full_recovery(self):
        rng = random.Random(40)
        a = _Allocator(0, 8192, reserved=((1024, 1536), (4096, 4352)))
        live = []
        for _ in range(5000):
            if live and rng.random() < .48:
                a.release(*live.pop(rng.randrange(len(live))))
            else:
                try:
                    block = a.alloc(rng.randrange(1, 500), align=rng.choice((64, 128, 256)))
                except BufferError:
                    continue
                start, size = block
                for lo, hi in a.reserved:
                    self.assertFalse(start < hi and start + size > lo)
                for lo, n in live:
                    self.assertFalse(start < lo + n and start + size > lo)
                live.append(block)
            self.assertEqual(a.used, sum(n for _, n in live))
        rng.shuffle(live)
        for block in live:
            a.release(*block)
        self.assertEqual((a.next, a.used, a.free_list), (0, 0, []))

    def test_small_memory_is_unchanged(self):
        plain = _Allocator(0x20000, 0x80000)
        protected = _Allocator(plain.base, plain.limit, reserved=DEVICE_RESERVED_RANGES)
        for size in (1, 1234, 8192, 77):
            self.assertEqual(plain.alloc(size), protected.alloc(size))
        self.assertEqual((plain.used, plain.capacity), (protected.used, protected.capacity))


if __name__ == '__main__':
    unittest.main()
