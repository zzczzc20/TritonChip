"""Storage base lookups used by native launches remain constant time."""
from types import SimpleNamespace
import unittest

from tnca.torch_device import DeviceMemory


class NoScanDict(dict):
    def items(self):
        raise AssertionError('a base-address lookup scanned every allocation')


class MemoryLookupTests(unittest.TestCase):
    def test_storage_base_uses_direct_lookup(self):
        memory = DeviceMemory()
        expected = SimpleNamespace(size=64)
        memory.buffers = NoScanDict({addr: SimpleNamespace(size=64)
                                    for addr in range(0, 640000, 64)})
        memory.buffers[639936] = expected
        self.assertEqual(memory.locate(639936), (expected, 0))

    def test_offset_and_invalid_addresses_keep_existing_semantics(self):
        memory = DeviceMemory()
        first, second = SimpleNamespace(size=8), SimpleNamespace(size=16)
        memory.buffers = {64: first, 128: second}
        self.assertEqual(memory.locate(131), (second, 3))
        with self.assertRaisesRegex(RuntimeError, 'no live storage'):
            memory.locate(80)


if __name__ == '__main__':
    unittest.main()
