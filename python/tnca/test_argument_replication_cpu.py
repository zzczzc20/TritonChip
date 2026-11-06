"""Ordinary physical-grid argument tables are byte-identical on every core."""
import unittest

from tnca.runtime import _replicate_physical_arguments


class ArgumentReplicationTests(unittest.TestCase):
    def test_exact_arguments_and_zero_padding_for_multiple_cores(self):
        for size in (0, 8, 16, 248, 256, 320):
            block = bytes((i * 19 + 7) % 256 for i in range(size))
            stride = max(256, (size + 255) // 256 * 256)
            for cores in (1, 3, 8):
                result = _replicate_physical_arguments(block, stride, cores)
                self.assertIsInstance(result, bytearray)
                self.assertEqual(len(result), stride * cores)
                for core in range(cores):
                    self.assertEqual(result[core * stride:core * stride + size], block)
                    self.assertEqual(result[core * stride + size:(core + 1) * stride], bytes(stride - size))

    def test_invalid_table_dimensions_are_refused(self):
        for args in ((b'12345678', 4, 3), (b'', -1, 1), (b'', 0, 0)):
            with self.assertRaises(ValueError):
                _replicate_physical_arguments(*args)


if __name__ == '__main__':
    unittest.main()
