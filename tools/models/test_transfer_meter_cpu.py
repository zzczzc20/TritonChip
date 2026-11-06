"""CPU-only checks for chat transfer accounting and nested warmup meters."""
import unittest
from unittest import mock

import triton  # Initialize backend discovery before importing its driver.
import qwen_tnca_layers as layers
import tnca.torch_device as device


class Memory:
    def read(self, address, size):
        return bytes(size)

    def write(self, address, data):
        return len(data)


class TransferMeterTests(unittest.TestCase):
    def test_nested_counters_restore_outer_meter_and_keep_call_sites(self):
        memory = Memory()
        with mock.patch.object(device, 'MEMORY', memory):
            with layers.TransferMeter(compact=True) as outer:
                memory.write(0, b'a')
                with layers.TransferMeter(compact=True) as inner:
                    memory.write(0, b'bc')
                    memory.read(0, 3)
                memory.write(0, b'd')
            outer_result, inner_result = outer.summary(), inner.summary()
        self.assertEqual((outer_result['host_read_calls'], outer_result['host_read_bytes']), (1, 3))
        self.assertEqual((outer_result['host_write_calls'], outer_result['host_write_bytes']), (3, 4))
        self.assertEqual((inner_result['host_write_calls'], inner_result['host_write_bytes']), (1, 2))
        self.assertEqual((inner_result['host_read_calls'], inner_result['host_read_bytes']), (1, 3))
        self.assertTrue(any('test_transfer_meter_cpu.py:' in name and
                            ':test_nested_counters_restore_outer_meter_and_keep_call_sites' in name
                            for name in inner_result['host_sites']))
        self.assertNotIn('read', memory.__dict__)
        self.assertNotIn('write', memory.__dict__)

    def test_exception_still_restores_the_outer_meter(self):
        memory = Memory()
        with mock.patch.object(device, 'MEMORY', memory):
            with layers.TransferMeter(compact=True) as outer:
                with self.assertRaisesRegex(ValueError, 'expected'):
                    with layers.TransferMeter(compact=True):
                        memory.read(0, 1)
                        raise ValueError('expected')
                memory.read(0, 2)
            self.assertEqual(outer.summary()['host_read_bytes'], 3)
        self.assertNotIn('read', memory.__dict__)


if __name__ == '__main__':
    unittest.main()
