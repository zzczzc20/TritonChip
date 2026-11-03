#!/usr/bin/env python3
"""Explicit host copy initialization: dense permutations and preserved gaps."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
import torch
import triton  # noqa: F401
import tnca.torch_device as D


class HostCopyTests(unittest.TestCase):
    def setUp(self):
        # The layout test uses CPU storage; only native address lookup is mocked.
        address = patch.object(D, 'storage_address',
                               side_effect=lambda t: t.untyped_storage().data_ptr())
        address.start()
        self.addCleanup(address.stop)

    def test_host_zero_cache_is_bounded_and_reuses_exact_bits(self):
        cache = D._HostZeroCache(max_bytes=128, min_bytes=1)
        first = cache.get((16,), torch.float32, 64)
        self.assertIs(first, cache.get((16,), torch.float32, 64))
        self.assertTrue(torch.equal(first.view(torch.uint8), torch.zeros(64, dtype=torch.uint8)))
        second = cache.get((24,), torch.float32, 96)
        self.assertEqual(cache.bytes, 96)
        self.assertIsNot(first, cache.get((16,), torch.float32, 64))
        self.assertLessEqual(cache.bytes, 128)
        self.assertIsNone(cache.get((100,), torch.float32, 400))
        self.assertTrue(torch.count_nonzero(second).item() == 0)

    def test_repeated_zero_upload_uses_the_same_host_storage(self):
        class Registrations:
            def __init__(self): self.ops = {}
            def impl(self, name, function, key): self.ops[name] = function
        registrations = Registrations()
        D._register_ops(registrations)
        zero = registrations.ops['zero_']
        dst = torch.empty(65536, dtype=torch.uint8)
        cache = D._HostZeroCache(max_bytes=131072, min_bytes=1)
        with patch.object(D, '_HOST_ZERO_CACHE', cache), \
             patch.dict('os.environ', {'TNCA_DEVICE_ZERO': '0'}), \
             patch.object(D.MEMORY, 'write') as write:
            zero(dst)
            zero(dst)
        first = write.call_args_list[0].args[1]
        second = write.call_args_list[1].args[1]
        self.assertEqual(first.obj.__array_interface__['data'][0],
                         second.obj.__array_interface__['data'][0])
        self.assertEqual(bytes(first), bytes(65536))

    def test_exact_contiguous_source_is_uploaded_without_a_second_host_tensor(self):
        dst = torch.empty((4, 8), dtype=torch.float32)
        src = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        with patch.object(D.MEMORY, 'write') as write, \
             patch.object(torch, 'empty', side_effect=AssertionError('unneeded host allocation')):
            D.from_host(dst, src)
        address, payload = write.call_args.args
        self.assertEqual(address, dst.data_ptr())
        self.assertEqual(bytes(payload), src.view(-1).view(torch.uint8).numpy().tobytes())
        self.assertEqual(payload.obj.__array_interface__['data'][0], src.data_ptr())

    def test_broadcast_or_dtype_change_still_materializes_the_correct_values(self):
        dst = torch.empty((2, 4), dtype=torch.float32)
        src = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        with patch.object(D.MEMORY, 'write') as write:
            D.from_host(dst, src)
        payload = write.call_args.args[1]
        actual = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(2, 4)
        self.assertTrue(torch.equal(actual, src.float().expand(2, 4)))

    def test_dense_transpose_initialization_never_reads_destination(self):
        dst = torch.empty((4, 3)).t()
        src = torch.arange(12).reshape(3, 4).float()
        with patch.object(D.MEMORY, 'read', side_effect=AssertionError('uninitialized read')), \
                patch.object(D.MEMORY, 'write') as write:
            D.from_host(dst, src)
        base, payload = write.call_args.args
        self.assertEqual(base, dst.data_ptr())
        _, got = D._staged(dst, payload, 0)
        torch.testing.assert_close(got, src, rtol=0, atol=0)

    def test_scalar_whole_storage_defines_only_allocator_padding(self):
        dst = torch.empty(1, dtype=torch.int32)
        with patch.object(D.MEMORY, 'write') as write:
            D.from_host(dst, torch.tensor([47], dtype=torch.int32))
        payload = write.call_args.args[1]
        self.assertEqual(payload, bytes([47, 0, 0, 0, 0, 0, 0, 0]))

    def test_partial_storage_copy_does_not_zero_neighbor(self):
        dst = torch.empty(2, dtype=torch.int32)[:1]
        with patch.object(D.MEMORY, 'write') as write:
            D.from_host(dst, torch.tensor([47], dtype=torch.int32))
        self.assertEqual(write.call_args.args[1], bytes([47, 0, 0, 0]))

    def test_gapped_view_preserves_original_bytes(self):
        original = torch.arange(30).reshape(3, 10).float()
        dst = original[:, 2:8:2]
        base, n, lo = D._span(dst)
        offset = base - original.data_ptr()
        raw = original.reshape(-1).view(torch.uint8).numpy().tobytes()
        with patch.object(D.MEMORY, 'read', return_value=raw[offset:offset + n]) as read, \
                patch.object(D.MEMORY, 'write') as write:
            D.from_host(dst, torch.full(dst.shape, -7.))
        read.assert_called_once_with(base, n)
        address, payload = write.call_args.args
        self.assertEqual(address, base)
        reconstructed = torch.frombuffer(bytearray(raw[:offset] + payload + raw[offset + n:]),
                                         dtype=torch.float32).reshape(3, 10)
        want = original.clone()
        want[:, 2:8:2] = -7.
        torch.testing.assert_close(reconstructed, want, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
