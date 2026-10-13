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
