"""CPU-only checks for the opt-in large-storage device-zero path."""
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
from tnca import torch_device


class _Registrations:
    def __init__(self):
        self.ops = {}

    def impl(self, name, func, key):
        self.ops[name] = func


class _Tensor:
    dtype = torch.uint8

    def __init__(self, count, *, offset=0, storage_bytes=None, contiguous=True):
        self.shape = (count,)
        self.count = count
        self.offset = offset
        self.storage_bytes = count if storage_bytes is None else storage_bytes
        self.contiguous = contiguous

    def numel(self): return self.count
    def element_size(self): return 1
    def is_contiguous(self): return self.contiguous
    def storage_offset(self): return self.offset
    def untyped_storage(self):
        return type('Storage', (), {'nbytes': lambda _: self.storage_bytes})()


class DeviceZeroTests(unittest.TestCase):
    def test_fast_path_requires_opt_in_and_an_entire_aligned_storage(self):
        registrations = _Registrations()
        torch_device._register_ops(registrations)
        zero = registrations.ops['zero_']
        whole = _Tensor(65536)
        views = [_Tensor(65536, offset=1), _Tensor(65536, storage_bytes=65544),
                 _Tensor(65536, contiguous=False), _Tensor(65543), _Tensor(8)]
        with mock.patch.dict(os.environ, {'TNCA_DEVICE_ZERO': '1'}), \
             mock.patch('tnca.torch_device.copy.zero_full_storage', side_effect=lambda x: x) as fast, \
             mock.patch.object(torch_device, 'from_host') as staged:
            self.assertIs(zero(whole), whole)
            fast.assert_called_once_with(whole)
            for view in views:
                self.assertIs(zero(view), view)
            self.assertEqual(staged.call_count, len(views))
        with mock.patch.dict(os.environ, {'TNCA_DEVICE_ZERO': '0'}), \
             mock.patch('tnca.torch_device.copy.zero_full_storage') as fast, \
             mock.patch.object(torch_device, 'from_host') as staged:
            self.assertIs(zero(whole), whole)
            fast.assert_not_called()
            staged.assert_called_once()

    def test_triton_interpreter_clears_exactly_the_selected_bytes(self):
        source = '''
import torch
from tnca.torch_device.copy import _zero_storage_kernel, _zero_view_kernel
for count in (8, 128, 1032, 65536, 65544):
    data = torch.arange(count + 16, dtype=torch.int64).to(torch.uint8)
    original = data.clone()
    _zero_storage_kernel[(8,)](data[8:-8], count, 128, num_warps=1)
    assert torch.equal(data[:8], original[:8])
    assert torch.equal(data[-8:], original[-8:])
    assert torch.equal(data[8:-8], torch.zeros(count, dtype=torch.uint8))
base = torch.arange(1 * 2 * 16 * 8, dtype=torch.int64).to(torch.uint8).reshape(1, 2, 16, 8)
expected = base.clone()
expected[:, :, 3:6, :] = 0
view = base[:, :, 3:6, :]
_zero_view_kernel[(8,)](view, view.numel(), tuple(view.shape), tuple(view.stride()), 1, 16, 8, num_warps=1)
assert torch.equal(base, expected)
'''
        env = dict(os.environ, TRITON_INTERPRET='1')
        subprocess.run([sys.executable, '-c', source], env=env, check=True, timeout=45)


if __name__ == '__main__':
    unittest.main()
