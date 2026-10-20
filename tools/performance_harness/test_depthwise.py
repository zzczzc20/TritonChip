"""Depthwise decomposition preserves dtype, padding and noncontiguous inputs."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]


class DepthwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import triton
        import tnca.torch_device as device
        if os.environ.get('TNCA_BACKEND', '').startswith('systemc-tlm'):
            import tlm_backend
            tlm_backend.register()
        device.register()
        cls.torch = torch
        torch.manual_seed(3107)

    def test_standard_convolution_geometries(self):
        torch = self.torch
        import torch.nn.functional as F
        from torch._inductor.utils import run_and_get_code
        from tnca.inductor import host_fallbacks
        from tnca.triton_backend.driver import launch_log
        cases = [(2, 5, 11, 3, 2, 2, 1, torch.bfloat16, True),
                 (1, 3, 9, 5, 1, 4, 2, torch.float16, True),
                 (2, 3, 12, 2, 3, 1, 1, torch.float32, False)]
        for batch, channels, length, kernel, stride, pad, dilation, dtype, has_bias in cases:
            with self.subTest(dtype=dtype, stride=stride, dilation=dilation):
                base = torch.randn(batch, length, channels).to(dtype)*.1
                x = base.transpose(1, 2)
                w = torch.randn(channels, 1, kernel).to(dtype)*.1
                bias = torch.randn(channels).to(dtype)*.1 if has_bias else None
                dx = base.to('tnca').transpose(1, 2)
                dw = w.to('tnca')
                db = bias.to('tnca') if has_bias else None
                self.assertEqual(dx.stride(), x.stride())
                fn = lambda a, b, c: F.conv1d(a, b, c, stride=stride, padding=pad,
                                               dilation=dilation, groups=channels)
                compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
                cold, code = run_and_get_code(compiled, dx, dw, db)
                self.assertFalse(host_fallbacks(code))
                with launch_log(isolated=True) as launches:
                    actual = compiled(dx, dw, db)
                self.assertTrue(launches)
                self.assertTrue(all(e['upload_bytes'] == e['download_bytes'] == 0 for e in launches))
                ref = fn(x.float(), w.float(), bias.float() if has_bias else None)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(cold.cpu().float(), ref, rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(actual.cpu().float(), ref, rtol=2e-3, atol=2e-3)

    def test_other_devices_keep_upstream_lowering(self):
        torch = self.torch
        from torch._inductor.decomposition import decompositions
        rule = decompositions[torch.ops.aten.convolution.default]
        self.assertIs(rule(torch.randn(1, 3, 9), torch.randn(3, 1, 3), None,
                           [1], [1], [1], False, [0], 3), NotImplemented)

    def test_loaded_column_broadcast_reaches_every_lane(self):
        torch = self.torch
        from torch._inductor.utils import run_and_get_code
        from tnca.inductor import host_fallbacks
        # Exercise the compiler independently of the convolution rule:
        # a loaded [R, 1] must broadcast its lane zero across each row.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.arange(27).reshape(9, 3).to(dtype).transpose(0, 1)*.01
                bias = torch.tensor([.1, .2, .3], dtype=dtype).reshape(3, 1)
                compiled = torch.compile(lambda a, b: a+b, backend='inductor',
                                         fullgraph=True, dynamic=False)
                actual, code = run_and_get_code(compiled, x.to('tnca'), bias.to('tnca'))
                self.assertFalse(host_fallbacks(code))
                torch.testing.assert_close(actual.cpu().float(), x.float()+bias.float(),
                                           rtol=2e-3, atol=2e-3)

    def test_unsupported_kernel_geometry_is_not_silently_changed(self):
        torch = self.torch
        from torch._inductor.decomposition import decompositions
        rule = decompositions[torch.ops.aten.convolution.default]
        x = torch.randn(1, 3, 19).to('tnca')
        w = torch.randn(3, 1, 17).to('tnca')
        self.assertIs(rule(x, w, None, [1], [0], [1], False, [0], 3), NotImplemented)


if __name__ == '__main__':
    unittest.main()
