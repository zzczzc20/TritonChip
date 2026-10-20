#!/usr/bin/env python3
"""Verify SDPA ablation modes and guards in the locked Torch environment."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))

import torch
from torch._inductor import config
from torch._inductor.decomposition import decompositions
from tnca import inductor, online_attention


class ConfigurationTests(unittest.TestCase):
    def test_modes_restore_and_reenable(self):
        ops = (torch.ops.aten.scaled_dot_product_attention.default,
               torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default)
        with patch.dict(os.environ, TNCA_SDPA='decomposed', TNCA_PACKED_WINDOW_UNPACK='1'):
            inductor.configure()
            baseline = [decompositions[op] for op in ops]
            os.environ['TNCA_SDPA'] = 'fused'
            inductor.configure()
            fused = [decompositions[op] for op in ops]
            self.assertTrue(all(a is not b for a, b in zip(baseline, fused)))
            # A CPU graph keeps the original public-op math semantics.
            q = torch.randn(1, 1, 3, 16)
            want = baseline[0](q, q, q)
            torch.testing.assert_close(fused[0](q, q, q), want)
            for _ in range(2):
                os.environ['TNCA_SDPA'] = 'decomposed'
                inductor.configure()
                self.assertEqual([decompositions[op] for op in ops], baseline)
                os.environ['TNCA_SDPA'] = 'fused'
                inductor.configure()
                self.assertEqual([decompositions[op] for op in ops], fused)
            os.environ['TNCA_PACKED_WINDOW_UNPACK'] = '0'
            with self.assertRaisesRegex(RuntimeError, 'PACKED_WINDOW_UNPACK'):
                online_attention.enable()
            os.environ['TNCA_PACKED_WINDOW_UNPACK'] = '1'
            with config.patch(fx_graph_cache=True):
                with self.assertRaisesRegex(RuntimeError, 'fx_graph_cache'):
                    online_attention.enable()
            os.environ['TNCA_SDPA'] = 'invalid'
            with self.assertRaisesRegex(ValueError, 'TNCA_SDPA'):
                inductor.configure()
            online_attention.disable()


if __name__ == '__main__':
    unittest.main()
