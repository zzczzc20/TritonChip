"""Unsafe metadata cannot reach candidate compilation or device measurement."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_inductor_tuning_compile as fixture


class PublicSelectionRefusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CompileWorkerTests.setUpClass()
        cls.tuner = fixture.CompileWorkerTests.tuner
        cls.a, cls.b = fixture.CompileWorkerTests.a, fixture.CompileWorkerTests.b
        cls.output = fixture.CompileWorkerTests.torch.empty_like(cls.a)

    def refused(self, args, *, capture=None, mutation=()):
        from tnca.inductor_tuning import choose, SELECTIONS
        old = self.tuner.mutated_arg_names
        self.tuner.mutated_arg_names = mutation
        self.tuner._tnca_selected_signature = None
        try:
            with patch('tnca.inductor_tuning._hardware_identity', side_effect=AssertionError('unsafe candidate reached hardware')), \
                 patch('tnca.torch_device.graph.active_capture', return_value=capture):
                choose(self.tuner, args, stream=0, kwargs={})
            self.assertEqual(SELECTIONS[-1]['status'], 'conservative')
            self.assertIn('unsupported argument metadata', SELECTIONS[-1]['reason'])
        finally:
            self.tuner.mutated_arg_names = old

    def test_alias_views(self):
        self.refused((self.a, self.a.view_as(self.a), self.output, 33))

    def test_mutation(self):
        self.refused((self.a, self.b, self.output, 33), mutation=['in_ptr0'])

    def test_capture(self):
        self.refused((self.a, self.b, self.output, 33), capture=object())

    def test_unknown_descriptor_metadata(self):
        self.refused((self.a, self.b, self.output, SimpleNamespace(base=self.a)))

    def test_disabled_device_clock_is_not_a_zero_timing(self):
        from tnca.inductor_tuning import _hardware_identity
        fake = SimpleNamespace(backend_name='unused',
                               dev=SimpleNamespace(bounded_launch_limits=True, device_clock=False))
        with patch('tnca.backend.needs_hardware', return_value=True):
            with self.assertRaisesRegex(ValueError, 'endpoints are disabled'):
                _hardware_identity(fake)


if __name__ == '__main__':
    unittest.main()
