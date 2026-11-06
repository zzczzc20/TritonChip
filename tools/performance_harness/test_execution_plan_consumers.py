"""Retile, refusal and tuning admission read the sealed execution resource plan.

Each consumer is given an artifact whose code is small but whose plan states
a code size over the limit. A decision that recomputed the size from the code
would accept it; one that reads the plan refuses it with the plan's number.
The untampered artifact is accepted with the same number its code has.
Device-free: compile through tnca.aot and call the installed decisions.
"""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def scale(X, Y, N: tl.constexpr):
        offs = tl.arange(0, 16)
        tl.store(Y + offs, tl.load(X + offs, mask=offs < N) * 2., mask=offs < N)
    return scale


class PlanConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tnca import aot, artifact
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name)/'scale.tnca'
        aot.compile_aot(_kernel(), signature=dict(X='*fp32', Y='*fp32', N='constexpr'),
                        constexprs=dict(N=13), profile='core-vrf-sync-v5', output=str(path),
                        options=dict(index_opt=1, opt_level=2))
        cls.blob = path.read_bytes()
        cls.manifest, segments = artifact.load(path)
        cls.text = segments['text']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def result(self, *, plan_code_bytes=None, kwargs=None):
        from tnca import artifact
        blob = self.blob
        if plan_code_bytes is not None:
            manifest = copy.deepcopy(self.manifest)
            for key in ('manifest_hash',):
                manifest.pop(key, None)
            plan = manifest['optional_metadata']['provenance']['execution_resources']
            plan['resources']['code_bytes'] = plan_code_bytes
            blob, _sealed = artifact.build(manifest, {'text': self.text})
        return SimpleNamespace(kernel=SimpleNamespace(asm={'tnca': blob}),
                               config=SimpleNamespace(kwargs=kwargs or {}))

    def test_plan_matches_the_sealed_code(self):
        from tnca.inductor import _code_bytes, _resource_plan
        plan = _resource_plan(self.result())
        self.assertEqual(plan['resources']['code_bytes'], len(self.text))
        self.assertEqual(_code_bytes(self.result()), len(self.text))
        self.assertEqual(_code_bytes(self.result(plan_code_bytes=12345)), 12345)

    def test_block_retile_reads_the_plan(self):
        import tnca.inductor as I
        from torch._inductor.runtime import triton_heuristics as H
        I._install_block_fallback()
        wide = SimpleNamespace(kwargs={'XBLOCK': 32})
        narrow = SimpleNamespace(kwargs={'XBLOCK': 16})
        over = I.WIDE_CODE_LIMIT + 8
        results = {32: self.result(plan_code_bytes=over, kwargs=wide.kwargs),
                   16: self.result(kwargs=narrow.kwargs)}
        tuner = SimpleNamespace(configs=[wide, narrow], compile_results=[],
                                device_props=SimpleNamespace(type='tnca'),
                                inductor_meta={'kernel_name': 'scale'},
                                _precompile_config=lambda c: results[c.kwargs['XBLOCK']])
        before = len(I._BLOCK_FALLBACKS)
        H.CachingAutotuner._precompile_worker(tuner)
        self.assertIs(tuner.compile_results[0], results[16])
        record = I._BLOCK_FALLBACKS[before]
        self.assertEqual(record['used'], {'XBLOCK': 16})
        self.assertIn(f'code {over} B exceeds {I.WIDE_CODE_LIMIT} B (execution resource plan)',
                      record['reasons'][0])
        self.assertEqual(record['used_resources']['code_bytes'], len(self.text))
        # The same small code with an honest plan is the first legal config.
        results[32] = self.result(kwargs=wide.kwargs)
        tuner.configs, tuner.compile_results = [wide, narrow], []
        H.CachingAutotuner._precompile_worker(tuner)
        self.assertIs(tuner.compile_results[0], results[32])

    def test_tuning_admission_and_capacities_read_the_plan(self):
        from tnca.inductor import WIDE_CODE_LIMIT, _resource_plan
        from tnca_cc.execution_resources import admission, tuning_resources
        honest = _resource_plan(self.result())
        self.assertEqual(admission(honest, code_limit=WIDE_CODE_LIMIT), [])
        over = _resource_plan(self.result(plan_code_bytes=WIDE_CODE_LIMIT + 1))
        self.assertEqual(admission(over, code_limit=WIDE_CODE_LIMIT),
                         [('code_bytes', WIDE_CODE_LIMIT + 1, WIDE_CODE_LIMIT)])
        used, limits = tuning_resources(honest, code_limit=WIDE_CODE_LIMIT)
        self.assertEqual(used['code_bytes'], len(self.text))
        self.assertEqual(limits['code_bytes'], WIDE_CODE_LIMIT)
        self.assertEqual(set(used), {'code_bytes', 'vector_regs', 'accumulators', 'barriers', 'descriptors'})
        # The selector refuses the over-limit candidate before measuring it.
        from tnca.autotune import GraphBudget, select
        over_used, _ = tuning_resources(over, code_limit=WIDE_CODE_LIMIT)
        measured = []
        def measure(candidate):
            measured.append(candidate)
            raise AssertionError('unreachable in this test')
        with self.assertRaises(ValueError):
            select(dict(kernel_source='x', clock_hz=1), [dict(knobs={'XBLOCK': 32}, resources=over_used)],
                   capacities=limits, capabilities=[], budget=GraphBudget(max_seconds=10),
                   measure=measure)
        self.assertEqual(measured, [])


if __name__ == '__main__':
    unittest.main()
