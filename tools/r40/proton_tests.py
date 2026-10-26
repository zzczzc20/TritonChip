#!/usr/bin/env python3
"""Compiler, capture corruption, overhead and resource-screening invariants."""
import copy
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.proton import lower_records
from tnca_cc.codegen import CodeGen
from tnca_cc.target import TargetInfo
from tnca_cc.diagnostics import Diagnostic
from tnca.proton import chrome_trace, paired_overhead
from tnca.costmodel import (resource_screen, calibrated_interval, counter_diagnosis,
                           CalibratedModel, implementation_screen)
from tnca_asm import Assembler

IDENTITY = dict(software_sha256='1' * 64, config_sha256='2' * 64,
               image_sha256='3' * 64, contract_sha256='4' * 64,
               clock_hz=100_000_000, backend='synthetic-test-fixture')


def kernel(markers=True):
    k = make_kernel('proton_test')
    b = Builder(k, k.role(0).region)
    if markers:
        b.proton_record('compute')
    value = b.sbin('add', b.const(10), b.const(20))
    if markers:
        b.proton_record('compute', start=False)
    b.sstore(b.const(0x10000), value)
    b.exit()
    return k


def assemble(k):
    g = CodeGen(k, TargetInfo.default())
    g.assign_shared()
    asm = g.run().text()
    return [(a, n, v) for a, n, v, _ in Assembler().assemble(asm)]


class Tests(unittest.TestCase):
    def test_off_identical_and_unlowered_refused(self):
        off, meta = lower_records(kernel(), buffer_base=0, buffer_bytes=0, enabled=False)
        self.assertIsNone(meta)
        self.assertEqual(assemble(off), assemble(kernel(False)))
        with self.assertRaises(Diagnostic):
            assemble(kernel())

    def test_lowering_and_no_region_stores(self):
        original = kernel()
        before = str(original)
        out, meta = lower_records(original, buffer_base=0x20000, buffer_bytes=64)
        self.assertEqual(before, str(original))
        assemble(out)
        kinds = [o.kind for o in out.role(0).region.ops]
        self.assertEqual(kinds.count('proton_timestamp'), 2)
        clocks = [i for i, kind in enumerate(kinds) if kind == 'proton_timestamp']
        self.assertNotIn('sstore', kinds[clocks[0]:clocks[-1]])
        self.assertEqual(meta['required_bytes'], 32)

    def test_scheduler_preserves_scope_boundaries(self):
        from tnca_cc.schedule import schedule_kernel
        k, _ = lower_records(kernel(), buffer_base=0x20000, buffer_bytes=64)
        schedule_kernel(k, TargetInfo.default())
        ops = k.role(0).region.ops
        lo, hi = [i for i, o in enumerate(ops) if o.kind == 'proton_timestamp']
        self.assertIn('sbin', [o.kind for o in ops[lo:hi]])
        assemble(k)

    def test_refusals(self):
        for base, size in ((1, 64), (0x20000, 31), ((1 << 64) - 8, 64)):
            with self.assertRaises((Diagnostic, ValueError)):
                lower_records(kernel(), buffer_base=base, buffer_bytes=size)
        for mutate in ('loop', 'unclosed', 'crossed', 'reused'):
            k = kernel()
            b = Builder(k, k.role(0).region)
            if mutate == 'loop':
                k.role(0).region.ops.insert(0, copy.copy(k.role(0).region.ops[-1]))
            elif mutate == 'unclosed':
                k.role(0).region.ops.pop(4)
            elif mutate == 'crossed':
                k.role(0).region.ops[4].attrs['name'] = 'wrong'
            else:
                markers = [o for o in k.role(0).region.ops if o.kind == 'proton.record']
                k.role(0).region.ops[-1:-1] = copy.deepcopy(markers)
            with self.assertRaises(Diagnostic, msg=mutate):
                lower_records(k, buffer_base=0x20000, buffer_bytes=256)

    def test_trace_corruption_and_clock_domains(self):
        _, meta = lower_records(kernel(), buffer_base=0x20000, buffer_bytes=64, grid=(2, 1, 1))
        raw = struct.pack('<8Q', 1, 0, 100, 120, 1, 2, 900, 940)
        trace = chrome_trace(raw, meta, identity=IDENTITY)
        spans = [e for e in trace['traceEvents'] if e['ph'] == 'X']
        self.assertEqual([e['dur'] for e in spans], [.2, .4])
        self.assertEqual([e['ts'] for e in spans], [0, 0])
        for data in (raw[:-1], bytes(len(raw)), struct.pack('<8Q', 1, 0, 120, 100, 1, 2, 900, 940)):
            with self.assertRaises(ValueError):
                chrome_trace(data, meta, identity=IDENTITY)
        with self.assertRaises(ValueError):
            chrome_trace(raw, meta, identity=dict(IDENTITY, image_sha256='short'))

    def test_overhead_detects_perturbation_and_mismatches(self):
        pairs = [dict(identity=IDENTITY, order='AB' if i % 2 == 0 else 'BA',
                      baseline_cycles=100, instrumented_cycles=120,
                      baseline_code_sha256='5'*64, instrumented_code_sha256='6'*64)
                 for i in range(6)]
        r = paired_overhead(pairs, identity=IDENTITY, max_relative_overhead=.01)
        self.assertEqual(r['status'], 'perturbed-use-coarser-scopes')
        pairs[-1]['identity'] = dict(IDENTITY, clock_hz=75_000_000)
        with self.assertRaises(ValueError):
            paired_overhead(pairs, identity=IDENTITY, max_relative_overhead=.01)

    def test_shared_port_bound_does_not_scale_with_cores(self):
        demand = {'pc0-read': dict(work=6400, unit='bytes', source='fixture work'),
                  'issue': dict(work=300, unit='instructions', source='fixture work')}
        caps = {'pc0-read': dict(per_cycle=16, unit='bytes', source='physical port limit'),
                'issue': dict(per_cycle=3, unit='instructions', source='three issue slots')}
        r = resource_screen(demand, caps, identity=IDENTITY, cycle_budget=300)
        self.assertEqual(r['lower_bound_cycles'], 400)
        self.assertEqual(r['decision'], 'reject-impossible-budget')
        caps['issue']['per_cycle'] = 30
        self.assertEqual(resource_screen(demand, caps, identity=IDENTITY)['lower_bound_cycles'], 400)
        with self.assertRaises(ValueError):
            resource_screen(demand, {}, identity=IDENTITY)

    def test_calibration_algebra_and_counter_partition(self):
        self.assertEqual(calibrated_interval(100, [.5]*19), (100/1.5, 200))
        self.assertEqual(calibrated_interval(100, [.1]*3), (0, None))
        c = dict(perf_cycles=100, perf_issued=30, perf_stall_total=60,
                 perf_x_frontend=10, perf_x_lsu=50, perf_x_hazard=10,
                 perf_stall_lsu=70)
        self.assertEqual(counter_diagnosis(c)['eliminate_one_component_speedup_ceiling']['perf_x_lsu'], 2)
        c['perf_x_lsu'] = 49
        with self.assertRaises(ValueError):
            counter_diagnosis(c)

    def test_calibrated_model_refuses_holdout_and_identity_extrapolation(self):
        cap = {'issue': dict(per_cycle=1, unit='instructions', source='fixture hardware limit')}
        def demand(n):
            return {'issue': dict(work=n, unit='instructions', source='fixture exact instruction count')}
        def row(n, split, index):
            return dict(id=f'{split}-{index}', split=split, identity=IDENTITY,
                        instrumented=False, demand=demand(n), cycles=10+2*n)
        m = CalibratedModel(IDENTITY, cap)
        m.fit([row(n, 'dev', i) for i, n in enumerate((100, 200, 300))])
        m.calibrate([row(150+i, 'validation', i) for i in range(19)])
        p = m.predict(demand(160), identity=IDENTITY)
        self.assertEqual(p['predicted_cycles'], 330)
        self.assertEqual(p['interval_90'], [330, 330])
        self.assertEqual(m.predict(demand(400), identity=IDENTITY)['status'], 'out-of-domain')
        self.assertEqual(m.predict(demand(160), identity=dict(IDENTITY, clock_hz=1))['status'], 'out-of-domain')
        with self.assertRaises(ValueError):
            m.calibrate([row(160, 'heldout', 0)])
        with self.assertRaises(ValueError):
            m.calibrate([row(150, 'validation', 0)])

    def test_outer_scope_allows_loop_but_not_dynamic_markers(self):
        k = make_kernel('loop')
        b = Builder(k, k.role(0).region)
        count = b.const(0)
        b.proton_record('outer')
        b.label('loop')
        b.sbini('add', count, 1)
        b.branch_nz(count, 'loop')
        b.proton_record('outer', start=False)
        b.exit()
        lowered, _ = lower_records(k, buffer_base=0x20000, buffer_bytes=64)
        assemble(lowered)
        ops = k.role(0).region.ops
        ops.insert(3, ops.pop(1))  # start marker now lives inside backedge
        with self.assertRaises(Diagnostic):
            lower_records(k, buffer_base=0x20000, buffer_bytes=64)

    def test_slr_resource_floor_cannot_borrow_remote_capacity(self):
        floors = {'SLR0/DSP': dict(minimum=101, unit='DSP', source='proven implementation minimum')}
        available = {'SLR0/DSP': dict(capacity=100, unit='DSP', source='fixture device spec'),
                     'SLR1/DSP': dict(capacity=1000, unit='DSP', source='fixture device spec')}
        r = implementation_screen(floors, available, identity=IDENTITY)
        self.assertEqual(r['decision'], 'reject-resource-floor')
        self.assertEqual(r['exceeded'], ['SLR0/DSP'])

    def test_existing_r38_interval_uses_actual_denominator(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('r38_costmodel', ROOT/'tools/r38/costmodel/model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model()
        model.coef['mul2'] = [100, 0, 0]
        model.interval = .5
        self.assertEqual(model.predict_interval('mul2', {'n': 16}), (100, 100/1.5, 200))
        model.calibrate([dict(operator='mul2', shape={'n': 16}, cycles=100)]*3)
        self.assertEqual(model.predict_interval('mul2', {'n': 16}), (100, 0, float('inf')))


if __name__ == '__main__':
    unittest.main()
