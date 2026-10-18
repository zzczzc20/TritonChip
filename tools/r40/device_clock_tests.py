#!/usr/bin/env python3
"""Read consistency and unavailable-image checks for the MMIO clock."""
from pathlib import Path
import sys
import threading
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'python'))
from tnca.backend import BackendError
from tnca.backends.u50 import U50Backend
from tnca.errors import DeviceLost
from tnca.device_timing import clock_span, audit_graph_samples, MASK
from tnca.transport import Reply


class ClockTests(unittest.TestCase):
    def backend(self, read):
        backend = object.__new__(U50Backend)
        backend.link = type('Link', (), {'read_reg': staticmethod(read)})()
        backend._io_lock = threading.RLock()
        backend.epoch = 3
        return backend

    def test_old_image_is_unavailable(self):
        self.assertIsNone(self.backend(lambda offset: 0).launch_timestamps(1))

    def test_stable_endpoints_and_epoch(self):
        regs = {0xE0: 1, 0xE8: 91, 0xF0: 140, 0xF8: 7}
        got = self.backend(regs.__getitem__).launch_timestamps(7)
        self.assertEqual(got, dict(start=91, end=140, sequence=7, epoch=3,
                                  clock='mmio-device-clock/1'))

    def test_stale_or_in_progress_sequence_refused(self):
        for sequence in (0, 6):
            regs = {0xE0: 1, 0xE8: 91, 0xF0: 140, 0xF8: sequence}
            with self.assertRaises(BackendError):
                self.backend(regs.__getitem__).launch_timestamps(7)

    def test_torn_read_retried(self):
        seq = iter((6, 7, 7, 7))
        regs = {0xE0: 1, 0xE8: 91, 0xF0: 140}
        def read(offset):
            return next(seq) if offset == 0xF8 else regs[offset]
        self.assertEqual(self.backend(read).launch_timestamps(7)['sequence'], 7)

    def test_unstable_read_reports_device_loss(self):
        seq = iter(range(16))
        def read(offset):
            return next(seq) if offset == 0xF8 else 1
        with self.assertRaises(DeviceLost):
            self.backend(read).launch_timestamps(7)

    def test_unknown_version_refused(self):
        with self.assertRaises(BackendError):
            self.backend(lambda offset: 2).launch_timestamps(7)

    def test_opt_in_launch_captures_before_releasing_io_lock(self):
        regs = {0xE0: 1, 0xE8: 91, 0xF0: 140, 0xF8: 7}
        backend = self.backend(regs.__getitem__)
        backend.local_limit = None
        backend.device_clock = True
        with patch('tnca.transport.Transport.launch', return_value=Reply(7, 0, 40, 5)):
            reply = backend.launch(0, 0)
        self.assertEqual(reply.device_timing['start'], 91)
        backend.device_clock = False
        with patch('tnca.transport.Transport.launch', return_value=Reply(8, 0, 40, 5)):
            self.assertFalse(hasattr(backend.launch(0, 0), 'device_timing'))

    def test_stand_in_propagates_host_wait_budget_to_link(self):
        from tnca.backends.u50_sim import stand_in
        with patch('tnca.backends.u50_sim.SimLink') as link, \
                patch('tnca.backends.u50_sim.U50Backend'):
            stand_in(host_timeout=1234)
            self.assertEqual(link.call_args.kwargs['host_timeout'], 1234)

    def test_complete_unknown_response_is_not_a_host_timeout(self):
        from tnca.backends.u50_sim import SimLink
        with tempfile.TemporaryDirectory() as tmp:
            link = object.__new__(SimLink)
            link.rsp = str(Path(tmp) / 'rsp')
            link.log = str(Path(tmp) / 'log')
            Path(link.rsp).write_text('58\nxxxxxxxxxxxxxxxx\n')
            self.assertIsNone(link._read_rsp())  # in-progress file
            Path(link.rsp).write_text('58\nxxxxxxxxxxxxxxxx\nend\n')
            with self.assertRaises(DeviceLost):
                link._read_rsp()
            Path(link.rsp).write_text('58\n0123456789abcdef\nend\n')
            self.assertEqual(link._read_rsp(), (58, [0x0123456789abcdef]))

    def record(self, start, end, **kw):
        return dict(start=start, end=end, clock='test', epoch=1, session_id=2, **kw)

    def test_graph_span_counts_gaps_and_out_of_order_completion(self):
        records = [self.record(100, 120), self.record(140, 180), self.record(110, 130)]
        self.assertEqual(clock_span(records)['cycles'], 80)

    def test_counter_wrap(self):
        result = clock_span([self.record(MASK - 9, MASK - 1), self.record(3, 12)])
        self.assertEqual(result['cycles'], 22)
        self.assertEqual(result['end'], 12)

    def test_missing_program_invalidates_whole_graph(self):
        self.assertIsNone(clock_span([self.record(1, 5), None]))
        self.assertIsNone(clock_span([]))

    def test_domain_or_epoch_change_refused(self):
        for field in ('clock', 'epoch', 'session_id'):
            other = self.record(6, 10)
            other[field] = 'different'
            with self.assertRaises(ValueError):
                clock_span([self.record(1, 5), other])

    def test_bad_duration_refused(self):
        for end in (1, 0, 1 + (1 << 63)):
            with self.assertRaises(ValueError):
                clock_span([self.record(1, end)])

    def test_archbench_keeps_serial_cycles_separate(self):
        from archbench.platforms.tnca import _timings
        import json
        records = [[dict(self.record(100,120), clock='mmio-device-clock/1'),
                    dict(self.record(140,180), clock='mmio-device-clock/1')],
                   [dict(self.record(200,240), clock='mmio-device-clock/1'),
                    dict(self.record(260,300), clock='mmio-device-clock/1')]]
        steady = [dict(device_cycles=n, request_s=.01, device_timing=clock_span(r), launches=len(r),
                       device_endpoints=r) for n,r in zip((20,30),records)]
        timings = _timings(999, steady, 5, 100_000_000)
        self.assertEqual(timings['device-cycles-serial-grid']['samples'], [20, 30])
        self.assertEqual(timings['device-graph-makespan']['samples'], [80, 100])
        self.assertEqual(timings['device-graph-makespan']['warmup'], 5)
        saved = json.loads(json.dumps(timings))['device-graph-makespan']
        self.assertEqual(saved['endpoint_schema'], 'tnca-launch-endpoints/1')
        self.assertEqual(saved['endpoint_records'], records)
        self.assertEqual([r['cycles'] for r in audit_graph_samples(
            saved['samples'], saved['endpoint_records'], launch_counts=[2,2])], [80,100])

    def test_raw_endpoints_refuse_active_sum_and_corruption(self):
        records = [dict(self.record(100,120), clock='mmio-device-clock/1'),
                   dict(self.record(140,180), clock='mmio-device-clock/1')]
        with self.assertRaisesRegex(ValueError, 'differs'):
            audit_graph_samples([60], [records], launch_counts=[2])  # active sum omits the gap
        for malformed in (None, [], [None], records[:1],
                          [dict(records[0], clock='host-wall-time'), records[1]],
                          [dict(records[0], epoch=2), records[1]],
                          [dict(records[0], start=-1), records[1]],
                          [dict(records[0], start=True), records[1]],
                          [dict(records[0], end=1<<64), records[1]]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                audit_graph_samples([80], [malformed], launch_counts=[2])
        with self.assertRaises(ValueError):
            audit_graph_samples([80,80], [records], launch_counts=[2,2])
        wrap = [dict(self.record(MASK-9,MASK-1), clock='mmio-device-clock/1'),
                dict(self.record(3,12), clock='mmio-device-clock/1')]
        self.assertEqual(audit_graph_samples([22], [wrap], launch_counts=[2])[0]['cycles'], 22)
        # Dropping an interior launch leaves the outer span unchanged.
        three = [records[0], dict(records[0], start=125,end=130), records[1]]
        self.assertEqual(audit_graph_samples([80], [three], launch_counts=[3])[0]['cycles'],80)
        with self.assertRaisesRegex(ValueError, 'launch count'):
            audit_graph_samples([80], [records], launch_counts=[3])

    def test_ablation_recomputes_raw_graph_samples(self):
        import copy
        sys.path.insert(0, str(ROOT/'tools/r40'))
        import ablate
        record = dict(point_id='fixture', time={'launches':2}, timings={'device-graph-makespan':dict(
            unit='cycles', samples=[80], endpoint_schema='tnca-launch-endpoints/1', endpoint_records=[[
                dict(self.record(100,120), clock='mmio-device-clock/1'),
                dict(self.record(140,180), clock='mmio-device-clock/1')]])})
        self.assertEqual(ablate.samples_of(record,'device-graph-makespan')[1], [80])
        self.assertEqual(ablate.samples_of(record,'graph-makespan'), ('graph-makespan',[80]))
        bad = copy.deepcopy(record)
        bad['timings']['device-graph-makespan']['endpoint_records'][0][-1]['end'] += 1
        with self.assertRaises(ablate.Refused):
            ablate.samples_of(bad,'device-graph-makespan')
        del record['timings']['device-graph-makespan']['endpoint_records']
        with self.assertRaises(ablate.Refused):
            ablate.samples_of(record,'device-graph-makespan')

    def test_archbench_missing_endpoint_has_no_makespan_claim(self):
        from archbench.platforms.tnca import _timings
        steady = [dict(device_cycles=20, request_s=.01, device_timing=dict(cycles=80)),
                  dict(device_cycles=30, request_s=.02, device_timing=None)]
        self.assertNotIn('device-graph-makespan', _timings(999, steady, 5, None))


if __name__ == '__main__':
    unittest.main()
