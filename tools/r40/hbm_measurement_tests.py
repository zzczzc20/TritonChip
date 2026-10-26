#!/usr/bin/env python3
"""Counter units, missing data and CPU validation ordering; requires Triton.

These collector checks are not board execution or performance proof.
"""
import contextlib
import sys
from pathlib import Path
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(Path(__file__).resolve().parent)]
import hbm_board as board


class Tests(unittest.TestCase):
    def test_outlet_lines_are_64_bytes_not_legacy_beats(self):
        a=dict(kind='outlet-lines',ports=[dict(port=0,read_lines=5,write_lines=8)])
        b=dict(kind='outlet-lines',ports=[dict(port=0,read_lines=8,write_lines=10)])
        delta=board._delta(a,b)
        self.assertEqual((delta['read_bytes'],delta['write_bytes']),(192,128))
        self.assertIsNone(delta['mean_read_latency'])
        self.assertIsNone(board._delta(None,b))
        self.assertIsNone(board._rate(100,None))
        with self.assertRaises(ValueError):board._delta(b,a)
        with self.assertRaises(ValueError):board._delta(a,dict(kind='outlet-lines',ports=[]))

    def test_legacy_units_and_unsupported_latency(self):
        a=dict(kind='legacy-axi',counters=dict(reads=0,writes=0,read_cycles=0,read_beats=0))
        b=dict(kind='legacy-axi',counters=dict(reads=3,writes=2,read_cycles=60,read_beats=24))
        d=board._delta(a,b)
        self.assertEqual(d['read_bytes'],192)
        self.assertEqual(d['mean_read_latency'],20)
        self.assertIsNone(d['write_bytes'])
        self.assertIsNone(board._delta(a,a)['mean_read_latency'])

    def test_native_interface_never_reads_legacy_registers(self):
        class Device:
            def capabilities(self):return dict(physical_grid=True)
            def memory_port_metrics(self):return [dict(port=0,read_lines=7,write_lines=9)]
            def axi_counters(self):raise AssertionError('unimplemented legacy counters read')
        class Session:dev=Device()
        with patch.object(board,'_session',return_value=Session()):
            self.assertEqual(board._counters()['kind'],'outlet-lines')

    def test_completed_measurement_does_not_read_output_on_cpu(self):
        # Verify collection boundaries only. Actual event endpoints come from
        # runtime integration tests; this fixture is not execution evidence.
        import triton  # Finish plugin discovery before importing its driver.
        from tnca.triton_backend import driver
        class Torch:no_grad=staticmethod(contextlib.nullcontext)
        class Output:
            def cpu(self):raise AssertionError('CPU handoff contaminated device traffic')
        stamp=lambda start,end:dict(clock='fixture',epoch=0,session_id=1,start=start,end=end)
        rows=[dict(device_cycles=10,device_timing=stamp(100,110)),
              dict(device_cycles=10,device_timing=stamp(120,130))]
        @contextlib.contextmanager
        def log():yield rows
        with patch.object(board,'torch',Torch,create=True),patch.object(driver,'launch_log',log):
            _,m=board._launch_cycles(Output)
            self.assertEqual(m['device_cycles'],30)  # common span, not active sum
            self.assertEqual(m['summed_kernel_cycles'],20)
            rows[1]['device_timing']=None
            self.assertIsNone(board._launch_cycles(Output)[1]['device_cycles'])


if __name__=='__main__':unittest.main()
