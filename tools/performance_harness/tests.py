#!/usr/bin/env python3
"""Replay preserved real RTL bundles; corrupted trace/cycles/code must refuse."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('performance_inspector', HERE/'audit.py')
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)
ARCHIVE = HERE.parents[1]/'evidence/r40/R40-09/performance_tools/raw.tar.gz'


class Tests(unittest.TestCase):
    def bundle(self, directory, name='fine04'):
        prefix = 'r40_proton_capture_' + name + '/'
        with tarfile.open(ARCHIVE) as archive:
            for leaf in ('report.json', 'runs.json', 'trace.chrome_trace', 'capture.bin'):
                (Path(directory)/leaf).write_bytes(archive.extractfile(prefix+leaf).read())
        return Path(directory)/'report.json'

    def test_preserved_real_bundles(self):
        for name in ('fine04', 'coarse03', 'loop_short04'):
            with tempfile.TemporaryDirectory() as temp:
                result = inspector.inspect_report(self.bundle(temp, name))
                self.assertEqual(result['status'], 'audited-diagnostic')
                self.assertEqual(result['performance_gate'], 'not-measured')
                self.assertIsNone(result['region_timing_error_bound_cycles'])
                self.assertEqual(result['overhead']['pair_count'], 6)

    def test_corruptions_are_refused(self):
        for mutation in ('trace', 'cycles', 'code', 'capture', 'incomplete'):
            with tempfile.TemporaryDirectory() as temp:
                report_path = self.bundle(temp)
                report = json.loads(report_path.read_text())
                if mutation == 'trace':
                    path = Path(temp)/'trace.chrome_trace'
                    trace = json.loads(path.read_text())
                    next(e for e in trace['traceEvents'] if e['ph'] == 'X')['dur'] += 1
                    path.write_text(json.dumps(trace))
                elif mutation == 'cycles':
                    report['overhead']['pairs'][0]['baseline_cycles'] += 1
                elif mutation == 'code':
                    report['code_sha256']['baseline'] = 'f'*64
                elif mutation == 'capture':
                    path = Path(temp)/'capture.bin'
                    raw = bytearray(path.read_bytes()); raw[0] = 0
                    path.write_bytes(raw)
                else:
                    report['status'] = 'failed'
                report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError, msg=mutation):
                    inspector.inspect_report(report_path)


if __name__ == '__main__':
    unittest.main()
