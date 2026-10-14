#!/usr/bin/env python3
"""Check execution sharing and failure propagation without board dependencies."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import acceptance


class RunnerTests(unittest.TestCase):
    def run_plan(self, returncode=0, execute=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.json'
            gates = [dict(id=f'R40-05.{name}', kind='capability', required=True,
                          command='run-board-suite', evidence=str(evidence))
                     for name in ('multicore', 'dispatch')]
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(work_packages=[
                dict(id='R40-05', gates=gates)])))
            report = root / 'report.json'

            def run(*args, **kwargs):
                # Evidence must be produced by the execution, not pre-seeded.
                evidence.write_text(json.dumps(dict(gates={g['id']: dict(
                    status='capability-pass', checks=[dict(name='board', ok=True)])
                    for g in gates})))
                return SimpleNamespace(returncode=returncode)

            with patch.object(acceptance.subprocess, 'run', side_effect=run) as call:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = acceptance.main(['--plan', str(plan), '--json', str(report)]
                                             + (['--run'] if execute else []))
                return status, call.call_count, json.loads(report.read_text())

    def test_one_execution_produces_multiple_gates(self):
        status, calls, report = self.run_plan()
        self.assertEqual((status, calls), (0, 1))
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['work_packages']['R40-05']['gates']), 2)

    def test_failed_execution_invalidates_every_gate(self):
        status, calls, report = self.run_plan(returncode=7)
        self.assertEqual((status, calls), (1, 1))
        self.assertTrue(all(g['status'] == 'fail' for g in
                            report['work_packages']['R40-05']['gates']))

    def test_read_only_does_not_execute_or_fill_missing_evidence(self):
        status, calls, report = self.run_plan(execute=False)
        self.assertEqual((status, calls), (1, 0))
        self.assertTrue(all(g['status'] == 'not-measured' for g in
                            report['work_packages']['R40-05']['gates']))

    def test_unknown_selection_cannot_pass_vacuously(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                acceptance.main(['--wp', 'R40-typo'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
