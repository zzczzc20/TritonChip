#!/usr/bin/env python3
"""Reject altered, empty and duplicate source archives."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from audit_xo_sources import audit, ROOT


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.xo = Path(self.temp.name) / 'candidate.xo'
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                              text=True).strip()
        self.path = 'rtl/common/tnca_defs.vh'
        self.data = subprocess.check_output(['git', 'show', self.commit + ':' + self.path], cwd=ROOT)
        self.name = 'ip_repo/kernel/src/tnca_defs.vh'

    def test_one_changed_byte_cannot_pass_source_identity(self):
        with zipfile.ZipFile(self.xo, 'w') as package:
            package.writestr(self.name, self.data)
        self.assertTrue(audit(self.xo, self.commit)['passed'])
        with zipfile.ZipFile(self.xo, 'w') as package:
            package.writestr(self.name, self.data + b'\n// changed\n')
        result = audit(self.xo, self.commit)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['sources'][0]['source'])

    def test_empty_archive_does_not_vacuously_pass(self):
        with zipfile.ZipFile(self.xo, 'w'):
            pass
        with self.assertRaisesRegex(ValueError, 'no packaged'):
            audit(self.xo, self.commit)

    def test_duplicate_archive_entries_refused(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.xo, 'w') as package:
                package.writestr(self.name, self.data)
                package.writestr(self.name, self.data + b'bad')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit(self.xo, self.commit)


if __name__ == '__main__':
    unittest.main()
