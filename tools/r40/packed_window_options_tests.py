#!/usr/bin/env python3
"""AOT cache identity includes resolved allocation/unpack environment flags."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
from tnca.aot import check_options, cache_key


class OptionTests(unittest.TestCase):
    switches = dict(packed_window_unpack='TNCA_PACKED_WINDOW_UNPACK',
                    window_reuse='TNCA_WINDOW_REUSE', bank_alloc='TNCA_BANK_ALLOC',
                    index_opt='TNCA_INDEX_OPT', sparse_tensormaps='TNCA_SPARSE_TENSORMAPS',
                    schedule='TNCA_SCHEDULE', rctx_spmd='TNCA_RCTX_SPMD', rctx_roles='TNCA_RCTX_ROLES')

    def key(self, options):
        return cache_key(source_kind='triton', source_hash='test', kernel_name='test',
                         signature=[], constexprs={}, profile='core-sim',
                         config_hash='test', options=options)

    def test_default_off_remains_absent(self):
        with patch.dict(os.environ, {env: '0' for env in self.switches.values()}):
            self.assertEqual(check_options({}), {})

    def test_each_environment_switch_changes_aot_cache_key(self):
        for name, env in self.switches.items():
            baseline = {e: '0' for e in self.switches.values()}
            with patch.dict(os.environ, baseline):
                off = self.key(check_options({}))
            with patch.dict(os.environ, dict(baseline, **{env: '1'})):
                resolved = check_options({})
                self.assertEqual(resolved, {name: 1})
                self.assertNotEqual(off, self.key(resolved))
                self.assertEqual(check_options({name: 0}), {name: 0})

    def test_invalid_environment_or_explicit_value_refused(self):
        for name, env in self.switches.items():
            invalid = 3 if name == 'rctx_roles' else 2
            baseline = {e: '0' for e in self.switches.values()}
            with patch.dict(os.environ, dict(baseline, **{env: str(invalid)})):
                with self.assertRaises(ValueError):
                    check_options({})
            with patch.dict(os.environ, baseline):
                with self.assertRaises(ValueError):
                    check_options({name: invalid})

    def test_forced_role_partition_has_its_own_cache_identity(self):
        baseline = {e: '0' for e in self.switches.values()}
        keys = []
        for mode in (0, 1, 2):
            with patch.dict(os.environ, dict(baseline, TNCA_RCTX_ROLES=str(mode))):
                keys.append(self.key(check_options({})))
                self.assertEqual(check_options({'rctx_roles': 0}), {'rctx_roles': 0})
        self.assertEqual(len(set(keys)), 3)


if __name__ == '__main__':
    unittest.main()
