#!/usr/bin/env python3
"""Tests of catalog_check.py on a temporary tree (no repository state is read)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_check as cc


def make_tree(files):
    root = tempfile.mkdtemp()
    for rel in files:
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('#\n')
    return root


def readme(*tokens, outside=()):
    body = '\n'.join('| `%s` | x |' % t for t in tokens)
    extra = '\n'.join('see `%s`' % t for t in outside)
    return 'intro\n%s\n%s\n%s\n%s\n' % (extra, cc.BEGIN, body, cc.END)


class CatalogCheck(unittest.TestCase):
    def test_complete(self):
        root = make_tree(['tools/r41/a.py', 'tools/r41/sub/b.sh', 'archbench/__init__.py'])
        r = cc.check(root, readme('tools/r41/a.py', 'tools/r41/sub/*.sh'))
        self.assertEqual(r['uncovered'], [])
        self.assertEqual(r['stale_catalog_paths'], [])

    def test_uncovered_script_is_reported(self):
        root = make_tree(['tools/r41/a.py', 'tools/r41/new.py'])
        r = cc.check(root, readme('tools/r41/a.py'))
        self.assertEqual(r['uncovered'], ['tools/r41/new.py'])

    def test_stale_path_and_empty_glob(self):
        root = make_tree(['tools/r41/a.py'])
        r = cc.check(root, readme('tools/r41/a.py', 'tools/r41/gone.py', 'tools/models/test_*.py'))
        self.assertEqual(sorted(r['stale_catalog_paths']), ['tools/models/test_*.py', 'tools/r41/gone.py'])

    def test_star_does_not_cross_directories(self):
        root = make_tree(['tools/r41/test_a.py', 'tools/r41/sub/test_b.py'])
        r = cc.check(root, readme('tools/r41/test_*.py'))
        self.assertEqual(r['uncovered'], ['tools/r41/sub/test_b.py'])

    def test_ignore_list(self):
        root = make_tree(['tools/r41/a.py', 'tools/r41/skip.py'])
        r = cc.check(root, readme('tools/r41/a.py'), ignore=('tools/r41/skip.py',))
        self.assertEqual(r['uncovered'], [])

    def test_stale_path_outside_catalog(self):
        root = make_tree(['tools/r41/a.py'])
        r = cc.check(root, readme('tools/r41/a.py', outside=['tools/r41/missing.py']))
        self.assertEqual(r['stale_other_paths'], ['tools/r41/missing.py'])

    def test_missing_region_is_an_error(self):
        root = make_tree(['tools/r41/a.py'])
        with self.assertRaises(ValueError):
            cc.check(root, 'no markers here')

    def test_main_exit_codes(self):
        root = make_tree(['tools/r41/a.py'])
        path = os.path.join(root, 'README.md')
        open(path, 'w').write(readme('tools/r41/a.py'))
        self.assertEqual(cc.main(['--root', root, '--readme', path]), 0)
        open(path, 'w').write(readme('tools/r41/b.py'))
        self.assertEqual(cc.main(['--root', root, '--readme', path]), 1)
        self.assertEqual(cc.main(['--root', root, '--readme', os.path.join(root, 'nope.md')]), 2)


if __name__ == '__main__':
    unittest.main()
