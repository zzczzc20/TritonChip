#!/usr/bin/env python3
"""Code-residency eviction and aliasing without opening a device session."""
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import triton  # Discover plugins before importing the TNCA driver directly.
from tnca.errors import BufferError
from tnca.triton_backend import driver


class FakeSession:
    def __init__(self):
        self.id, self.epoch = 501, 0
        self.resident, self.unloaded = [], []
        self.drains = 0

    def load(self, path):
        if len(self.resident) == 2:
            raise BufferError('overlaps the device heap')
        module = types.SimpleNamespace(path=path, code_size=64)
        self.resident.append(module)
        return module

    def unload(self, module):
        self.resident.remove(module)
        self.unloaded.append(module.path)

    def drain(self):
        self.drains += 1


class CodeCacheTests(unittest.TestCase):
    def setUp(self):
        self.loaded = mock.patch.object(driver, '_loaded', {})
        self.by_artifact = mock.patch.object(driver, '_loaded_by_artifact', {})
        self.hashes = mock.patch.object(driver, '_artifact_hashes', {})
        for patcher in (self.loaded, self.by_artifact, self.hashes):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_frequency_keeps_hot_kernel_and_cleans_aliases(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'TNCA_CODE_CACHE_POLICY': 'lfu-sqrt'}):
            paths = [Path(directory) / name for name in ('a', 'alias', 'b', 'c')]
            for path, contents in zip(paths, ('a', 'a', 'b', 'c')):
                path.write_text(contents)
            a, alias, b, c = map(str, paths)
            session = FakeSession()
            _, first = driver._load(session, a)
            _, shared = driver._load(session, alias)
            self.assertIs(first, shared)
            driver._load(session, b)
            driver._load(session, a)
            driver._load(session, c)
            self.assertEqual(session.unloaded, [b])
            self.assertEqual(session.drains, 1)
            self.assertIs(driver._load(session, alias)[1], first)
            self.assertEqual(len(session.resident), 2)

    def test_epoch_change_forgets_frequency(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'TNCA_CODE_CACHE_POLICY': 'lfu-fourth'}):
            path = Path(directory) / 'a'
            path.write_text('a')
            session = FakeSession()
            driver._load(session, str(path))
            session.epoch = 1
            session.resident.clear()
            driver._load(session, str(path))
            state = driver._cache_frequency[session]
            self.assertEqual(state['epoch'], 1)
            self.assertEqual(sum(state['counts'].values()), 1)


if __name__ == '__main__':
    unittest.main()
