#!/usr/bin/env python3
"""Count real loader calls, not artifact aliases, without opening a device."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import triton  # Discover the TNCA plugin before importing its driver.
from tnca.errors import BufferError
from tnca.triton_backend import driver


class FakeSession:
    def __init__(self, capacity=1):
        self.id, self.epoch = 301, 0
        self.capacity, self.require_empty = capacity, set()
        self.resident = []

    def load(self, path):
        if len(self.resident) >= self.capacity or (path in self.require_empty and self.resident):
            raise BufferError('overlaps the device heap')
        module = types.SimpleNamespace(path=path)
        self.resident.append(module)
        return module

    def unload(self, module):
        self.resident.remove(module)

    def drain(self):
        pass


class CodeLoadTests(unittest.TestCase):
    def test_reuse_and_aliases_do_not_count_as_loads(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(driver, '_loaded', {}), \
                mock.patch.object(driver, '_loaded_by_artifact', {}), \
                mock.patch.object(driver, '_artifact_hashes', {}), \
                mock.patch.object(driver, 'loaded', 0), mock.patch.object(driver, 'evicted', 0):
            a, alias, b = [Path(directory) / name for name in ('a', 'alias', 'b')]
            a.write_text('a'); alias.write_text('a'); b.write_text('b')
            session = FakeSession()
            first = driver._load(session, str(a))[1]
            self.assertIs(first, driver._load(session, str(alias))[1])
            self.assertIs(first, driver._load(session, str(a))[1])
            self.assertEqual(driver.loaded, 1)
            driver._load(session, str(b))
            self.assertEqual((driver.loaded, driver.evicted), (2, 1))
            driver._load(session, str(a))
            self.assertEqual((driver.loaded, driver.evicted), (3, 2))

    def test_eviction_skips_deleted_aliases_when_multiple_modules_must_leave(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(driver, '_loaded', {}), \
                mock.patch.object(driver, '_loaded_by_artifact', {}), \
                mock.patch.object(driver, '_artifact_hashes', {}), \
                mock.patch.object(driver, 'loaded', 0), mock.patch.object(driver, 'evicted', 0):
            a, alias, b, c = [Path(directory) / name for name in ('a', 'alias', 'b', 'c')]
            a.write_text('a'); alias.write_text('a'); b.write_text('b'); c.write_text('c')
            session = FakeSession(capacity=2)
            session.require_empty.add(str(c))
            driver._load(session, str(a))
            driver._load(session, str(alias))
            driver._load(session, str(b))
            driver._load(session, str(c))
            self.assertEqual((driver.loaded, driver.evicted), (3, 2))
            self.assertEqual([module.path for module in session.resident], [str(c)])


if __name__ == '__main__':
    unittest.main()
