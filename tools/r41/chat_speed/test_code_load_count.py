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
        module = types.SimpleNamespace(path=path, params=[('x', 'i32')], epoch=self.epoch,
                                       alive=True, code_addr=0, code_size=64)
        self.resident.append(module)
        return module

    def unload(self, module):
        module.alive = False
        self.resident.remove(module)

    def drain(self):
        pass


class CodeLoadTests(unittest.TestCase):
    def test_verified_equivalent_code_shares_allocation_and_keeps_provenance(self):
        from contextlib import ExitStack
        from tnca import artifact

        def manifest(path):
            name = Path(path).name
            return dict(identity=dict(kernel_name='matmul', abi_version=[1, 0], source_hash=name),
                        code=dict(segments=[dict(name='text', sha256='identical-code')]),
                        parameters=dict(signature=['different' if name == 'c' else 'x']),
                        resources=dict(scratch_bytes=0), execution=dict(roles=1),
                        optional_metadata=dict(provenance=dict(source=name)), manifest_hash=name)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ('_loaded', '_loaded_by_artifact', '_loaded_by_executable',
                         '_module_views', '_artifact_hashes', '_executable_hashes'):
                stack.enter_context(mock.patch.object(driver, name, {}))
            stack.enter_context(mock.patch.object(driver, 'loaded', 0))
            stack.enter_context(mock.patch.object(driver, 'evicted', 0))
            stack.enter_context(mock.patch.object(artifact, 'load',
                                  side_effect=lambda path: (manifest(path), {'text': b'same'})))
            a, b, c = [Path(directory) / name for name in ('a', 'b', 'c')]
            for path in (a, b, c):
                path.write_text(path.name)
            session = FakeSession()
            original = driver._load(session, str(a))[1]
            view = driver._load(session, str(b))[1]
            self.assertEqual(driver.loaded, 1)
            self.assertEqual(view.code_addr, original.code_addr)
            self.assertEqual(view.path, str(b))
            self.assertEqual(view.manifest['identity']['source_hash'], 'b')
            self.assertIs(driver._load(session, str(b))[1], view)
            driver._load(session, str(c))  # Different argument contract cannot alias.
            self.assertFalse(view.alive)
            self.assertEqual((driver.loaded, driver.evicted), (2, 1))
            driver._load(session, str(b))
            self.assertEqual(driver.loaded, 3)

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
