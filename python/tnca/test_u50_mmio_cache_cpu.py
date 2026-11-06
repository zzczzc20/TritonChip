"""CPU checks of exclusive-context request-register reuse."""
import os
import unittest
from unittest import mock
from tnca.backends.u50 import (R_DOORBELL, R_REQ_A1, R_REQ_SEQ, U50Backend,
                               XDMALink)


class RegisterFile:
    def __init__(self):
        self.values = {}
        self.writes = []
        self.requests = []
        self.fail = None
    def write(self, offset, value):
        if offset == self.fail:
            raise OSError('simulated register failure')
        self.values[offset] = value
        self.writes.append((offset, value))
        if offset == R_DOORBELL:
            self.requests.append(dict(self.values))
    def close(self):
        pass


class RegisterCacheTests(unittest.TestCase):
    def link(self, access, *, exclusive):
        link = XDMALink(accessor=None if exclusive else access)
        if exclusive:
            link._open_access = lambda: access
        link.open()
        return link

    def test_same_requests_emit_same_registers_with_fewer_writes(self):
        control, optimized = RegisterFile(), RegisterFile()
        with mock.patch.dict(os.environ, {'TNCA_MMIO_DESCRIPTOR_CACHE': '1'}):
            a, b = self.link(control, exclusive=False), self.link(optimized, exclusive=True)
        baseline, candidate = U50Backend(a), U50Backend(b)
        for i in range(20):
            args = (i + 1, 7 if i < 10 else 8, 0x2000, 0x4000 + (i % 2) * 64,
                    1, 2, 3, 4)
            baseline._put_request(*args)
            candidate._put_request(*args)
        self.assertEqual(control.requests, optimized.requests)
        self.assertEqual(sum(offset == R_REQ_SEQ for offset, _ in optimized.writes), 20)
        self.assertEqual(sum(offset == R_DOORBELL for offset, _ in optimized.writes), 20)
        self.assertLess(len(optimized.writes), len(control.writes))
        self.assertGreater(b.request_register_reuses, 0)

    def test_write_failure_invalidates_all_cached_registers(self):
        registers = RegisterFile()
        with mock.patch.dict(os.environ, {'TNCA_MMIO_DESCRIPTOR_CACHE': '1'}):
            link = self.link(registers, exclusive=True)
        backend = U50Backend(link)
        backend._put_request(1, 7, 8, 9, 10, 11, 12, 13)
        registers.fail = R_REQ_A1
        with self.assertRaises(OSError):
            backend._put_request(2, 7, 8, 99, 10, 11, 12, 13)
        self.assertEqual(link._request_register_shadow, {})
        registers.fail = None
        count = len(registers.writes)
        backend._put_request(3, 7, 8, 9, 10, 11, 12, 13)
        self.assertEqual(len(registers.writes) - count, 10)
        self.assertEqual(registers.requests[-1][R_REQ_SEQ], 3)

    def test_reopen_and_explicit_disable_do_not_reuse_stale_values(self):
        registers = RegisterFile()
        with mock.patch.dict(os.environ, {'TNCA_MMIO_DESCRIPTOR_CACHE': '1'}):
            link = self.link(registers, exclusive=True)
            link.write_reg(R_REQ_A1, 9)
            link.close()
            link.open()
            link.write_reg(R_REQ_A1, 9)
        self.assertEqual(registers.writes.count((R_REQ_A1, 9)), 2)
        disabled = RegisterFile()
        with mock.patch.dict(os.environ, {'TNCA_MMIO_DESCRIPTOR_CACHE': '0'}):
            link = self.link(disabled, exclusive=True)
            link.write_reg(R_REQ_A1, 9)
            link.write_reg(R_REQ_A1, 9)
        self.assertEqual(disabled.writes.count((R_REQ_A1, 9)), 2)


if __name__ == '__main__':
    unittest.main()
