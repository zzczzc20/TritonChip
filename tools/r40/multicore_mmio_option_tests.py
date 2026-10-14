#!/usr/bin/env python3
"""Legacy/v2 identity and opcode boundaries before hardware side effects."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tnca.backends.u50 import U50Backend, MAGIC, R_REQ_OP, R_DOORBELL
from tnca.backends.multicore_mmio import MulticoreMMIOBackend
from tnca.errors import ArgumentError, DeviceLost, TncaError


class Link:
    def __init__(self):
        self.writes = []
        self.closed = False

    def open(self):
        return self

    def close(self):
        self.closed = True

    def read_reg(self, address):
        return MAGIC

    def write_reg(self, address, value):
        self.writes.append((address, value))

    def describe(self):
        return 'boundary-only register stub'


def refused(fn, exception):
    try:
        fn()
    except exception:
        return
    raise AssertionError('operation was not refused')


def main():
    link = Link()
    legacy = U50Backend(link)
    refused(lambda: legacy._put_request(1, 17, 0, 0, 0, 0, 0, 0), TncaError)
    assert not link.writes
    newer = MulticoreMMIOBackend(link)
    newer._put_request(1, 17, 0, 0, 0, 0, 0, 0)
    assert (R_REQ_OP, 17) in link.writes and link.writes[-1] == (R_DOORBELL, 1)
    refused(lambda: newer._put_request(2, 256, 0, 0, 0, 0, 0, 0), TncaError)
    other = Link()
    refused(lambda: MulticoreMMIOBackend(other).start(), DeviceLost)
    assert other.closed and not other.writes  # legacy magic cannot receive v2 writes
    for options in ({'hbm_dma': True}, {'core_count': 0}, {'core_count': 17},
                    {'memory_ports': 4}, {'core_count': 1.5}):
        refused(lambda opts=options: MulticoreMMIOBackend(Link(), **opts), (ArgumentError, TypeError))
    assert MulticoreMMIOBackend(Link(), image_profile=None).image_profile == 'core-vrf-sync-v5'
    print('MULTICORE MMIO OPTIONS PASS 10 boundaries', flush=True)


if __name__ == '__main__':
    main()
