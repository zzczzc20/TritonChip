#!/usr/bin/env python3
"""Legacy/v2 identity and opcode boundaries before hardware side effects."""
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tnca.backends.u50 import U50Backend, MAGIC, R_REQ_OP, R_DOORBELL
from tnca.backends.multicore_mmio import MulticoreMMIOBackend, MulticoreXDMALink
from tnca.errors import ArgumentError, DeviceLost, TncaError, DependencyFailed


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
                    {'memory_ports': 4}, {'core_count':8,'memory_ports':4}, {'core_count': 1.5}):
        refused(lambda opts=options: MulticoreMMIOBackend(Link(), **opts), (ArgumentError, TypeError))
    assert MulticoreMMIOBackend(Link(), image_profile=None).image_profile == 'core-vrf-sync-v5'
    allocation_link = MulticoreXDMALink(xclbin='unopened-test.xclbin')
    class Allocator:
        def __init__(self): self.calls=[]
        def bo_open(self, size, bank): self.calls.append((size,bank));return 0x100000000
    allocator=Allocator();allocation_link._accessor=allocator
    shared=dict(index=32,used=True,size=4096,base=0,tag='HBM[0:31]')
    with patch('tnca.backends.multicore_mmio._xclbin_bank',side_effect=[shared,shared,{**shared,'index':33}]):
        refused(lambda: allocation_link.hbm_open(4096),DependencyFailed)
    assert not allocator.calls
    with patch('tnca.backends.multicore_mmio._xclbin_bank',return_value=shared), patch('tnca.backends.u50._xclbin_bank',return_value=shared):
        assert allocation_link.hbm_open(8192)==(0x100000000,4096)
    assert allocator.calls==[(4096,32)]
    print('MULTICORE MMIO OPTIONS PASS 13 boundaries', flush=True)


if __name__ == '__main__':
    main()
