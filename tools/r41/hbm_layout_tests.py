#!/usr/bin/env python3
"""Host DMA/RTL address-layout agreement and capability refusal boundaries."""
from pathlib import Path
import copy
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.hbm_layout import StripedHBMLayout, STRIPED_BYTES, STRIPED_SIGNATURE
from tnca.backends import multicore_mmio as mmio, u50
from tnca.errors import ArgumentError, DependencyFailed, DeviceLost


def oracle(offset):
    page, byte = divmod(offset, 4096)
    local_page, pc = divmod(page, 32)
    return pc * (1 << 28) + local_page * 4096 + byte


def topology():
    return dict(physical_banks=[dict(pc=i, base=i << 28, size=1 << 28, used=True,
                allocated_base=i << 28, allocated_size=1 << 28) for i in range(32)])


class Accessor:
    def __init__(self, *, striped=True, base=0, signature=STRIPED_SIGNATURE):
        self.striped, self.base, self.signature = striped, base, signature
        self.pages, self.calls, self.allocations = {}, [], []

    def read(self, offset):
        if offset == 0xb8:
            return 64 if self.striped else 0
        if offset == 0x120 and self.striped:
            return self.signature
        raise AssertionError('unexpected register read')

    def bo_open(self, size, bank):
        self.allocations.append((size, bank))
        return self.base

    def bo_write(self, offset, data):
        self.calls.append(('write', offset, len(data)))
        done = 0
        while done < len(data):
            page, byte = divmod(offset+done, 4096)
            n = min(len(data)-done, 4096-byte)
            self.pages.setdefault(page, bytearray(4096))[byte:byte+n] = data[done:done+n]
            done += n

    def bo_read(self, offset, size):
        self.calls.append(('read', offset, size))
        data = bytearray()
        while len(data) < size:
            page, byte = divmod(offset+len(data), 4096)
            n = min(size-len(data), 4096-byte)
            data += self.pages.get(page, bytes(4096))[byte:byte+n]
        return bytes(data)


class LayoutTests(unittest.TestCase):
    def test_bijection_and_pc_assignment(self):
        randomizer = random.Random(41)
        offsets = [0, 4095, 4096, (1 << 17)-1, 1 << 17, STRIPED_BYTES-1]
        offsets += [randomizer.randrange(STRIPED_BYTES) for _ in range(2000)]
        for offset in offsets:
            actual = StripedHBMLayout.physical(offset)
            self.assertEqual(actual, oracle(offset))
            self.assertEqual(StripedHBMLayout.logical(actual), offset)
        self.assertEqual({StripedHBMLayout.physical(pc*4096) >> 28 for pc in range(32)}, set(range(32)))

    def test_coalesced_dma_against_independent_byte_oracle(self):
        layout = StripedHBMLayout(0, STRIPED_BYTES)
        for start, size in [(0, 0), (4093, 9001), (31*4096+4090, 270001),
                            ((1 << 32)-37, 170003), (STRIPED_BYTES-9000, 9000)]:
            with self.subTest(start=start, size=size):
                accessor = Accessor()
                data = random.Random(start).randbytes(size)
                layout.write(accessor, start, data)
                self.assertLessEqual(len(accessor.calls), 32)
                for i in range(0, size, 97):
                    physical = oracle(start+i)
                    self.assertEqual(accessor.pages[physical//4096][physical % 4096], data[i])
                self.assertEqual(layout.read(accessor, start, size), data)
                self.assertEqual(layout.read(accessor, start+size, 0), b'')

    def test_invalid_allocation_and_ranges_have_no_dma_side_effects(self):
        for base, size in [(1, STRIPED_BYTES), (0, 1 << 32), (1 << 28, STRIPED_BYTES)]:
            with self.assertRaises(ValueError):
                StripedHBMLayout(base, size)
        layout = StripedHBMLayout(0, STRIPED_BYTES)
        accessor = Accessor()
        for start, size in [(-1, 2), (0, -1), (STRIPED_BYTES, 1), (STRIPED_BYTES+1, 0)]:
            with self.assertRaises(ValueError):
                layout.read(accessor, start, size)
        self.assertEqual(accessor.calls, [])

    def open_link(self, accessor, *, size=STRIPED_BYTES, mapping=None):
        link = mmio.MulticoreXDMALink(xclbin='fixture', accessor=accessor)
        bank = dict(index=32, used=True, size=STRIPED_BYTES)
        with patch.object(mmio, '_xclbin_bank', return_value=bank), \
             patch.object(u50, '_xclbin_bank', return_value=bank), \
             patch.object(u50, '_xclbin_hbm_allocation', return_value=mapping or topology()):
            link.hbm_open(size)
        return link

    def test_failed_allocation_cannot_fall_back_to_linear_dma(self):
        accessor = Accessor(base=1 << 28)
        link = mmio.MulticoreXDMALink(xclbin='fixture', accessor=accessor)
        bank = dict(index=32, used=True, size=STRIPED_BYTES)
        with patch.object(mmio, '_xclbin_bank', return_value=bank), \
             patch.object(u50, '_xclbin_bank', return_value=bank), \
             patch.object(u50, '_xclbin_hbm_allocation', return_value=topology()):
            with self.assertRaises(DependencyFailed):
                link.hbm_open(STRIPED_BYTES)
        with self.assertRaises(DependencyFailed):
            link.hbm_write(0, b'blocked')
        with self.assertRaises(DependencyFailed):
            link.hbm_read(0, 7)
        self.assertEqual(accessor.calls, [])

    def test_image_capability_selects_real_link_dma_permutation(self):
        accessor = Accessor()
        link = self.open_link(accessor)
        payload = bytes(range(256))*37
        link.hbm_write(8190, payload)
        self.assertEqual(link.hbm_read(8190, len(payload)), payload)
        self.assertEqual(link.hbm_allocation['layout'], StripedHBMLayout.name)
        self.assertIn(('write', oracle(8190), 2), accessor.calls)

    def test_legacy_link_keeps_linear_dma_and_never_reads_layout_register(self):
        accessor = Accessor(striped=False, base=1 << 32)
        link = self.open_link(accessor, size=1 << 30)
        self.assertIsNone(link.hbm_layout)
        link.hbm_write(4097, b'legacy')
        self.assertEqual(link.hbm_read(4097, 6), b'legacy')
        self.assertEqual(accessor.calls, [('write', 4097, 6), ('read', 4097, 6)])

    def test_bad_capability_or_size_refused_before_allocation(self):
        for accessor, size, error in [(Accessor(signature=0), STRIPED_BYTES, DeviceLost),
                                      (Accessor(), 1 << 32, ArgumentError)]:
            with self.assertRaises(error):
                self.open_link(accessor, size=size)
            self.assertEqual(accessor.allocations, [])

    def test_oversized_request_is_clamped_to_complete_striped_group(self):
        accessor = Accessor()
        link = self.open_link(accessor, size=2 * STRIPED_BYTES)
        self.assertEqual(accessor.allocations, [(STRIPED_BYTES, 32)])
        self.assertEqual(link.hbm_allocation['requested_bytes'], 2 * STRIPED_BYTES)
        self.assertEqual(link.hbm_allocation['exposed_bytes'], STRIPED_BYTES)
        self.assertEqual(link.hbm_allocation['layout'], StripedHBMLayout.name)

    def test_nonzero_base_and_incorrect_topology_refused(self):
        with self.assertRaises(DependencyFailed):
            self.open_link(Accessor(base=1 << 28))
        for change in ('missing', 'tag', 'unused', 'offset', 'size'):
            mapping = copy.deepcopy(topology())
            if change == 'missing': mapping['physical_banks'].pop()
            if change == 'tag': mapping['physical_banks'][0]['pc'] = 31
            if change == 'unused': mapping['physical_banks'][0]['used'] = False
            if change == 'offset': mapping['physical_banks'][0]['base'] = 1
            if change == 'size': mapping['physical_banks'][0]['size'] = 4096
            with self.subTest(change=change), self.assertRaises(DependencyFailed):
                self.open_link(Accessor(), mapping=mapping)

    def test_rtl_adapter_and_readonly_capability(self):
        offsets = {0, 64, STRIPED_BYTES-64, (1 << 32)-64, 1 << 32}
        for pc in range(32):
            offsets.update([pc*4096, pc*4096+4032, (pc+32)*4096])
        offsets = sorted(offsets)
        with tempfile.TemporaryDirectory(prefix='tnca-hbm-layout-') as temp:
            vectors = Path(temp)/'vectors.hex'
            vectors.write_text(''.join(f'{offset:016x}{oracle(offset):016x}\n' for offset in offsets))
            for bad in range(3):
                binary = str(Path(temp)/'layout.vvp')
                subprocess.run(['iverilog', '-g2012', '-s', 'tb_hbm_stripe',
                    f'-Ptb_hbm_stripe.BAD_WINDOW={bad}', '-o', binary,
                    'rtl/fpga/tnca_line_axi.v', 'tb/fpga/tb_hbm_stripe.v'], cwd=ROOT, check=True, timeout=60)
                subprocess.run(['vvp', binary, f'+vectors={vectors}', f'+count={len(offsets)}'],
                               cwd=ROOT, check=True, timeout=60)
            for enabled in (0, 1):
                for striped in (0, 1):
                    binary = str(Path(temp)/'mmio.vvp')
                    subprocess.run(['iverilog', '-g2012', '-s', 'tb_mmio_memory_controls',
                        f'-Ptb_mmio_memory_controls.ENABLE={enabled}',
                        f'-Ptb_mmio_memory_controls.HBM_STRIPED={striped}', '-o', binary,
                        'rtl/fpga/tnca_multicore_mmio.v', 'tb/fpga/tb_mmio_memory_controls.v'],
                        cwd=ROOT, check=True, timeout=60)
                    subprocess.run(['vvp', binary], cwd=ROOT, check=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
