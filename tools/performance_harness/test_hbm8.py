#!/usr/bin/env python3
"""R41 hbm8: runtime/backend readiness for an 8 GiB window, without a card.

The B40 xclbin's real HBM topology (evidence/r41/hbm8/xclbin_hbm_map.json,
parsed read-only from the image file) drives the production XRT link code
with a recording BO accessor, and the production U50/multicore backends run
over a register stand-in. Checks: allocation in the HBM[0:31] group, PC0-PC31
mapping, the 16 GiB clamp, window registers and heap, the reserved TMU page,
DMA offsets above 4 GiB (no high-bit truncation), host-side and allocator
refusals, and that the registered board manifests are the generated ones.
"""
import contextlib
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'tools/r40'), str(HERE)]
from tnca.backends import u50, multicore_mmio                        # noqa: E402
from tnca.errors import ArgumentError, BufferError, DependencyFailed  # noqa: E402
from tnca.runtime import _Allocator, DEVICE_RESERVED_RANGES          # noqa: E402
import experiment                                                    # noqa: E402
import hbm8_case                                                     # noqa: E402
import hbm8_manifest                                                 # noqa: E402
import hbm8_summary                                                  # noqa: E402

GIB = 1 << 30
MAP = json.loads((ROOT / 'evidence/r41/hbm8/xclbin_hbm_map.json').read_text())


def topology_section():
    rows = MAP['hbm_topology']
    return struct.pack('<i4x', len(rows)) + b''.join(
        struct.pack('<BB6xQQ16s', r['type'], r['used'], r['size'] // 1024, r['base'], r['tag'].encode())
        for r in rows)


class Accessor:
    """Records BO calls; a flat bytearray stands in for nothing larger than needed."""
    def __init__(self, paddr=0):
        self.paddr, self.calls = paddr, []

    def bo_open(self, nbytes, bank):
        self.calls.append(('open', nbytes, bank))
        self.size = nbytes
        return self.paddr

    def bo_write(self, offset, data):
        if offset < 0 or offset + len(data) > self.size:
            raise ArgumentError('past the BO')
        self.calls.append(('write', offset, len(data)))

    def bo_read(self, offset, nbytes):
        if nbytes < 0 or offset < 0 or offset + nbytes > self.size:
            raise ArgumentError('past the BO')
        self.calls.append(('read', offset, nbytes))
        return bytes(nbytes)


def link(paddr=0):
    lk = multicore_mmio.MulticoreXDMALink(xclbin='fixture.xclbin', accessor=Accessor(paddr))
    return lk


def banks(path, ip, arg_index=0):
    return dict(MAP['arg_group_banks'][arg_index])


@contextlib.contextmanager
def xclbin():
    """The B40 image's group banks and HBM topology for both link classes."""
    section = {u50._MEM_TOPOLOGY: topology_section()}
    with patch.object(u50, '_xclbin_bank', side_effect=banks), \
         patch.object(multicore_mmio, '_xclbin_bank', side_effect=banks), \
         patch.object(u50, '_xclbin_sections', return_value=section):
        yield


class XRTWindowTests(unittest.TestCase):
    def open(self, nbytes, paddr=0):
        lk = link(paddr)
        with xclbin():
            got = lk.hbm_open(nbytes)
        return lk, got

    def test_three_outlets_share_the_8gib_group(self):
        self.assertEqual({b['index'] for b in MAP['arg_group_banks']}, {41})
        self.assertEqual({(b['tag'], b['size']) for b in MAP['arg_group_banks']}, {('MBG[0:31]', 8 * GIB)})
        self.assertEqual(MAP['plain_connectivity_count'], 96)
        self.assertTrue(all(len(v) == 32 for v in MAP['plain_connectivity_per_arg'].values()))

    def test_8gib_bo_maps_pc0_to_pc31(self):
        lk, got = self.open(8 * GIB)
        self.assertEqual(got, (0, 8 * GIB))
        self.assertEqual(lk._accessor.calls, [('open', 8 * GIB, 41)])
        a = lk.hbm_allocation
        self.assertEqual((a['requested_bytes'], a['exposed_bytes']), (8 * GIB, 8 * GIB))
        self.assertEqual([b['pc'] for b in a['physical_banks']], list(range(32)))
        # The fixture re-encodes only the 32 HBM rows; the board's full section
        # hashes to MAP['allocation_8gib']['topology_sha256'] (window case checks it).
        self.assertEqual(len(a['topology_sha256']), 64)
        self.assertEqual([b['pc'] for b in MAP['allocation_8gib']['physical_banks']], list(range(32)))
        self.assertEqual(a['physical_banks'][16]['allocated_base'], 4 * GIB)

    def test_request_above_the_group_is_clamped_not_extended(self):
        lk, got = self.open(16 * GIB)
        self.assertEqual(got, (0, 8 * GIB))
        self.assertEqual(lk.hbm_allocation['requested_bytes'], 16 * GIB)
        self.assertEqual(lk.hbm_allocation['exposed_bytes'], 8 * GIB)

    def test_4gib_window_of_r40_is_pc0_to_pc15(self):
        lk, _ = self.open(4 * GIB)
        self.assertEqual([b['pc'] for b in lk.hbm_allocation['physical_banks']], list(range(16)))

    def test_mismatched_outlet_groups_are_refused(self):
        lk = link()
        bad = lambda path, ip, arg_index=0: dict(MAP['arg_group_banks'][0], index=41 + arg_index)  # noqa: E731
        with patch.object(multicore_mmio, '_xclbin_bank', side_effect=bad), self.assertRaises(DependencyFailed):
            lk.hbm_open(8 * GIB)


class Registers:
    """MMIO v2 register stand-in: identity, window, status, refusals by range."""
    def __init__(self, local=0x20000):
        self.r = {u50.R_MAGIC: multicore_mmio.MulticoreMMIOBackend.identity_magic,
                  u50.R_AXI_STAT: ((local >> 12) << 32) | 2 | 16 | 32}
        self.local = local

    def read(self, off):
        if off in (0x100, 0x108):
            off = u50.R_AXI_BASE
        return self.r.get(off, 0)

    def write(self, off, v):
        if off in (0x100, 0x108):
            off = u50.R_AXI_BASE
        self.r[off] = v


class BackendTests(unittest.TestCase):
    def backend(self, nbytes=8 * GIB, paddr=0):
        lk = link(paddr)
        regs = Registers()
        lk._accessor.read, lk._accessor.write = regs.read, regs.write
        be = multicore_mmio.MulticoreMMIOBackend(lk, hbm_dma=True, hbm_bytes=nbytes)
        be._handoffs = []
        be._handoff = lambda addr, n: be._handoffs.append((addr, n))
        with xclbin():
            be._open_window(regs.r[u50.R_AXI_STAT])
        return be, regs

    def test_window_registers_and_capacity(self):
        be, regs = self.backend()
        self.assertEqual((regs.r[u50.R_AXI_BASE], regs.r[u50.R_AXI_SIZE]), (0, 8 * GIB))
        self.assertEqual(be.local_limit, 0x20000)
        self.assertEqual(be.memory_bytes, 0x20000 + 8 * GIB)
        self.assertEqual(be.link.words, (0x20000 + 8 * GIB) // 8)
        self.assertEqual(be.axi_window['size'], 8 * GIB)

    def test_dma_offsets_above_4gib_keep_every_bit(self):
        be, _ = self.backend(paddr=0)
        for addr in (0x20000, 0x1_0000_0000 - 8, 0x1_0000_0000, 0x1_0002_0000, 0x1_7FFF_0000,
                     0x2_0002_0000 - 64):
            be.link._accessor.calls.clear()
            be._handoffs.clear()
            be._put_payload(addr, [0x1122334455667788] * (8 if addr + 64 <= be.memory_bytes else 1))
            kind, offset, n = be.link._accessor.calls[-1]
            self.assertEqual((kind, offset), ('write', addr - 0x20000), hex(addr))
            self.assertEqual(be._handoffs, [(addr, n)])
            be.link._accessor.calls.clear()
            be._get_payload(addr, 1)
            self.assertEqual(be.link._accessor.calls[-1][:2], ('read', addr - 0x20000))

    def test_host_transfers_past_the_window_are_refused_before_the_bus(self):
        be, _ = self.backend()
        end = be.memory_bytes
        for addr, n in ((end - 8, 2), (end, 1), (end + 8 * GIB, 1)):
            be.link._accessor.calls.clear()
            be._handoffs.clear()
            with self.assertRaises(ArgumentError):
                be._put_payload(addr, [0] * n)
            self.assertEqual((be.link._accessor.calls, be._handoffs), ([], []))
        # The XRT accessor's own BO bounds (checked before any library call).
        xrt = u50._XRTRegisters.__new__(u50._XRTRegisters)
        xrt._bo_size, xrt._lib, xrt._handle, xrt._bo = 8 * GIB, None, None, None
        for call in (lambda: xrt.bo_write(8 * GIB - 8, b'\0' * 16), lambda: xrt.bo_read(8 * GIB, 8),
                     lambda: xrt.bo_read(0, -8), lambda: xrt.bo_write(-64, b'\0' * 8)):
            with self.assertRaises(ArgumentError):
                call()


class AllocatorTests(unittest.TestCase):
    def heap(self):
        return _Allocator(0x20000, 0x20000 + 8 * GIB, reserved=DEVICE_RESERVED_RANGES)

    def test_reserved_page_is_kept_in_an_8gib_heap(self):
        h = self.heap()
        self.assertEqual(h.reserved, ((0x7fff0000, 0x7fff1000),))
        self.assertEqual(h.capacity, 8 * GIB - 4096)
        a, _ = h.alloc(0x7fff0000 - 0x20000 - 64)
        b, _ = h.alloc(128)
        self.assertEqual((a, b), (0x20000, 0x7fff1000))

    def test_largest_single_storage(self):
        h = self.heap()
        a, _ = h.alloc(int(2.25 * GIB))
        self.assertEqual(a, 0x7fff1000)
        h = self.heap()
        biggest = 0x20000 + 8 * GIB - 0x7fff1000
        a, _ = h.alloc(biggest)
        self.assertEqual(a + biggest, 0x20000 + 8 * GIB)
        with self.assertRaises(BufferError):
            self.heap().alloc(biggest + 64)
        # In the R40 4 GiB window no single storage can exceed ~2 GiB.
        h4 = _Allocator(0x20000, 0x20000 + 4 * GIB, reserved=DEVICE_RESERVED_RANGES)
        with self.assertRaises(BufferError):
            h4.alloc(int(2.25 * GIB))

    def test_refusal_leaves_state_unchanged(self):
        h = self.heap()
        h.alloc(4096)
        state = (h.next, list(h.free_list))
        with self.assertRaises(BufferError):
            h.alloc(8 * GIB)
        self.assertEqual((h.next, list(h.free_list)), state)


class PlanTests(unittest.TestCase):
    def test_plans_are_inside_ordinary_memory(self):
        for base in (0, 1 << 32):
            w = hbm8_case.window_like(base, 8 * GIB)
            for mode in ('board', 'sim'):
                for case in hbm8_case.CASES:
                    placements, _ = hbm8_case.plan(case, mode, w)
                    ranges = hbm8_case.sparse_ranges(case, mode, w)
                    self.assertEqual(ranges, sorted(ranges))
                    for name, a, n in placements:
                        if case == 'bigoffset':
                            continue
                        lo = a - 0x20000
                        self.assertTrue(any(o <= lo and lo + n <= o + m for o, m in ranges), (case, name))

    def test_samples_cover_every_pc_boundary_and_the_edges(self):
        w = hbm8_case.window_like(0, 8 * GIB)
        layout = hbm8_case.sample_layout(w)
        offs = [e['offset'] for e in layout]
        self.assertEqual(offs[0], 0)
        self.assertEqual(offs[-1] + 512, 8 * GIB)
        for pc in range(1, 32):
            self.assertIn(pc * (256 << 20) - 256, offs)
        self.assertIn((1 << 32) - 0x20000 - 256, offs)

    def test_alias_pairs_differ_in_one_named_bit_set(self):
        placements, _ = hbm8_case.plan('alias', 'board', hbm8_case.window_like(0, 8 * GIB))
        at = {n: a for n, a, _ in placements}
        for name, delta in (('bit30', 1 << 30), ('bit31', 1 << 31), ('bit32', 1 << 32), ('bits32+31', 3 << 31)):
            self.assertEqual(at[f'{name}-high'] - at[f'{name}-low'], delta)
        self.assertEqual(at['tmu+4GiB'], 0x1_7FFF_0000)
        self.assertEqual(len(set(at.values())), len(at))

    def test_board_stream_is_one_storage_above_2gib_across_4gib(self):
        (name, a, n), = hbm8_case.plan('stream', 'board', hbm8_case.window_like(0, 8 * GIB))[0]
        self.assertGreater(n, 2 * GIB)
        self.assertLess(a, 1 << 32)
        self.assertGreater(a + n - 0x20000, 4 * GIB)


class ManifestTests(unittest.TestCase):
    def test_registered_manifests_are_generated_and_complete(self):
        for name in hbm8_manifest.GROUPS:
            path = ROOT / f'spec/r41/hbm8/experiments/{name}.json'
            spec = json.loads(path.read_text())
            self.assertEqual(spec, hbm8_manifest.manifest(name), name)
            self.assertEqual(spec['allowed_hosts'], ['acf3019'])
            self.assertEqual(spec['registration']['status'], 'registered-not-run')
            board = spec['board']
            self.assertEqual(board['image_sha256'], 'a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083')
            self.assertEqual(board['memory_map']['source_sha256'], experiment.digest(ROOT / board['memory_map']['source']))
            self.assertEqual(board['memory_map']['window_bytes'], 8 * GIB)
            outputs = set()
            for step in spec['steps']:
                self.assertIn(step['kind'], experiment.KINDS)
                tool = next(a for a in step['argv'] if a.startswith('tools/'))
                self.assertTrue((ROOT / tool).is_file(), tool)
                if step['kind'] == 'fpga':
                    self.assertTrue(step['timing_boundary'] and step['cache_state'])
                    args = json.loads(step['env']['TNCA_BACKEND_ARGS'])
                    want = 16 * GIB if step['id'] == 'clamp' else 8 * GIB
                    self.assertEqual(args['hbm_bytes'], want)
                    declared = dict(board, **(step.get('board') or {}))
                    self.assertEqual(declared['backend_args']['hbm_bytes'], want)
                outputs |= set(step.get('outputs', []))
            self.assertIn(spec['conclusion_source'], outputs)
        cases = [c for g in hbm8_manifest.GROUPS.values() for c in g]
        self.assertEqual(sorted(cases), sorted(hbm8_case.CASES))

    def test_board_probe_refuses_other_images_clocks_and_users(self):
        spec = hbm8_manifest.manifest('board-hbm8')
        identity = dict(image_uuid=spec['board']['image_uuid'], image_sha256='x', clock_hz=90000000)
        good = dict(loaded_uuid=identity['image_uuid'], clock_counter_mhz=[90, 450], contexts=0, client_pid=None)
        self.assertIsNone(experiment.board_refusal(identity, good))
        for change in (dict(loaded_uuid='428c00e5-6996-92c7-1f8f-ec41f2d037b3'), dict(clock_counter_mhz=[100]),
                       dict(contexts=1), dict(client_pid='3887377'), dict(contexts=None)):
            self.assertIsNotNone(experiment.board_refusal(identity, dict(good, **change)), change)


class SummaryTests(unittest.TestCase):
    def summarize(self, mode='board', fault=None):
        import io
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix='hbm8-summary-'))
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        argv = ['--group', 'board-hbm8', '--out', str(tmp / 'summary.json')]
        for i, name in enumerate(hbm8_manifest.GROUPS['board-hbm8']):
            row = dict(status='functional-pass', mode=mode, software_tree='files:one',
                       checks=[dict(name='values', ok=True)],
                       axi_status_at_end=dict(error=False, stray=False))
            if i == 0:
                if fault == 'omit':
                    continue
                if fault == 'tree':
                    row['software_tree'] = 'files:two'
                if fault == 'mode':
                    row['mode'] = 'sim'
                if fault == 'check':
                    row['checks'][0]['ok'] = False
                if fault == 'axi':
                    row['axi_status_at_end']['error'] = True
                if fault == 'status':
                    row['status'] = 'failed-preserved'
            path = tmp / f'{name}.json'
            path.write_text(json.dumps(row))
            argv.append(f'{name}={path}')
        with contextlib.redirect_stdout(io.StringIO()):
            rc = hbm8_summary.main(argv)
        return rc, json.loads((tmp / 'summary.json').read_text())

    def test_complete_board_reports_qualify(self):
        rc, record = self.summarize()
        self.assertEqual(rc, 0)
        self.assertTrue(record['qualified'])
        self.assertEqual(record['mode'], 'board')

    def test_simulation_does_not_claim_board_qualification(self):
        rc, record = self.summarize(mode='sim')
        self.assertEqual(rc, 0)
        self.assertEqual(record['mode'], 'sim')
        self.assertIn('run the waiting board experiment', record['next_step'])

    def test_incomplete_mixed_or_failed_evidence_cannot_qualify(self):
        for fault in ('omit', 'tree', 'mode', 'check', 'axi', 'status'):
            with self.subTest(fault=fault):
                rc, record = self.summarize(fault=fault)
                self.assertEqual(rc, 1)
                self.assertFalse(record['qualified'])
                self.assertTrue(record['qualification_problems'])


class QueueScriptTests(unittest.TestCase):
    """evidence/r41/hbm8/launches/hbm8_queue.sh against a fake sysfs, never a card."""
    SCRIPT = ROOT / 'evidence/r41/hbm8/launches/hbm8_queue.sh'

    def run_queue(self, *, uuid='1bada9db-78f3-5bc8-7ba6-cce062e7ade8', mhz='90', mem_kb=32 << 20,
                  stage1='BOARD-QUEUE-DONE', stage1_wait_polls=0, busy_once=False):
        import subprocess
        import tarfile
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix='hbm8-queue-'))
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        dev, mgmt = tmp / 'sys/0000:04:00.1', tmp / 'sys/0000:04:00.0/clock_freq_counter.m.28311552'
        dev.mkdir(parents=True)
        mgmt.mkdir(parents=True)
        (dev / 'kdsstat').write_text('xclbin:\t\t\tx\noutstanding execs:\t0\ntotal execs:\t\t0\ncontexts:\t\t0\nclient pid:\n')
        (dev / 'xclbinuuid').write_text(uuid + '\n')
        (mgmt / 'clock_counter_freqs').write_text(f'{mhz}\n500\n450\n')
        (tmp / 'meminfo').write_text(f'MemTotal: 65000000 kB\nMemAvailable: {mem_kb} kB\n')
        (tmp / 'stage1.log').write_text('2026-10-21T13:18:39Z queue start\n' + (stage1 + '\n' if stage1 else ''))
        src = tmp / 'archive/tools/performance_harness'
        src.mkdir(parents=True)
        (src / 'experiment.py').write_text(
            'import sys,pathlib\n'
            'pathlib.Path(sys.argv[sys.argv.index("--root")+1]).parent.joinpath("ran.txt").open("a").write(sys.argv[1]+"\\n")\n'
            'print("completed-experiment")\n')
        with tarfile.open(tmp / 'r41hbm8_abc.tar.gz', 'w:gz') as t:
            t.add(tmp / 'archive/tools', arcname='tools')
        env = dict(HBM8_HOME=str(tmp), HBM8_SYSFS=str(tmp / 'sys'), HBM8_STAGE1_LOG=str(tmp / 'stage1.log'),
                   HBM8_STAGE1_PATTERN='^bash [^ ]*' + __import__('uuid').uuid4().hex, HBM8_IDLE_MINUTES='2', HBM8_POLL_SECONDS='0',
                   HBM8_MEMINFO=str(tmp / 'meminfo'), HBM8_MEM_WAIT_SECONDS='0',
                   HBM8_PYTHON=sys.executable, PATH='/usr/bin:/bin')
        fakebin = tmp / 'bin'
        fakebin.mkdir()
        if stage1_wait_polls:
            # Keep the predecessor alive for N polls. No real process is
            # created or killed, and neither manifest may run during them.
            fake = fakebin / 'pgrep'
            fake.write_text(
                '#!/bin/bash\n'
                f'n=$(cat "{tmp}/queue-polls" 2>/dev/null || echo 0)\n'
                f'n=$((n+1)); echo "$n" > "{tmp}/queue-polls"\n'
                f'test ! -e "{tmp}/r41hbm8_board_abc/ran.txt" || exit 99\n'
                f'test "$n" -le {stage1_wait_polls}\n')
            fake.chmod(0o755)
        if busy_once:
            fake = fakebin / 'sleep'
            fake.write_text(
                '#!/bin/bash\n'
                f'n=$(cat "{tmp}/idle-polls" 2>/dev/null || echo 0)\n'
                f'n=$((n+1)); echo "$n" > "{tmp}/idle-polls"\n'
                f'if [ "$n" = 1 ]; then sed -i "s/contexts:.*/contexts: 1/" "{dev}/kdsstat"; fi\n'
                f'if [ "$n" = 2 ]; then sed -i "s/contexts:.*/contexts: 0/" "{dev}/kdsstat"; fi\n')
            fake.chmod(0o755)
        env['PATH'] = str(fakebin) + ':' + env['PATH']
        r = subprocess.run(['bash', str(self.SCRIPT), 'abc'], env=env, capture_output=True, text=True, timeout=120)
        if stage1_wait_polls:
            self.assertEqual(int((tmp / 'queue-polls').read_text()), stage1_wait_polls + 1)
        if busy_once:
            # Two idle observations, interrupted by busy, require a fresh
            # pair. Counting nonconsecutive idle observations would take 3.
            self.assertEqual(int((tmp / 'idle-polls').read_text()), 4)
        d = tmp / 'r41hbm8_board_abc'
        ran = (d / 'ran.txt').read_text().split() if (d / 'ran.txt').exists() else []
        return r.returncode, (d / 'queue.log').read_text(), ran

    def test_runs_both_manifests_after_stage1_on_b40(self):
        rc, log, ran = self.run_queue()
        self.assertEqual(rc, 0, log)
        self.assertEqual(ran, ['spec/r41/hbm8/experiments/board-hbm8.json', 'spec/r41/hbm8/experiments/board-hbm8-big.json'])
        self.assertIn('BOARD-QUEUE-DONE', log)
        self.assertTrue(log.rstrip().endswith('HBM8-QUEUE-DONE'))

    def test_stage1_exit_without_done_still_requires_b40(self):
        rc, log, ran = self.run_queue(stage1='')
        self.assertEqual(rc, 0, log)
        self.assertIn('did not log BOARD-QUEUE-DONE', log)

    def test_waits_for_the_live_stage1_queue_to_exit(self):
        rc, log, ran = self.run_queue(stage1='', stage1_wait_polls=3)
        self.assertEqual((rc, len(ran)), (0, 2), log)

    def test_busy_card_restarts_the_consecutive_idle_period(self):
        rc, log, ran = self.run_queue(busy_once=True)
        self.assertEqual((rc, len(ran)), (0, 2), log)

    def test_other_image_or_clock_is_never_loaded_over(self):
        for kw in (dict(uuid='428c00e5-6996-92c7-1f8f-ec41f2d037b3'), dict(mhz='245')):
            rc, log, ran = self.run_queue(**kw)
            self.assertEqual((rc, ran), (5, []), log)
            self.assertIn('without loading anything', log)

    def test_low_host_memory_stops_before_running(self):
        rc, log, ran = self.run_queue(mem_kb=8 << 20)
        self.assertEqual((rc, ran), (7, []), log)


if __name__ == '__main__':
    unittest.main()
