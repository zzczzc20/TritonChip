"""CPU unit checks for preserving valid code RAM across module eviction."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest import TestCase, main, mock
import threading

from tnca.runtime import Session, _Allocator


class CodeAddressReuseTests(TestCase):
    def test_allocator_reserves_only_free_aligned_ranges(self):
        allocator = _Allocator(0, 512, reserved=((320, 384),))
        blocks = [allocator.alloc(64) for _ in range(3)]
        allocator.release(*blocks[0])
        allocator.release(*blocks[1])
        self.assertEqual(allocator.alloc_at(64, 64), (64, 64))
        self.assertEqual(allocator.free_list, [(0, 64)])
        self.assertFalse(allocator.can_alloc_at(64, 64))  # still live
        self.assertFalse(allocator.can_alloc_at(64, 320))  # reserved
        self.assertFalse(allocator.can_alloc_at(64, 17))  # unaligned
        self.assertEqual(allocator.alloc_at(64, 256), (256, 64))
        self.assertEqual(allocator.used, 192)
        for block in ((64, 64), blocks[2], (256, 64)):
            allocator.release(*block)
        self.assertEqual((allocator.used, allocator.next, allocator.free_list), (0, 0, []))

    def test_free_exact_old_locations_avoid_uploading_unchanged_code(self):
        class Device:
            code_payload_shadow = True
            local_limit = 512
            def __init__(self):
                self._code_window_shadow, self.uploaded_words = {}, []
            def upload(self, addr, code):
                written = 0
                for offset in range(0, len(code), 8):
                    value = int.from_bytes(code[offset:offset + 8], 'little')
                    if self._code_window_shadow.get(addr + offset) != value:
                        self._code_window_shadow[addr + offset] = value
                        written += 1
                self.uploaded_words.append(written)
                return SimpleNamespace(ok=True)
        session = Session.__new__(Session)
        session._check_usable = lambda: None
        session._lock = threading.RLock()
        session._code = _Allocator(0, 512)
        session._code_address_hints = OrderedDict()
        session._active_events = OrderedDict()
        session._modules = []
        session.profile, session.epoch, session.dev = 'core-vrf-sync-v5', 0, Device()
        codes = {name: bytes(name, 'ascii') * 64 for name in ('A', 'B', 'C')}
        def load(path):
            manifest = {'identity': {'abi_version': [1, 0], 'kernel_name': path},
                        'parameters': {'signature': []},
                        'code': {'segments': [{'name': 'text', 'sha256': path}]}}
            return manifest, {'text': codes[path]}
        with mock.patch('tnca.runtime.artifact.load', side_effect=load), \
             mock.patch('tnca.profiles.check_executes'):
            first = [session.load(name) for name in ('A', 'B', 'C')]
            self.assertEqual([m.code_addr for m in first], [0, 64, 128])
            session.unload(first[0]); session.unload(first[1])
            again_b, again_a = session.load('B'), session.load('A')
            self.assertEqual((again_b.code_addr, again_a.code_addr), (64, 0))
            self.assertEqual(session.dev.uploaded_words[-2:], [0, 0])
            session.unload(again_b)
            session.dev._code_window_shadow[64] = 0  # stale code must be uploaded again
            changed = session.load('B')
            self.assertEqual(changed.code_addr, 64)
            self.assertGreater(session.dev.uploaded_words[-1], 0)


if __name__ == '__main__':
    main()
