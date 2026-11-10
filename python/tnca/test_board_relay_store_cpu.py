"""The board relay's chunk store (2026-11-11): a piece a client wrote once is written again from the card host's
copy, by digest, and a piece read back unchanged is answered from the client's own copy -- the same bytes at the
same addresses, with less over the SSH tunnel (R41-F3: 4.9 GB each way for one llama decode step).

A fake link stands for the card (its HBM a bytearray, its registers quiet); the relay and the client talk over a
real Unix socket, one connection per RelayLink as on the card host."""
import os
import socket
import tempfile
import threading
import unittest

from tnca.backends import board_relay as br
from tnca.errors import ArgumentError

MiB = 1 << 20


class FakeLink:
    """The card as the relay sees it: idle registers, and HBM that a test may change behind the client's back,
    as a kernel would."""

    def __init__(self, size=64 * MiB):
        self.hbm = bytearray(size)
        self.writes = self.reads = 0

    def read_reg(self, offset):
        return 0

    def read_host_status(self):
        return 0

    def describe(self):
        return 'a fake card'

    def hbm_write(self, offset, data):
        self.writes += len(data)
        self.hbm[offset:offset + len(data)] = data

    def hbm_read(self, offset, nbytes):
        self.reads += nbytes
        return bytes(self.hbm[offset:offset + nbytes])


class Relay:
    """A BoardRelay serving one connection at a time on a Unix socket, in a thread."""

    def __init__(self, link, store):
        self.dir = tempfile.mkdtemp(prefix='tnca-relay-test-')
        self.path = os.path.join(self.dir, 'relay.sock')
        self.relay = br.BoardRelay(link, self.dir, image=None, store=store)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(4)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        number = 0
        while True:
            try:
                sock, _ = self.srv.accept()
            except OSError:
                return
            number += 1
            conn = self.relay.accept(sock, number)
            while br._serve_message(self.relay, conn):
                pass
            self.relay.end(conn)

    def link(self, **kw):
        return br.RelayLink(self.path, label='t', **kw).open()

    def close(self):
        self.srv.close()


class ChunkStoreThroughTheRelay(unittest.TestCase):
    def setUp(self):
        self.card = FakeLink()
        self.store_dir = tempfile.mkdtemp(prefix='tnca-chunks-')
        self.store = br.ChunkStore(self.store_dir, 1 << 30)
        self.relay = Relay(self.card, self.store)
        self.addCleanup(self.relay.close)
        self.weights = os.urandom(3 * MiB)

    def test_a_piece_crosses_once_then_comes_from_the_card_host(self):
        first = self.relay.link()
        first.hbm_write(0, self.weights)
        first.close()
        self.assertEqual((first.stats['sent'], first.stats['put_hits']), (len(self.weights), 0))
        second = self.relay.link()                      # another session, another candidate: the same weights
        second.hbm_write(8 * MiB, self.weights)
        second.close()
        self.assertEqual((second.stats['sent'], second.stats['put_hits']), (0, len(self.weights)))
        self.assertEqual(bytes(self.card.hbm[8 * MiB:8 * MiB + len(self.weights)]), self.weights)
        self.assertEqual(self.store.counts['hit_bytes'], len(self.weights))

    def test_a_piece_the_device_left_unchanged_is_not_read_back(self):
        link = self.relay.link()
        link.hbm_write(0, self.weights)
        reads_before = self.card.reads
        self.assertEqual(link.hbm_read(0, len(self.weights)), self.weights)
        self.assertEqual((link.stats['read_unchanged'], link.stats['read']), (len(self.weights), 0))
        # The relay read it on the card host to digest it; nothing crossed back.
        self.assertEqual(self.card.reads - reads_before, len(self.weights))
        link.close()

    def test_a_piece_the_device_changed_is_read_back(self):
        link = self.relay.link()
        link.hbm_write(0, self.weights)
        self.card.hbm[MiB + 5] ^= 0xFF                 # a kernel wrote into it
        got = link.hbm_read(0, len(self.weights))
        self.assertEqual(got, bytes(self.card.hbm[:len(self.weights)]))
        self.assertNotEqual(got, self.weights)
        self.assertEqual(link.stats['read'], len(self.weights))
        link.close()

    def test_an_overlapping_write_forgets_what_it_covered(self):
        link = self.relay.link()
        link.hbm_write(0, self.weights)
        patch = os.urandom(br.DIGEST_MIN)
        link.hbm_write(MiB + 4096, patch)              # inside the first piece, at another start
        expect = bytearray(self.weights)
        expect[MiB + 4096:MiB + 4096 + len(patch)] = patch
        self.assertEqual(link.hbm_read(0, len(self.weights)), bytes(expect))
        link.close()

    def test_a_mutable_buffer_is_kept_as_it_was_written(self):
        link = self.relay.link()
        data = bytearray(self.weights)
        link.hbm_write(0, data)
        data[0] ^= 0xFF                                # the caller reuses its buffer
        self.assertEqual(link.hbm_read(0, len(self.weights)), self.weights)
        link.close()

    def test_small_pieces_cross_as_they_always_did(self):
        link = self.relay.link()
        small = os.urandom(br.DIGEST_MIN - 8)
        link.hbm_write(0, small)
        self.assertEqual(link.hbm_read(0, len(small)), small)
        self.assertEqual((link.stats['sent'], link.stats['read'], self.store.counts['kept']), (len(small),
                                                                                              len(small), 0))
        link.close()

    def test_a_relay_without_digests_is_spoken_to_as_before(self):
        link = self.relay.link()
        link.relay_info = dict(link.relay_info, digest=None, cache=None)     # an older relay's hello
        link.hbm_write(0, self.weights)
        self.assertEqual(link.hbm_read(0, len(self.weights)), self.weights)
        self.assertEqual((link.stats['sent'], link.stats['read']), (len(self.weights), len(self.weights)))
        self.assertEqual(self.store.counts['kept'], 0)
        link.close()

    def test_a_relay_without_a_store_still_saves_the_reads(self):
        plain = Relay(FakeLink(), None)
        self.addCleanup(plain.close)
        link = plain.link()
        link.hbm_write(0, self.weights)
        self.assertEqual(link.hbm_read(0, len(self.weights)), self.weights)
        self.assertEqual((link.stats['sent'], link.stats['read_unchanged']),
                         (len(self.weights), len(self.weights)))
        link.close()


class TheStore(unittest.TestCase):
    def test_a_name_is_the_digest_of_the_bytes_the_relay_received(self):
        store = br.ChunkStore(tempfile.mkdtemp(), 1 << 30)
        data = os.urandom(br.DIGEST_MIN)
        digest = store.keep(data)
        self.assertEqual(digest, br._digest(data))
        self.assertEqual(store.get(digest, len(data)), data)
        self.assertIsNone(store.get(digest, len(data) - 1))           # another length is another piece
        self.assertIsNone(store.get('0' * 64, len(data)))
        with self.assertRaises(ArgumentError):
            store.get('../../etc/passwd', 10)
        self.assertEqual(oct(os.stat(store.root).st_mode & 0o777), '0o700')

    def test_a_client_naming_a_digest_it_cannot_know_gets_nothing(self):
        card = FakeLink()
        relay = br.BoardRelay(card, tempfile.mkdtemp(), image=None,
                              store=br.ChunkStore(tempfile.mkdtemp(), 1 << 30))
        log = open(os.devnull, 'w')
        self.addCleanup(log.close)
        conn = br._Connection(None, 1, 'p', log)
        reply, _, _ = relay.handle(conn, {'op': 'hbm_put', 'offset': 0, 'nbytes': MiB, 'digest': 'a' * 64}, b'')
        self.assertEqual(reply, {'ok': True, 'hit': False})
        self.assertEqual(card.writes, 0)

    def test_the_store_is_bounded_least_recently_used_first(self):
        store = br.ChunkStore(tempfile.mkdtemp(), 3 * br.DIGEST_MIN)
        pieces = [os.urandom(br.DIGEST_MIN) for _ in range(5)]
        digests = []
        for i, p in enumerate(pieces):
            digests.append(store.keep(p))
            os.utime(os.path.join(store.root, digests[-1]), (1000 + i, 1000 + i))
        self.assertLessEqual(store.used, 3 * br.DIGEST_MIN)
        self.assertIsNotNone(store.get(digests[-1], br.DIGEST_MIN))
        self.assertIsNone(store.get(digests[0], br.DIGEST_MIN))
        again = br.ChunkStore(store.root, 3 * br.DIGEST_MIN)               # a later session counts what is there
        self.assertEqual(again.used, store.used)


class Kept(unittest.TestCase):
    def test_overlaps_are_forgotten_and_the_budget_holds(self):
        k = br._Kept(10)
        k.add(0, b'abcd', 'd0')
        k.add(4, b'efgh', 'd1')
        self.assertEqual(k.get(4, 4), (b'efgh', 'd1'))
        k.drop(3, 2)                                   # touches both
        self.assertEqual((k.get(0, 4), k.get(4, 4)), (None, None))
        k.add(0, b'abcd', 'd0')
        k.add(10, b'ijkl', 'd2')
        k.add(20, b'mnop', 'd3')                       # 12 bytes > 10: the least recently used goes
        self.assertIsNone(k.get(0, 4))
        self.assertEqual(k.size, 8)
        self.assertIsNone(k.get(10, 3))                # only the exact piece


if __name__ == '__main__':
    unittest.main()
