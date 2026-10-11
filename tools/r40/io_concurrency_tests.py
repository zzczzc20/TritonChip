#!/usr/bin/env python3
"""Mailbox ownership under concurrent transfers, including a lock mutation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.transport import Transport, Reply, OP_UPLOAD, OP_DOWNLOAD, ST_OK


class Mailbox(Transport):
    """A deliberately shared payload slot, like the real file/register ABI.

    Scheduling gaps on both sides of the request expose upload overwrite
    and wrong-download races without invoking a simulator or compiler.
    """
    def __init__(self, workdir):
        super().__init__(None, workdir=workdir)
        self.memory = {}

    def alive(self):
        return True

    def _put_payload(self, addr, words):
        self.payload = b''.join(w.to_bytes(8, 'little') for w in words)
        time.sleep(0.001)

    def _put_request(self, seq, op, a0, a1, a2, a3, a4, a5):
        if op == OP_UPLOAD:
            self.memory[a0] = self.payload
        elif op == OP_DOWNLOAD:
            self.download_payload = self.memory[a0]

    def _await(self, seq, timeout=None):
        time.sleep(0.001)
        return Reply(seq, ST_OK, 0, 0)

    def _get_payload(self, addr, words):
        return self.download_payload


def run(medium):
    def worker(index):
        for i in range(32):
            addr = index * 0x10000 + i * 8
            data = ((index << 48) + i + 1).to_bytes(8, 'little')
            assert medium.upload(addr, data).ok
            reply, got = medium.download(addr, 8)
            assert reply.ok and got == data, (index, i, got, data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    seqs = [e['seq'] for e in medium.history]
    assert seqs == list(range(1, 257)), 'sequence/reply ownership changed'


def main():
    with tempfile.TemporaryDirectory() as d:
        run(Mailbox(d))
        print('PASS 4 threads, 128 uploads + 128 downloads, exact bytes and sequences')
        class UnlockedPayload(Mailbox):
            upload = Transport.upload.__wrapped__
            download = Transport.download.__wrapped__
        try:
            run(UnlockedPayload(d))
        except AssertionError:
            print('PASS removing whole-operation locks is detected despite request lock')
        else:
            raise AssertionError('concurrency test did not catch the lock mutation')
    print('r40-io-concurrency-tests PASS (2/2)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
