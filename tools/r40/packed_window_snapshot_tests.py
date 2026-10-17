#!/usr/bin/env python3
"""Diagnostic RTL: unpack the DMA snapshot after overwriting its source.

Uses the IR builder to isolate lane bits and ownership. These results do
not count as standard Torch performance evidence.
"""
import os
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(path))
import triton  # noqa: F401
from tnca import aot, device
from tnca.runtime import Session
from tnca_cc.frontend import TTIRImporter


def main():
    profile = 'core-vrf-sync-v5/hbm'
    device.prepare(profile)
    with tempfile.TemporaryDirectory(prefix='r40-packed-snapshot-') as tmp:
        for dtype in ('f16', 'bf16'):
            def program(b, target):
                imp = TTIRImporter(target)
                imp.kernel, imp.b = b.kernel, b
                imp._packed_window_unpack = True
                args = b.special('arg_base')
                src, dst = b.sload(args, 0), b.sload(args, 8)
                imp._scratch = b.sload(args, 16)  # explicit diagnostic TensorMap workspace
                descriptor = imp._build_descriptor(base=src, elem=dtype, extents=[4, 16],
                    byte_strides=[32, 2], block_shape=[4, 16], what='snapshot test')
                zero = b.const(0)
                tile = imp._dma_in(descriptor, [zero, zero], [4, 16], dtype, 'snapshot')
                clear = b.splat(zero, dtype=dtype)
                for row in range(4):
                    b.store(b.sbini('add', src, row * 32), clear, size='h')
                b.fence()
                get = imp._tile_row_getter(tile)
                for row in range(4):
                    b.store(b.sbini('add', dst, row * 32), get(row), size='h')
                b.fence()
                b.exit()
            result = aot.compile_aot(program, kernel_name='packed_snapshot_' + dtype,
                signature={'src': '*fp16' if dtype == 'f16' else '*bf16',
                           'dst': '*fp16' if dtype == 'f16' else '*bf16', 'maps': '*i8'},
                profile=profile.split('/')[0], source_kind='tnca-ir', options={'window_reuse': 1},
                output=str(Path(tmp) / (dtype + '.tnca')))
            words = [0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01, 0xfe02,
                     1, 0x8001, 0x7fff, 0xffff, 0x0123, 0x4567, 0x89ab, 0xcdef]
            if dtype == 'bf16':
                words[4:8] = [0x7f80, 0xff80, 0x7fc1, 0xffc2]
            payload = b''.join(struct.pack('<H', words[(col + row * 3) % 16])
                               for row in range(4) for col in range(16))
            with Session(profile=profile, host_timeout=172800, deadline_cycles=200000) as session:
                module = session.load(result.path)
                src, dst = session.allocate(128), session.allocate(128)
                maps = session.allocate(256)
                session.upload(src, payload)
                session.upload(dst, bytes(128))
                event = session.submit(module, {'src': src, 'dst': dst, 'maps': maps})
                session.wait(event)
                assert event.ok, event.error
                assert session.download(src) == bytes(128), 'overwrite did not execute'
                assert session.download(dst) == payload, 'unpack reread memory or changed lane bits'
                print(f'PASS {dtype}: source overwritten, original snapshot bits preserved '
                      f'(zeros, signs, infinities, NaN payloads, both packed rows)', flush=True)
    print('r40-packed-window-snapshot-tests PASS (2/2)', flush=True)


if __name__ == '__main__':
    main()
