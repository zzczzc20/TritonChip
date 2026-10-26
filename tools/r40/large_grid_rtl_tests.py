#!/usr/bin/env python3
"""Diagnostic: negotiated launch PID/extent bits reach the executing core.

IR builder isolates the transport; this is not Torch performance evidence.
"""
from pathlib import Path
import struct
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
import triton  # load the plugin before importing runtime adapters
from tnca import aot, device
from tnca.runtime import Session
from tnca.errors import AdmissionError


def main():
    profile = 'core-vrf-sync-v2'
    def program(b, target):
        out = b.sload(b.special('arg_base'), 0)
        for i in range(3):
            b.sstore(b.sbini('add', out, i * 8), b.program_id(i), size='d')
            b.sstore(b.sbini('add', out, (i + 3) * 8), b.num_programs(i), size='d')
        b.exit()
    def streamed(b, target):
        base = b.special('arg_base')
        out, counter = b.sload(base, 0), b.sload(base, 8)
        x, y, z = [b.program_id(i) for i in range(3)]
        nx, ny = b.num_programs(0), b.num_programs(1)
        linear = b.sbin('add', x, b.sbin('mul', nx,
            b.sbin('add', y, b.sbin('mul', ny, z))))
        b.sstore(b.sbin('add', out, b.sbini('mul', linear, 8)), linear, size='d')
        b.atomic('add', counter, b.const(1), size='d')
        b.fence()
        b.exit()
    with tempfile.TemporaryDirectory(prefix='r40-grid-rtl-') as tmp:
        result = aot.compile_aot(program, kernel_name='grid_metadata',
            signature={'out': '*i64'}, source_kind='tnca-ir', profile=profile,
            output=str(Path(tmp) / 'grid.tnca'))
        stream = aot.compile_aot(streamed, kernel_name='grid_stream',
            signature={'out': '*i64', 'counter': '*i64'}, source_kind='tnca-ir',
            profile=profile, options={'rctx_spmd': 1},
            output=str(Path(tmp) / 'stream.tnca'))
        for backend in ('sim-file', 'fpga-u50-sim'):
            if backend == 'sim-file':
                device.prepare(profile)
            with Session(profile=profile, backend=backend, host_timeout=172800) as s:
                m = s.load(result.path)
                out, args = s.allocate(48), s.allocate(8)
                s.upload(args, struct.pack('<Q', out.addr))
                assert s.dev.large_grid_limit() == 1 << 24
                for pid, npid in (((2, 3, 4), (5, 6, 7)),
                                 ((65536, 65537, 65538), (65539, 65540, 65541)),
                                 ((0x80000001, 0x80000002, 0x80000003),
                                  (0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD))):
                    s.upload(out, bytes(48))
                    reply = s.dev.launch(m.code_addr, args.addr, pid=pid, npid=npid)
                    assert reply.ok, reply
                    got = struct.unpack('<6Q', s.download(out))
                    assert got == (*pid, *npid), (backend, got, pid, npid)
                    print(f'PASS {backend}: core PID={got[:3]} extent={got[3:]}', flush=True)
                sm = s.load(stream.path)
                slots, counter = s.allocate(6144 * 8), s.allocate(8)
                before = len(s.dev.history)
                try:
                    s.submit(sm, {'out': slots, 'counter': counter}, grid=(6144, 1, 1))
                    raise AssertionError('default admission accepted a large grid')
                except AdmissionError:
                    assert len(s.dev.history) == before, 'refusal reached the device'
                s.large_grid = True
                for grid in ((6144, 1, 1), (64, 32, 3)):
                    s.upload(slots, bytes([255]) * slots.size)
                    s.upload(counter, bytes(8))
                    event = s.submit(sm, {'out': slots, 'counter': counter}, grid=grid)
                    s.wait(event)
                    assert event.ok, event.error
                    assert struct.unpack('<6144Q', s.download(slots)) == tuple(range(6144))
                    assert struct.unpack('<Q', s.download(counter))[0] == 6144, 'duplicate program'
                    print(f'PASS {backend}: grid={grid}, 6144 unique IDs, exactly 6144 atomics', flush=True)
    print('r40-large-grid-rtl PASS (10/10)', flush=True)


if __name__ == '__main__':
    main()
