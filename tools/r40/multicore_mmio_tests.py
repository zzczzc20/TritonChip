#!/usr/bin/env python3
"""Physical grids through the actual v2 register protocol and AXI RTL."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.backends.multicore_mmio_sim import stand_in
from tnca.backends.u50 import R_REQ_SEQ, R_REQ_OP, R_REQ_A0, R_DOORBELL, R_SERVED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, choices=(1, 2, 3, 4, 8), default=3)
    parser.add_argument('--ports', type=int, default=3)
    parser.add_argument('--large-grid', action='store_true')
    parser.add_argument('--board-top', action='store_true')
    parser.add_argument('--pipeline', type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='r40-mmio-') as temp:
        image = Path(temp) / 'grid.bin'
        subprocess.run(['python3', 'tools/asm/tnca_asm.py',
                        'tb/core/progs/r40_grid_mmio_memory.s', '-o', str(image.with_suffix('.hex')),
                        '--size', '256'], cwd=ROOT, check=True, timeout=30)
        code = b''.join(int(line, 16).to_bytes(8, 'little')
                        for line in image.with_suffix('.hex').read_text().splitlines())
        dev = stand_in(core_count=args.cores, memory_ports=args.ports, host_timeout=1800,
                       noc_link_stages=args.pipeline, dispatch_stages=args.pipeline, home_stages=args.pipeline, code_stages=args.pipeline, board_top=args.board_top)
        try:
            dev.start()
            assert dev.request(8).status == 3
            assert dev.request(9).status == 3
            assert dev.request(15).status == 3
            assert dev.request(99).status == 3  # cannot truncate into a launch
            assert dev.request(17, args.ports).status == 3
            assert dev.request(12, args.cores).status == 3
            assert dev.upload(0, code).ok
            reply, data = dev.download(0, len(code))
            assert reply.ok and data == code
            if args.board_top:
                base=dev.link.read_reg(0xa8)
                for alias in (0x100,0x108):
                    dev.link.write_reg(alias,base)
                    assert dev.link.read_reg(alias)==base and dev.link.read_reg(0xa8)==base
            before = dev.physical_grid_metrics()
            rejected = dev.request(11, 0, 0x40000, 0x1000, 13 | (7 << 32),
                                   5 | (((1 << args.cores) - 1) << 32), 256)
            assert rejected.status == 3
            assert dev.physical_grid_metrics() == before
            timestamps = []
            for pc in (0, 256):
                reply = dev.launch_grid(pc, 0x40000, (13, 7, 5), arg_stride=256)
                assert reply.ok
                timestamps.append(reply.device_timing)
                metrics = dev.physical_grid_metrics()
                assert sum(row['issued'] for row in metrics['cores']) == 455
                assert all(row['issued'] == row['completed'] for row in metrics['cores'])
                assert metrics['peak_active'] == args.cores
            assert timestamps[1]['start'] > timestamps[0]['end']
            reply, data = dev.download(0x30000, 455 * 8)
            assert reply.ok
            assert [int.from_bytes(data[k:k+8], 'little') for k in range(0, len(data), 8)] == list(range(4096, 455+4096))
            counters = dev.memory_port_metrics()
            assert all(row['read_lines'] > 0 and row['write_lines'] > 0 for row in counters)
            # Stale doorbells cannot execute a mutated descriptor or duplicate
            # a completion. Read watermark and physical counts independently.
            prior = dev.physical_grid_metrics()
            served = dev.link.read_reg(R_SERVED)
            dev.link.write_reg(R_REQ_SEQ, served)
            dev.link.write_reg(R_REQ_OP, 11)
            dev.link.write_reg(R_REQ_A0, 0)
            dev.link.write_reg(R_DOORBELL, 1)
            assert dev.link.read_reg(R_SERVED) == served
            assert dev.physical_grid_metrics() == prior
            marker = (0xFEDCBA9876543210).to_bytes(8, 'little')
            assert dev.upload(0x50000, marker).ok
            marker_reply, marker_data = dev.download(0x50000, 8)
            assert marker_reply.ok and marker_data == marker
            assert dev.launch_grid(0, 0x40000, (0, 7, 5), arg_stride=256).ok
            zero = dev.physical_grid_metrics()
            assert zero['peak_active'] == 0 and all(row['issued'] == row['completed'] == 0 for row in zero['cores'])
            if args.large_grid:
                # Same three-tile image, same home/port budget, successive masks.
                # Read every produced slot: counts alone cannot detect a duplicate
                # PID that silently replaces a missing PID.
                for mask in ((1 << n) - 1 for n in range(1, args.cores + 1)):
                    assert dev.upload(0x30000,bytes(4103*8)).ok
                    assert dev.launch_grid(0, 0x40000, (4103, 1, 1), active_cores=mask).ok
                    large=dev.physical_grid_metrics()
                    assert sum(row['issued'] for row in large['cores'])==4103
                    assert all(row['issued']==row['completed'] for row in large['cores'])
                    assert large['peak_active']==mask.bit_count()
                    assert all(row['issued']==0 for i,row in enumerate(large['cores']) if not (mask >> i) & 1)
                    result,slots=dev.download(0x30000,4103*8)
                    assert result.ok
                    assert [int.from_bytes(slots[i:i+8],'little') for i in range(0,len(slots),8)]==list(range(4096,8199))
                    print(f'LARGE PHYSICAL GRID PASS programs=4103 active_mask={mask:#x} metrics={large}',flush=True)
            old_deadline = dev.deadline_cycles
            dev.deadline_cycles = 20
            timeout_reply = dev.launch_grid(0, 0x40000, (13, 7, 5), arg_stride=256)
            dev.deadline_cycles = old_deadline
            assert timeout_reply.status == 2
            drained = dev.physical_grid_metrics()
            assert 0 < sum(row['issued'] for row in drained['cores']) < 455
            assert all(row['issued'] == row['completed'] for row in drained['cores'])
            assert dev.quiesce().ok
            print(f'MULTICORE MMIO PASS cores={args.cores} ports={args.ports} '
                  f'programs=455 timestamps=2 counters={counters}', flush=True)
        finally:
            dev.stop()


if __name__ == '__main__':
    main()
