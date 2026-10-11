#!/usr/bin/env python3
"""measure_clock.py SRC XCLBIN: the device clock, measured -- two spin kernels of
different lengths, each launch timed on the host; the difference in device
cycles over the difference in wall time cancels the fixed launch overhead.

The shell's frequency counter (clock_counter_freqs in sysfs) measures the
kernel clock; this measures the clock the device actually counts, after
rtl/board/tnca_u50_xo.v's divider. Run with the image already loaded
(program.sh --go). On 2026-10-09, the /2 image: 74.965 and 75.017 MHz, the
loop at exactly 6 cycles an iteration both times.
"""
import struct, sys, time
SRC, X = sys.argv[1], sys.argv[2]
sys.path[:0] = [SRC + '/python', SRC, SRC + '/tools/asm']
import tnca_asm
from tnca import backend

def image(n):
    a = tnca_asm.Assembler()
    a.assemble(f"""
        .li      s20, #{n}
loop:
        s.addi   s20, s20, #-1
        br.nz    s20, loop
        exit
""")
    return b''.join(struct.pack('<Q', w) for w in tnca_asm.build_image(a.records))

dev = backend.open_backend('fpga-u280', xclbin=X, deadline_cycles=10**11)
dev.start()
rows = []
for n in (5_000_000, 40_000_000, 5_000_000, 40_000_000):
    dev.upload(0, image(n))
    t0 = time.perf_counter(); r = dev.launch(0, 0); t1 = time.perf_counter()
    assert r.ok, r
    rows.append((n, r.r0, t1 - t0))
    print(f"n={n:>10d}  cycles={r.r0:>11d}  wall={t1 - t0:.4f}s  cycles/iter={r.r0 / n:.3f}")
for (n1, c1, w1), (n2, c2, w2) in ((rows[0], rows[1]), (rows[2], rows[3])):
    print(f"device clock: ({c2} - {c1}) / ({w2:.4f} - {w1:.4f}) = {(c2 - c1) / (w2 - w1) / 1e6:.3f} MHz")
