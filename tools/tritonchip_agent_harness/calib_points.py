"""Calibration cases of the R41 development set that are not development points (decided 2026-11-23).

The calibration set is small and complete in coverage: every mechanism a development point reaches in the CA
model's counters, reached by some case the RTL simulator can run (the trace gate compares the two cycle by
cycle). The development set's small points and the test-case skill's cases reach all of them but two kinds:
a working set past the 96 KiB of home cache with dirty lines written back, and back-pressure on the memory
path (request stalls, full response or writeback queues). These two cases reach them, measured on the CA model
for the base on 2026-11-23: the copy at 11,334 cycles; the gemv at 362,031 (a few hours in the RTL simulator,
the smallest weight stream found that reaches back-pressure; k256 n1024 at 178,023 cycles did not).

Registered as ArchBench points of tier `membound`, like dev_standins.py; dev_cases.py exports them as cases
marked `calibration`.
"""
import torch
from archbench.ops import _linear, gen, op, randn
from archbench.registry import Demand

CALIBRATION = []


class Copy(torch.nn.Module):
    """A tensor copied into a new one: streaming read and write, nothing else."""

    def forward(self, x):
        return x.clone()


N = 65536
op("copy", "membound", "fp16", dict(n=N), lambda seed, n=N: (Copy(), (randn(gen(seed), (n,), "fp16"),)),
   Demand(read_bytes=2 * N, write_bytes=2 * N))
CALIBRATION.append("op.copy/membound/fp16-n65536")

K, NN = 512, 1024


def _mk_gemv(seed, k=K, n=NN):
    g = gen(seed)
    return _linear(k, n, "fp16", g, False), (randn(g, (1, k), "fp16"),)


op("gemv", "membound", "fp16", dict(k=K, n=NN), _mk_gemv,
   Demand(macs=K * NN, read_bytes=(K + K * NN) * 2, write_bytes=NN * 2))
CALIBRATION.append("op.gemv/membound/fp16-k512-n1024")
