# R40-01.device-time: per-point regressions on fe1b (diagnosis, not a gate record)

The fe1b campaign's device-time A/B (B40-CPU control: CPU tensors, per-kernel staging, 3 cores; candidate: native
`tnca` tensors, same knobs) gives S_geo 1.0015, 95% CI [0.9947, 1.0080] over the 77 points: the aggregate floor
(1/1.05) holds, but 4 points are slower than 5% and 4 more slower than 2%, so the gate's per-point rule fails.

| Point | CPU path cycles | native cycles | native/CPU speed |
|---|---:|---:|---:|
| op.softmax/tiny/fp32-r3-c37 | 1,951 | 2,112 | 0.924 |
| op.softmax/small/fp16-r64-c512 | 193,089 | 207,271 | 0.932 |
| op.gemv/tiny/fp16-k64-n64 | 11,840 | 12,632 | 0.937 |
| op.softmax/medium/fp16-r80-c512 | 246,385 | 262,655 | 0.938 |
| op.silu/tiny/fp16-n256 | 1,332 | 1,392 | 0.957 |
| op.matmul/medium/fp16-m128-k192-n128 | 232,980 | 239,018 | 0.975 |
| op.argmax/tiny/fp16-r2-v256 | 3,614 | 3,695 | 0.978 |
| op.matmul/tiny/fp16-m16-k32-n16 | 4,279 | 4,373 | 0.979 |

Causes, from an offline compile of the same points on both paths (fe1b tree, GB10,
`tools/r40/v3_compile_preflight.py --device cpu|native-stub`):

1. softmax: Inductor emits `tl.debug_barrier()` between a store and its reload inside the reduction loop for a
   non-CPU device; TNCA lowers it to `fence.acq_rel.device`. The native kernel has one more fence in the loop than the
   CPU-path kernel (softmax small: 2 vs 1; tiny fp32: 1 vs 0); everything else is the same instruction stream.
2. gemv/silu tiny: one launch, byte-identical kernel code on both paths, 4-5% more device cycles natively (gemv
   12,632 vs 12,043; silu 1,412 vs 1,353 in one record pair). Per-kernel staging places the staged buffers at different
   device addresses on the two paths (different cache sets / HBM PCs); on kernels of a few thousand cycles that shows
   as a few percent. Not a code generation difference.

Correcting (1) is a compiler change (lower the barrier to the ordering TNCA actually needs), i.e. a new product
identity; (2) has no code-level cause to correct.
