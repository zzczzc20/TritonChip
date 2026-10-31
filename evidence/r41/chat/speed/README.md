# R41 chat speed work

The active-event runtime change and this candidate were developed outside the
fixed board archive. All numbers here are uncalibrated DP SystemC/TLM; no
board speedup is claimed.

## Equal-row lowering

On a DP target the compiler now proves equal row addresses and masks before
sharing a load or row-wise computation; it keeps the shared row in scratch
to control register lifetime. `TNCA_EQUAL_ROWS=0` is the control. The
[matched official fused FP8 results](equalrow_default_20261031.tgz) used
the default division setting, fresh processes and independent exact DP
references. These steady calls each used one launch and had zero code loads
and evictions:

| M×N×K | Instructions/MAC before → after | Device cycles before → after | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.70867919921875 → 0.63916015625 | 161804 → 118469 | 0/256 each |
| 1×250×384 | 0.7098645833333334 → 0.6386770833333333 | 256585 → 195907 | 0/250 each |
| 1×1024×256 | 0.70867919921875 → 0.63916015625 | 536417 → 408542 | 0/1024 each |
| 16×256×256 | 0.047428131103515625 → 0.047428131103515625 | 186575 → 186575 | 0/4096 each |

The all-zero activation also matched bitwise. The FP8 and compiled-graph
suite passed 16/16 and the isolated candidate's 113-point default audit
compared 113 unchanged ([gate records](equalrow_gate_partial_20261031.tgz)).
Four [mask and wide-reduction cases](equalrow_mask_20261031.tgz) with equal
or differing addresses and masks matched bitwise with the optimisation off
and on. An [actual integration-tree TLM run](equalrow_integrated_20261031.tgz)
reproduced the 118469 cycles and 0/256 mismatches, with assembly SHA-256
`f59e9b8010bbffb3a1768119a318643610532deaa85875b69aec093e26ce0f7c`
identical to the isolated final candidate.

The [first integration attempt](equalrow_integrated_failed_20261031.tgz)
failed because `os` was not imported in that source tree; the successful
retry imports it locally. Two [integration audit attempts](equalrow_integration_audit_failed_20261031.tgz)
had 0/113 passing because the copied tree lacked the ISA file and then the
`tools/r38` files, respectively. These are not passing audits. The corrected
[actual integration-tree audit](equalrow_integration_audit_20261031.tgz)
compared 113/113 unchanged, with zero changed or uncompared. Board comparison
is pending.

## Pipelined weight transfers

The DP X8 path now saves fallback gather indices in scratch, reducing vector
register pressure so the existing next-weight transfer pipeline can be
allocated. Two fallback scratch tiles are alternated: a tile is reused only
after its prior transfer's barrier has completed. This matters for consecutive
noncontiguous N chunks, since a core fence does not wait for asynchronous DMA.
`TNCA_X8_SPILL_LANES=0` is the control.

The [DP TLM records](pipeline_double_20261031.tgz) compare against the
equal-row version above, with default settings and exact DP reference. Steady
calls all used one launch and zero code loads/evictions:

| M×N×K | Instructions/MAC before → after | Device cycles before → after | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.63916015625 → 0.64068603515625 | 118469 → 113675 | 0/256 each |
| 1×250×384 | 0.6386770833333333 → 0.6394895833333333 | 195907 → 190395 | 0/250 each |
| 16×256×256 | 0.047428131103515625 → 0.04752349853515625 | 186575 → 182082 | 0/4096 each |

The 1×7×256 consecutive-wrap case also matched 0/7 bits. The FP8 and
compiled-graph suite passed 16/16, and the 113-point default-code audit
compared 113 unchanged, zero changed or uncompared. The uncalibrated TLM
result does not establish a board speedup; board A/B is pending.

## Experimental E4M3 K32 retiling (not merged)

The DP TLM profile was used for these **diagnostics**, not the board. The
existing chat FP8 fused path already emits K32. A separate worktree explored
padding K16 with negative-zero times positive-one and splitting packed K64
into two ordered K32 commands. [Raw diagnostic records](k32_pad_packed_20261031.tgz)
and the [research patch](k32_research_20261031.patch) are retained. The patch
is based on the experimental worktree and also includes its earlier X8 lane
spill experiment; it is not an integration patch.

A standard Torch `torch._scaled_mm` 16x16x64 test, using a forced 16x16x64
GEMM tile and non-default `TNCA_WINDOW_REUSE=1`, required experimental handling
of packed FP8 row chunks on both sides: K64 baseline used
`TNCA_FP8_PACKED_ROW_CHUNKS=1`, K32 used `TNCA_FP8_PACKED_K32=1`. These are
not unchanged default-code measurements. Each result was compared bitwise
with the independent DP reference.

| Steady route | Instructions/MAC | Device cycles | Launches | Code loads/evictions | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| K64 | 0.0552978515625 | 8355 | 1 | 0/0 | 0/256 |
| K32 + K32 | 0.05908203125 | 8227 | 1 | 0/0 | 0/256 |

The K32 assembly contains two DP K32 commands and no K64 command; the
baseline has one K64. An additional transposed-weight AOT test with nonzero
accumulator compared signed zeros, mixed finite values and NaN/infinity
against the exact DP reference: each case had 0/256 mismatches. The
[raw Torch and special-value runs](k32_torch_20261031.tgz) include compiled
assembly and results. The small cycle difference does not establish a chat
speedup; larger shapes, default-path behavior and the 113-point audit remain
unverified. Earlier forced-K64 Torch compilation without the experimental
row-chunk handling was rejected, so that failure is not presented as a
successful baseline. No board retest is queued for this ungated experiment.

The same forced-tile Torch diagnostic was extended to three boundary shapes;
[raw records](k32_torch_shapes_20261031.tgz) report the second call:

| M×N×K | Instructions/MAC K64 → K32 | Cycles K64 → K32 | Launches | Loads/evictions | Bit mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×16×64 | 0.802734375 → 0.86328125 | 8218 → 8087 | 1 → 1 | 0/0 → 0/0 | 0/16 each |
| 1×17×64 | 4.5900735294 → 4.7352941176 | 21388 → 21245 | 1 → 1 | 0/0 → 0/0 | 0/17 each |
| 16×17×128 | 0.2447150735 → 0.2537913603 | 43652 → 43491 | 1 → 1 | 0/0 → 0/0 | 0/272 each |

These show slightly fewer cycles but more instructions, and do not change the
non-default configuration or acceptance caveats above. The [initial shape
runner log](k32_torch_shape_initial_20261031.log.gz) also preserves its path
collision mistakes and the explicit rejection of a forced K64 tile without
window reuse.

## Experimental frequency-aware code eviction (not merged)

A generic [experimental runtime patch](code_cache_lfu_experiment.patch) ranks
resident code by recent artifact use, aging counts every 256 lookups. The
control is the existing least-recently-used eviction. The
[diagnostic runner](../../../tools/r41/chat_speed/bench_code_cache_dp.py)
replays the recorded legacy VL one-layer decode load-request order, mapping
its kernels to unique existing DP artifacts by name or closest size. It
loads those artifacts but does **not** execute them; after each sequence it
executes one compiled E4M3 DP dot and checks all 256 FP32 bits against the
independent exact DP reference. Thus this is a synthetic code-pressure test,
not a full-model trace or a board result.

In 11 measured steady repetitions (23 mapped load requests and one dot launch
per repetition), the control → candidate totals were 203 → 187 code loads,
202 → 187 evictions and 2,634,816 → 1,229,888 uploaded code bytes. The dot
had 0 mismatches in all 2,816 output elements for each route, 0.1002197265625
instructions/MAC and 7,271 device cycles per launch for both. The candidate
used non-default `TNCA_CODE_CACHE_POLICY=lfu`; it is not integrated. Raw
results, artifact mapping and [earlier failed setup attempts](code_cache_synthetic_20261031.tgz)
are preserved. Real reduced-depth model profiling and a default-code audit
are running; no full-model speedup or board acceptance is claimed.
