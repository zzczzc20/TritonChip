# R41: qualify an 8 GiB window on the three-core B40 image

**Conclusion: the current B40 image supports an 8 GiB window, subject to the
follow-on board run. No new image is indicated by the RTL audit or simulation.**
The remaining qualification is whether XRT can allocate the full group BO on
this host and whether the platform routes PC16–PC31 for all three masters.
That board experiment is registered below and **has not been started**.

Image: `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`,
UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`, 90 MHz, three cores.
Its link configuration connects hbm0/1/2 to HBM[0:31]. The static xclbin parse
in [xclbin_hbm_map.json](xclbin_hbm_map.json) records all 96 connectivity rows
and a shared 8 GiB group (`MBG[0:31]`, memory index 41, base 0). These are
connectivity declarations, not measurements of the upper PCs on B40.

R40 exposed 4 GiB and certified PC0–PC15. The convergence record
(`evidence/r40/v2/convergence/README.md`, “Repaired original GEMM and declared
capacity”) documents a 4 GiB preflight and a workload needing about 3 GiB.
It records no hardware reason for stopping at 4 GiB. The evidence supports
an allocation/certification choice; it does not establish the original
operator's motive. The single-core `u280_r40ram` already passed an 8 GiB BO,
PC0–PC31 and accesses above 4 GiB (`evidence/r40/R40-04/ram_image/`). That is
precedent, not a substitute for the three-core board experiment.

## RTL address audit

The full [audit](audit_54249f7.md) gives source file:line references for each
carrier, including every narrower field and its effect. References are to
image-source commit `54249f7f7ac0cf8f5d1b82d8e63a3046d27522d3`, with the
attempt03 image parameters: three cores/outlets, 32 sets, eight ways, full
line tags and two-stage links. The current frozen RTL has seven changed
files, covering FPU corrections and registered control paths; the address
widths remain intact. No product RTL changes are made by this branch.

The LSU/TXF bases and effective addresses, request flit, MSHRs, victims,
outlets, aperture translation, MMIO range controls, handoff and m_axi ports
carry 64-bit addresses. The narrower fields do not discard an address bit:

| Narrow field | Reference at image source | Effect on 8 GiB |
|---|---|---|
| signed 32-bit gather/scatter index | `rtl/core/tnca_lsu.v:67,404–406` | added to a 64-bit base; limits one instruction's relative index, not the window |
| 53-bit cache tag, addr[63:11] | `rtl/nuca/tnca_nuca_slice.v:191,212–213`; `rtl/tile/tnca_tile.v:336–337` | full tag with set/line offset; no alias |
| 52-bit TMU page comparison | `rtl/nuca/tnca_nuca_slice.v:861`; `rtl/tnca_top.v:84` | only 0x7fff0000–0x7fff1000 is MMIO |
| 16-bit DCO tile hint | `rtl/nuca/tnca_nuca_slice.v:320–330`; `rtl/nuca/tnca_tmu.v:273–275` | replacement hint; not a data tag |
| 8-bit home hash | `rtl/noc/tnca_core_ni.v:83–91` | XOR of all bytes of the 64-bit line index, mod 3; routing only |
| 32-bit local-limit page count | `rtl/fpga/tnca_multicore_mmio.v:74` | describes the small local region, not the HBM bound |
| AXI-Lite 32-bit halves | `rtl/fpga/tnca_axi_lite.v:28–45,145` | assembled into a 64-bit register |
| TXF 32-bit iterators and 8-bit VRF offsets | `rtl/core/tnca_tensor.v:275,285,288,429` | local counts/VRF windows, not truncated device bases |

`tb/fpga/tb_line_axi_hbm8.v` passes 35 checks on image-source and frozen
`tnca_line_axi`. It checks legal read/write AXI addresses at 0/2/4/6 GiB
and 8 GiB−64; rejects 8/12/16 GiB offsets, below-local, unaligned and
64-bit-wrapping requests without an AXI transaction. BO bases are 0,
4 GiB and 2^64−8 GiB. The 33-bit AXI-address mutant is caught at BO base
4 GiB (base 0 alone cannot expose loss of physical bit 33). Removing
bounds or alignment checks is also caught. Raw logs: `unit/` and the archives.

## Standard Torch RTL simulation

`hbm8_case.py --sim` runs standard tnca Torch tensors and
`torch.compile(fullgraph=True)` through Inductor, Triton, the production
multicore MMIO backend and the three-core board top
`tnca_multicore_hbm_xo`. This is RTL simulation, not the TLM model.
`hbm8_sim.py` supplies a sparse AXI BO with a selectable physical base.
Only touched ranges are backed; an unmapped access is fatal. Alias probes
use distinct values so a lost bit mapping into another backed range also
fails. Physical PC labels in simulation are synthetic; PC routing is a
board question.

The placement helper uses the production allocator, untouched `torch.empty`
fillers, and a real launch-frame hole. Exact-end samples have a 64 KiB
prefix larger than that hole. The hole is released before advancing to the
final storage, preserving the runtime's 32 KiB launch reserve. Other cases
leave 1 MiB free at the top. No allocator or device fallback is used.

All **14 runs / 393 checks pass**, with one software-tree fingerprint
`files:e9948a967c92cfd0a45dfdc831c19a59bfe66ff585404d287d59880f989d2346`,
which also matches the delivered worktree. [rtl-results.json](rtl-results.json)
indexes the reports in [rtl-final-raw-logs.tgz](rtl-final-raw-logs.tgz).
The archive includes command lines, raw logs, generated kernels, source
hashes, resource-affinity records and both simulator summary reports.

| Case | What it checks | Image RTL, BO 0 | Image RTL, BO 4 GiB | Frozen RTL, BO 0 |
|---|---|---|---|---|
| window | 8 GiB aperture, register readback, heap and reserved page | 10/10 | — | — |
| samples | 34 fp16 DMA/device-add/DMA probes; every PC boundary, both BO edges and virtual 4 GiB | 76/76 | 76/76 | 76/76 |
| crossing | device store-load, DMA out/in, load again across virtual/physical boundaries and near the top | 17/17 | 17/17 | 17/17 |
| alias | distinct bit-30/31/32/32+31 pairs and displaced TMU pages; device loads then stores | 36/36 | 36/36 | — |
| bigoffset | one 2.25 GiB storage, strided reads/writes beyond 2^31 byte offsets | 6/6 | — | — |
| dma | 16 KiB DMA at the top, device copy to 6 GiB, both results read back | 6/6 | — | — |
| stream | 8 KiB across physical 4 GiB, DMA in, device row reduction, DMA out | 7/7 | — | — |
| negatives | allocator/host/MMIO bounds and wrap refusals; last word coherence | 10/10 | — | — |
| clamp | 16 GiB request exposes the 8 GiB group | 3/3 | — | — |

The final case/bench sources match the delivered worktree (hashes in the
archive); the image RTL hashes also match the FPGA-host source check.
The GB10 batch required at least 40 GiB available RAM before starting,
ran five cases concurrently, and used at most six CPU cores through
process-tree affinity. Observed aggregate RSS was about 4.4 GiB. Functional
checks have no wall-time or throughput gate.

Every final case checks that `R_AXI_STAT` has neither error nor stray status.
The sample and alias kernels verify all three cores; samples also verify
traffic on all three outlets. The crossing case performs a compiled device
store followed by a separate compiled device load across virtual 4 GiB,
physical 4 GiB and near the 8 GiB end, then checks DMA out, DMA replacement
and another device load. Samples reach the exact final 512 bytes of the BO.

### Review of WIP ca0cf45

The checkpoint was not accepted as a completed qualification. Its raw
records are preserved in [wip-raw-logs.tgz](wip-raw-logs.tgz), with
[wip-results.json](wip-results.json) as the index. Nine image-source/base-0
cases had `functional-pass`. Five supplemental runs completed their value
checks but ended `failed-preserved` because the software fingerprint changed
during execution. They remain failures. Earlier preserved failures include
allocator launch-reserve exhaustion, a fullgraph recompilation limit and a
probe grid using only two cores. The revised placement/grid in the WIP had
already repaired those base-0 cases; this follow-up shares placement with
the exact-end probes and asserts final AXI status. Final reruns use a stable
source snapshot; no failed record is overwritten or relabelled.

## Runtime and backend readiness

The production runtime/backend needs no change to expose this window:
pass `hbm_bytes=8589934592` with `hbm_dma=True` to
`fpga-u280-multicore`. `test_hbm8.py` uses the real xclbin topology with a
recording XRT accessor and register stand-in; it allocates no real BO.

* All three arguments must resolve to the same memory group. An 8 GiB
  request selects group 41 and maps PC0–PC31; 4 GiB maps PC0–PC15.
  A 16 GiB request is clamped to the 8 GiB group, with requested/exposed
  sizes recorded separately.
* `R_AXI_SIZE=0x200000000`, base and aliases read back at 64 bits. Device
  addresses span `0x20000` through `0x200020000` exclusive. DMA BO offsets
  equal `device_address−0x20000`, including above 4 GiB and the last line.
* The allocator reserves `0x7fff0000–0x7fff1000`. Usable heap capacity is
  8,589,930,496 bytes (8 GiB−4 KiB), with ordinary intervals
  `0x20000–0x7fff0000` and `0x7fff1000–0x200020000`. A 2.25 GiB storage
  fits after the hole; the largest contiguous storage is 6,442,643,456
  bytes. Rejected allocations leave state unchanged.
* Host transfer overruns are rejected before handoff or BO access. XRT's
  own offset/length guards reject negative spans and overruns. MMIO
  upload/download, host window and handoff negatives are checked in RTL.
  Device-side illegal accesses latch a sticky fault; only the RTL unit
  test sends them, never the shared card.
* Distinct bit-30/31/32/32+31 address pairs and the TMU page displaced by
  2/4/6 GiB check high-bit aliasing through device loads and stores.

Limits outside window capacity: same-device eager copies in this baseline
reject spans ≥2^31 bytes (`python/tnca/torch_device/copy.py`); allocator
`copy_storage` also has a <2^31-byte limit. Large storages here are filled by
chunked host DMA and compiled kernels. Host DMA stages through Python
objects and can require several times the transfer size in host RAM; board
cases use 64 MiB chunks. An 8 GiB XRT BO itself pins about 8 GiB of host RAM.
Compiler address-arithmetic work and TLM 8 GiB runs belong to the separate
`TritonChip-addr64` and `TritonChip-tlm8` worktrees. This branch does not
incorporate or claim their results.

## Follow-on board experiment: registered, not started

Generated manifests in `spec/r41/hbm8/experiments/`:

| Manifest | Cases | Estimated time, not measured |
|---|---|---|
| `board-hbm8.json` | window, negatives, samples, alias, crossing, dma, clamp, summary | 25 min |
| `board-hbm8-big.json` | bigoffset, stream, summary | 16 min |

Each case is a fresh process/session. Board DMA uses 256 MiB at the top
and 6 GiB. The big cases use one 2.25 GiB storage; stream transfers all
bytes in 64 MiB chunks and runs a device row reduction. Their failure is
kept separate from basic window qualification. `experiment.py` checks the
UUID, 90 MHz clock and idle state before and after each FPGA step.

[launches/hbm8_queue.sh](launches/hbm8_queue.sh) never programs, resets or
loads an image. Its gates are:

1. Wait until `~/r41s1_board_25094197f6e6/queue.log` contains
   `BOARD-QUEUE-DONE`, or that specific queue process has exited.
2. Require 15 consecutive idle minutes using the Stage 1 kdsstat rule:
   no contexts, outstanding executions or client PID.
3. Require UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8` and 90 MHz;
   stop on mismatch without loading anything.
4. Require at least 16 GiB MemAvailable, using a six-hour memory-wait budget and
   rechecking idle/identity while waiting (idle qualification itself can
   continue longer if the card stays busy).
5. Run the two manifests from an archived source tree. Changed identity or
   a busy card stops the queue. Logs preserve per-manifest exit status;
   `HBM8-QUEUE-DONE` means queue completion, not qualification success.

Fake-sysfs tests cover successful execution, waiting for a live Stage 1
queue to exit, exit without its done marker, restarting the idle period
after activity, wrong UUID, wrong clock, and insufficient host memory. The real
queue has not been launched. These are the exact commands for a future
operator to launch it from this worktree (they are **not executed here**):

```sh
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
C=$(git -C /home/ubuntu/TritonChip-hbm8 rev-parse HEAD)
git -C /home/ubuntu/TritonChip-hbm8 archive --format=tar.gz -o "$S/r41hbm8_$C.tar.gz" "$C"
rsync -e "$S/gb_rsh" "$S/r41hbm8_$C.tar.gz" /home/ubuntu/TritonChip-hbm8/evidence/r41/hbm8/launches/hbm8_queue.sh gb10:
"$S/gb" "scp -o BatchMode=yes r41hbm8_$C.tar.gz hbm8_queue.sh zzhouch@acf3019.ece.ust.hk: && ssh -o BatchMode=yes zzhouch@acf3019.ece.ust.hk 'setsid -f bash ~/hbm8_queue.sh $C > ~/r41hbm8_queue_$C.out 2>&1 < /dev/null'"
# Progress after an operator launches it:
"$S/fp" "tail ~/r41hbm8_board_$C/queue.log"
```

## Validation and reproduction

[validation-results.json](validation-results.json) indexes
[validation-raw-logs.tgz](validation-raw-logs.tgz):

* 26 runtime/backend/allocator, manifest, queue and summary tests: pass.
  Summary tests reject missing cases, mixed trees/modes, failed checks and
  AXI faults; simulator summaries explicitly require the board experiment.
* Registered manifests match their generator; queue shell syntax: pass.
* AXI adapter regression matrix: 35 configurations pass.
* Three-core dense AXI regression: two- and three-outlet configurations pass.
* Dense DMA ownership regressions: board top and shell top both pass,
  including dirty writeback, invalidation, partial-line preservation and
  range guards.
* Image-source 8 GiB adapter unit: 35 checks pass; inherited frozen-source
  unit also passes 35. All three address-path mutants are caught.
* [rtl-source-check.json](rtl-source-check.json): all 77 FPGA-host image
  RTL files match the git source. [default-code-invariance.json](default-code-invariance.json)
  records unchanged default code and the frozen 113-point suite digest.

No files in `python/tnca`, `compiler`, `rtl`, `xls`, `archbench`, `isa` or
`tools/asm` changed relative to the pre-WIP base `21d09f3`. Thus default
compiler/runtime code for the original 113 points is unchanged; a fresh
113-point compile campaign is not claimed. Added experiment specs change
the software-tree fingerprint. The bench defaults retain dense memory and
the existing BO base; regression checks cover the shared bench edits.

```sh
python3 tools/performance_harness/test_hbm8.py
python3 tools/performance_harness/hbm8_manifest.py --check
iverilog -g2012 -o /tmp/hbm8.vvp tb/fpga/tb_line_axi_hbm8.v rtl/fpga/tnca_line_axi.v
vvp /tmp/hbm8.vvp
python3 tools/r40/line_axi_tests.py
python3 tools/r40/multicore_axi_tests.py --cores 3
# GB10: source ~/tc_env/env.sh; use ~/tc_env/torchenv/bin/python.
# Extract git archive 54249f7 rtl into a separate directory for image RTL.
python tools/performance_harness/hbm8_case.py --sim --case samples \
  --rtl-root /path/to/image-source --hbm-base 0x100000000 --out /path/to/new-run
```
