# R41: standard Torch in an exact 8 GiB model window

Development evidence from GB10 (`gx10-6646`), using the native SystemC/TLM
executor and the fast and slow Ramulator2 DRAM paths. Host RSS and wall times
are measurements. Device cycles are uncalibrated predictions. No FPGA was
accessed; the map is **planned allocation, not measured**.

## Address convention

[planned-map/map.txt](planned-map/map.txt) describes **one 8 GiB BO over
PC0–PC31**, 256 MiB per PC, extrapolated from the recorded B40 allocation's
connected memory group. The recorded local limit is `0x20000` (128 KiB).
For a logical DRAM address `L`, BO offset/physical address is `L - 0x20000`;
PC is that offset divided by `0x10000000`. The reserved TMU page remains
`[0x7fff0000, 0x7fff1000)`.

The planned BO's logical interval is `[0x20000, 0x200020000)`.
**Every final Torch case uses `TNCA_TLM_MEMORY_BYTES=0x200000000` exactly**:
the model's absolute interval is `[0, 0x200000000)`, so its DRAM portion
excludes the planned BO's final 128 KiB. “Near the top” in the Torch results
means the top of this exact model window. A separate native run opens the
full BO top (`0x200020000`) to check its final eight valid bytes and the
first invalid byte. These two bounds must not be conflated.

The map exporter reproduces the recorded 4 GiB map, SHA256
`a1019970fe9548fda5e9b7e44442a14f1a886decd397f472c498dd15f5970bc2`.
The planned 8 GiB map SHA256 is
`d8b8f31de63937643e13002912d4a4d7dbbc528d787445bea325fbcd83c50180`.
Its [audit report](planned-map/report.json) retains the recorded source and
explicit planned status. It establishes neither a board allocation nor
bank/row mapping or calibrated controller timing.

## Execution and numerical checks

`tools/performance_harness/hbm8_window_case.py` uses ordinary Torch tensors
on `tnca` and `torch.compile(backend='inductor', fullgraph=True)`. Empty
storages control allocation placement without touching their bytes. Native
model launches and per-PC DRAM byte deltas are retained for every graph.
CPU tensors supply independently generated inputs and references; device
arithmetic executes in the model.

- Placement: 1 MiB operands in PC16 and PC31, output in PC31 within the
  final 8 MiB; `2*a+b`, sum, maximum and minimum checked independently.
- Crossing: a 16 MiB result is stored by one graph and loaded by another,
  crossing both logical `2^32` and physical PC15/PC16 at `0x100020000`.
  Elementwise outputs and extrema match CPU bitwise.
- GEMM: BF16 `96x80x160` mm and FP16 `33x31x65` addmm, with operands and
  outputs in the upper half. Independent CPU FP32 references use
  `rtol=atol=2e-3`; per-K FMA bit comparisons are also retained separately.
- Stream: source **and** destination are each one **2,214,592,512-byte
  (2 GiB + 64 MiB)** FP32 storage. Device views process 64 MiB per command.
  Device partial sum/max/min results are combined by another device graph;
  CPU float64 sum tolerates `1e-4` relative error, extrema are exact.
  A compiled `dst.copy_(src)` graph streams the copy, including view
  addresses/offsets beyond `2^31`. Every copied byte is checked bitwise
  against independently regenerated CPU input, staged in 16 MiB chunks.
  The copy crosses physical 4 GiB; graph counters must show every PC16–PC31.
- The large stream also uploads and retains 1.375 GiB of separate device
  residency. Source + destination + extra residency total **5.5 GiB touched**;
  graph input/output footprint alone is 4.125 GiB. Empty padding is excluded.

## Boundaries and refusals

`test_hbm8_native.py` checks 2,832 output bytes per path and requires exact
per-PC counters from the independent Python map translator. Both the exact
8 GiB window and the full-BO window are tested. Each campaign includes 20
negative rows: device access at the window top, host/device one-byte access
at BO top (one past its last valid byte), a word crossing BO top, accesses
to either end of the TMU hole, host writes into the hole, a DRAM load below
the local limit, an overlarge mapped window, and baseline/legacy refusals.
Slow child failures deliberately hide the child's stderr; their parent
refusal and the identical artifact's specific fast-path refusal are retained.

Torch additionally checks allocator exhaustion, reservation of the TMU page,
placement of a would-straddle storage above the hole, correct arithmetic
there, valid access to the exact window's final eight bytes, and refusal of
a raw upload crossing its top. Refusal terminates that model session.

Slow Torch numerical passes do **not** establish timing convergence. The
retained legacy-profile logs include unconverged commands and GEMM unmatched
timing requests. The device-service smoke also records an inconclusive
fixed point. The final summary independently classifies each case; numerical
passes retain `inconclusive-fixed-point` whenever these exceptions occur. Native boundary campaigns use eight passes and report
`uncalibrated-fixed-point-checked`. No numerical pass is promoted to a timing
qualification.

## Final results (device-service profile, exact 8 GiB window)

[summary.json](summary.json) is produced by `summarize.py --raw <extracted
tlm8-logs.tgz>/hbm8-tlm`; it re-asserts every check below from the raw reports.

| Case | Fast | Slow (Ramulator2) | Slow timing status |
|---|---|---|---|
| placement (PC16, PC31, top 8 MiB) | pass, 4 checks | pass, 4 checks | inconclusive (2 unconverged, 1 unmatched) |
| crossing (store/load across 4 GiB) | pass, 5 checks | pass, 5 checks | inconclusive (4 unconverged, 48 unmatched) |
| gemm (BF16 mm, FP16 addmm, upper half) | pass, 4 checks | pass, 4 checks | inconclusive (2 unconverged, 3 unmatched) |
| refusal (Torch negatives) | pass, 7 checks | pass, 7 checks | inconclusive (1 unconverged) |
| stream 5.5 GiB touched, 67 graphs | pass, 6 checks | pass, 6 checks | inconclusive (165 unconverged, 12,221 unmatched; 293 slow commands) |

Both streams have per-PC traffic on every PC16-PC31, a bitwise-equal
2,214,592,512-byte copy, and extrema plus sum checks against CPU. Native
campaigns: 2 positives and 20 refused negatives each for the exact window and
the full BO top. Together these cover one byte past the BO top, the window
top, and the TMU hole. All slow cases are **numerically** correct.
None is timing-qualified: the replay fixed point was checked but
did not converge everywhere. No device-cycle result is claimed.

## Resources and limits

Measured on GB10 with six CPUs. KiB values from the reports, shown in GiB:

| Run | Touched | Path | Parent RSS | Slow child RSS | Peak summed RSS / PSS | Wall |
|---|---|---|---|---|---|---|
| prior scale-256 † | 0.5 GiB | fast | 0.76 | - | - | 73 s |
| prior scale-256 † | 0.5 GiB | slow | 1.59 | 1.38 | - | 472 s |
| prior scale-1024 † | 2 GiB | fast | 3.02 | - | - | 283 s |
| prior scale-1024 † | 2 GiB | slow | 6.33 | 5.59 | - | 2,271 s |
| final stream | 5.5 GiB | fast | 5.53 | - | 6.31 / 6.11 | 1,036 s |
| final stream | 5.5 GiB | slow | 5.74 | 5.71 | 12.15 / 6.43 | 5,059 s |

† Earlier diagnostics. They use the legacy host-gap profile, a 0x200020000
window and unchunked storages. They show the scaling trend only.

With bounded chunks, model parent RSS tracks the touched footprint at about
1.0-1.05 GiB per touched GiB. The earlier unchunked runs needed about 1.5x. The slow path adds a forked Ramulator2 replay child of similar
RSS. Most of its pages are shared, so summed RSS is about 2.2x touched, but
PSS is only about 1.2x. **A Qwen-sized run (5-6 GiB touched) fits GB10's
limits on both paths.** Measured at 5.5 GiB, peak summed RSS is 6.3 GiB on the
fast path and 12.2 GiB on the slow path. Both are under the 16 GiB shared-host
budget and far below GB10's 119 GiB physical memory. Linear extrapolation
to 6 GiB on the slow path gives about 13.3 GiB summed RSS. That figure is a
prediction, not a measurement. The practical limit is wall time. Streaming the
5.5 GiB footprint once took 17 minutes on the fast path and 84 minutes on the
slow path. A full Qwen inference re-reads its weights for every token, so it
would take correspondingly longer. That time was not measured.

The earlier legacy-profile 5.5 GiB fast stream passed in 727.059 seconds
with 5.53 GiB peak model RSS. Its slow counterpart was interrupted before
its first full graph completed. Under the legacy host-gap charges, each tiny
replay processed about 45 million refreshes. This is why the final runs use
the device-service profile. The resource runner restricts affinity to six CPUs,
requires at least 40 GiB initially available, and stops only its owned PIDs
at 16 GiB summed process-tree RSS. Shared pages are counted in each process's
RSS; PSS additionally reports their physical sharing. Model parent RSS, slow
child RSS, combined PSS, Python RSS, wall time, and raw samples are retained.
Sampling periods are one second inside the case and two seconds in the guard;
short peaks can be missed. Parent VmHWM and Python ru_maxrss supplement them.

Wall time includes cold compilation, initialization, references, transfers,
and verification. Graph wall times include compilation on first use. Neither
is a device performance measurement. A residency/stream test does not measure
an entire Qwen inference or establish its end-to-end execution time.

[tlm8-logs.tgz](tlm8-logs.tgz) is the complete raw archive. It holds every
`final/` run, including all service-profile cases with model counters,
fixed-point logs and resource samples. It also holds the interrupted legacy
slow and fast attempts, both native campaigns, the build and unit logs, the
4 and 8 GiB maps, the prior diagnostics,
`configuration-diff.json` and `source-manifest.json`. Compiler caches and
build trees are excluded. The earlier interim
[boundary-and-fast-logs.tgz](boundary-and-fast-logs.tgz) is kept unchanged;
it additionally holds the map exporter logs.

## Reproduction and evidence

Source `~/tc_env/env.sh` on GB10 and use `~/tc_env/torchenv/bin/python`;
no packages were installed. Build Release with SystemC 3.0.2 and the pristine
Ramulator2 revision in `systemc/ramulator/PIN`, using at most six build jobs (the verification build used two):

```sh
cmake -S tools/performance_harness/systemc -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/tc_env/systemc-3.0.2/install" \
  -DRAMULATOR2_DIR="$HOME/tc_env/ramulator2"
cmake --build "$BUILD" -j 4
```

A fresh Release build reproduced the exercised model binary byte-for-byte:
`f2da72a2a9b3d0d89c5a2a28f1c555ae85c40c8cb6ec8ee7b1cb2d0880826d02`.
The native mapper tests include the 8 GiB boundary geometry, and all Ramulator2
bridge unit checks pass. Build and unit logs are retained.

For Torch, use these environment values (all case reports retain the exact
knobs, source identity, binary/configuration/library hashes and tool versions):

```sh
export PYTHONPATH="$PWD:$PWD/python:$PWD/compiler/python:$PWD/tools/asm:$PWD/tools/performance_harness/systemc"
export TNCA_BACKEND=systemc-tlm-grid TNCA_TLM_BINARY="$BUILD/tnca-chip"
export TNCA_TLM_CONFIG="$PWD/tools/performance_harness/systemc/configs/r41-hbm8-device-service.cfg"
export TNCA_TLM_MEMORY_BYTES=0x200000000 TNCA_DRAM_MAP="$MAP/map.txt"
export TNCA_TORCH_DEVICE=native TNCA_TENSOR_DESCRIPTORS=1
export TNCA_PROFILE=core-vrf-sync-v5 TNCA_ACTIVE_CORES=3
export TNCA_DEADLINE=1000000000000 TNCA_HOST_TIMEOUT=86400
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
```

The final Torch profile copies the M2 configuration and changes only
`host_launch_cycles` and `host_transfer_cycles_per_kib` to zero
(`host_transfer_cycles` was already zero). Host OS wall measurements still
include actual transfers and launches. This removes the opaque fitted host
gaps from the device-service simulation; device/DRAM settings and the model
binary are identical on both paths. The original M2 default is unchanged.
`configuration-diff.json` in the raw archive records this exact difference.

For each `TNCA_DRAM_PATH=fast` and `slow`, run placement, crossing, gemm,
refusal, then stream, each in a fresh output directory. For the large stream:

```sh
export TNCA_DRAM_PATH=slow  # repeat with fast
~/tc_env/torchenv/bin/python evidence/r41/hbm8/tlm/resource_runner.py "$PREFIX" \
  ~/tc_env/torchenv/bin/python tools/performance_harness/hbm8_window_case.py \
  --case stream --stream-bytes 2214592512 --stream-chunk-bytes 67108864 \
  --resident-extra-bytes 1476395008 --map-audit "$MAP/report.json" --out "$OUT"
```

The manifest is `spec/r41/hbm8/experiments/tlm8-window.json`. Its native step
checks the exact window by default; repeat with `--memory-bytes 0x200020000`
to reach the full planned BO top. The retained earlier unchunked scale runs
are separate diagnostics; the current stream case uses bounded views.

The incoming WIP used `0x200020000` for Torch. Prior 64/256/1024 MiB scale
runs are retained under `prior/` in the raw archive and are labelled as that
larger logical window; they do not substitute for the final exact-window
runs. The first large attempt was interrupted after review found that its
copy placement would skip PC21–PC22. Its raw `running` report, nonzero process
exit and interruption note are retained; it is excluded from passing results. The later legacy-profile slow attempt
is likewise retained as interrupted for excessive host-gap replay overhead.

Compiler/runtime/assembler and original-point sources are byte-identical to
`c2c96d3`, whose retained `default_change_audit.py` result has **113 unchanged,
0 changed, 0 not compared**. `default-code-continuation.json` records the
source comparison and audit hash. This is continuation by source identity,
not a newly executed 113-point compilation. No default compiler change, model-
specific special case, or host numerical fallback was added.
