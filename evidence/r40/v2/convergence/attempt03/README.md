# Current conditional delivery status — 2026-10-19

Required gate delta: **no new final-candidate gate promotion**. Historical
`R40-01.device`, `R40-01.mixed-device`, and `R40-05.multicore` pass on immutable
product tree `44051ad9…d2436`. The complete original graphs3/model zero-transfer
observations now pass the strict residency auditor on that same historical
source. The repaired runtime has product identity `80dd1a82…b6e4ca` and requires
fresh matching evidence. See `required_gate_ledger_reserved_1a7c643.json` for
every required gate's blocker, direct evidence, next command and outcome branches.

The qualified image remains `a88ebdf5…c83083`, at **90 MHz** with routed setup
WNS +0.016 ns, three physical SLR cores and three outlets. 100 MHz failed and
is not claimed. Both complete host conformance logs and the complete original
prefill512/decode64 model report are retained with their original identities.
CI run `39164382436` passed at `6bd07b7`; the current source needs its own CI.

## Original large-model failure and controlled allocator repair

Original large Llama decode error 0.11417 and prefill error 0.15730 exceeded
the unchanged 0.05 tolerance. Exact native-host weights and inputs were checked
against full FP32 references computed on GB10. Expanded-output diagnosis found
all 16 layer outputs and KV correct; only logits failed. The actual output
weight allocation crossed the TMU register page `[0x7fff0000, 0x7fff1000)`.
Host BO writes bypass this decode, whereas core reads are intercepted as MMIO.

Commit `1a7c643` excludes that page from code/heap allocation, free-space reuse,
recovery and usable-capacity accounting. The original unmodified decode graph
now passes both cold and steady calls: logits error **0.00271378**, new keys
0.00221558 and new values 0.00232000. Original weights, inputs, shape, tolerance,
image and operating clock were retained. All 363 original cold launches have
the same order, grids and executable payloads. Eleven container hashes differ
only in source-provenance fields; their payloads are byte-identical. Full
container identities and verified comparisons are preserved rather than relabelled.

Original prefill replay is running. Its success unlocks the serial repaired
snapshot queue: actual-window probes, complete native77/HBM14/full frozen model,
physical capability and host conformance, original large24, and all chained
transfer observations. A failure stops the queue and preserves that stage.
These functional runs do not establish a performance gate or fixed-bucket decode.
The earlier formal performance queue stopped before measurements when large24
failed; it did not produce performance evidence.

## Contract decision

The user authorized one workload-justified v3 amendment on 2026-10-19.
`contract_review_20261019.json` records why **no additional relaxation is yet
justified**: the numerical failure is repairable, fixed-bucket compilation remains
useful, and missing implementations/measurements do not prove a performance
threshold unreasonable. No v3 contract has been frozen. The v2 lock and all
original shapes/tolerances remain unchanged. The existing user-authorized
conditional 90 MHz qualification remains explicitly separate from 100 MHz.

## Archived implementation launch and prerequisite record

# Final authorized three-SLR implementation (historical launch)

Required gate delta: none. The final authorized complete implementation started
at 2026-10-17 21:20:13 UTC. The budget is maximum 3, used 3, remaining 0.
No implementation retry or board load is authorized by this launcher.

`launch.json`, `reservation.json` and `launch_preflight.json` retain the actual
launch and exclusive budget reservation. `image.tcl`, `image.json` and
`preflight.json` bind the configuration and all 105 source inputs to software
commit `54249f7f7ac0cf8f5d1b82d8e63a3046d27522d3`. The source archive SHA256 is
`6b254137501f4a3802c11f9e084e961c60724093a4da5654ce721b70a0d8db84`.
The archive consists of that Git snapshot plus the three explicit configuration
files; subsequent evidence-only commits are not a different hardware source.

## Root-cause hypothesis and prerequisite evidence

The candidate retains three physical cores, three memory outlets, HBM[0:31],
32 sets, eight ways, full tags, eight MSHRs and 16 targets, two-stage links,
the original 100 MHz undivided clock and `SSI_SpreadSLLs`. It cuts the observed
owner/victim/writeback cone at synchronous tag outputs, owner snapshots,
pair minima and the selected-way register. Local coherent TMU copies replace
cross-SLR metadata broadcast; accepted notices drain before an acknowledged
metadata update. Byte-enabled cache writes remove the wide merge/rewrite path.
Targeted SLL guidance applies only to eligible existing response register pairs.
No timing exceptions or workload reductions were introduced.

The routed predecessor module has setup WNS +0.361 ns and hold WHS +0.010 ns,
zero TNS/THS and routing errors, 96 RAM32M16 tag roots, and actual local TMU
state sites in their respective SLRs. Its new worst sampled path is MSHR index
to data RAM: 18 levels, 9.203 ns (1.446 logic / 7.757 routed net). The owner-to-pair
cut is 16 levels / 7.654 ns, pair-to-selected 7 / 3.471 ns, and selected-to-writeback
9 / 5.832 ns. These sparse module measurements exclude cores, NoC, HBM and timed I/O.

The final one-line completion removes the inactive-state live owner mux without
adding state or cycles. Exact before/after data, hint and counter cycle comparisons
are retained in `../p0_selected_victim_tlw/owner_boundary_complete.json`.
The exact final synth/opt netlist contains 3,288 metadata FFs, 195 notice FFs and
24 request-owner snapshot FFs, with zero direct metadata-to-notice combinational
paths. Snapshot-to-notice is 11 levels / 3.033 ns with estimated nets; it is not
routed timing. `prerequisites.json` and the raw archives preserve both identities.
The final completed source did not receive another module placement/route;
its first full placement/route is this final authorized run.

## Current blocker, next experiment and branches

The unique physical blocker is the absence of a qualified final-image route,
100 MHz setup/hold, CDC and loadable xclbin. Direct evidence is the launched
manifest with null image SHA and not-measured timing/CDC, and attempt 2's
terminal failure. On the FPGA host, inspect this same run without restarting it:

```sh
tail -n 30 ~/tnca_build/r40_attempt03_20261017/xo/link.log
cat ~/tnca_build/r40_attempt03_20261017/build/attempt.exit
```

The exit marker does not exist while the run is active. The existing read-only
observer waits for terminal completion, then collects reports without concurrent
large-DCP inspection. The current observer is recorded in
`observer_replacement.json`; only the waiting reporter was replaced. Its separate
immutable tooling bundle does not alter the frozen hardware source or launch.
It also extracts the final owner/pair/selected register segments and actual
tag mapping. `final_cut_query_probe.json` retains the successful read-only query
on the saved predecessor module DCP, with the raw archive and all 20 file hashes.
Missing required stages or timing segments fail extraction; an empty direct
owner-to-selected report remains an absence, not a zero-delay measurement. Success requires actual placement/route, all-clock min/max
timing, CDC and complete image identity before board loading. Failure preserves
its terminal checkpoint and exact limiting paths; there is no remaining full
implementation budget and no automatic retry.

Exact-source CI runs in parallel and must pass before image qualification or
board loading. After physical qualification, run the immutable original >=64 MiB
Torch checkpoint, then complete native77, HBM14 and frozen models. All final-image
measurements remain not-measured until collected on that exact image.
