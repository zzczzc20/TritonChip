# Final authorized three-SLR implementation

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
large-DCP inspection. Success requires actual placement/route, all-clock min/max
timing, CDC and complete image identity before board loading. Failure preserves
its terminal checkpoint and exact limiting paths; there is no remaining full
implementation budget and no automatic retry.

Exact-source CI runs in parallel and must pass before image qualification or
board loading. After physical qualification, run the immutable original >=64 MiB
Torch checkpoint, then complete native77, HBM14 and frozen models. All final-image
measurements remain not-measured until collected on that exact image.
