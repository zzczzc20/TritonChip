# R40-04 checkpoint — 2026-10-11, not accepted

`gates.json` preserves the simulation checks available at this checkpoint.
The package remains not-measured pending real HBM, timing/CDC, calibrated
model fidelity and frozen bandwidth/performance gates.

- AXI bridge: 42/42 (six geometries, seven cases each).
- Semantic mutations: 11/11 detected.
- Address planning: 9/9, including PC31 and a nonzero window base.
- Board wrappers: r36-fpga-tests 8/8 on gx10-6646.
- Protocol: 34/34 on the local tree.
- Vivado 2023.2 packages the v5 HBM configuration with OOC-only clock
  constraints and all requested profile parameters exposed by the wrappers.
- First HBM link failed: u280_r40a, WNS -0.683 ns; no xclbin was produced.
  Diagnostics and a fresh link are in progress; see doc/r40_hbm.md §7.

Raw simulation logs are under ~/tritonchip-r40-next/sim/build/r40 on
gx10-6646. Vivado outputs are under ~/tnca_build/u280_r40next on acf3019.

HBM device group revalidated 8/8 on gx10-6646: default native 16/16,
adverse native 5/5, capacity 24 MiB into the device, all 32 MiB model
samples, delayed/reordered samples, and line8 adverse/capacity/samples.
This is RTL simulation, not the physical 32-PC or board bandwidth gate.

Standard native Torch HBM result checkpoint: the original MLP tiny point
passes on core-vrf-sync-v5/hbm (101542 cycles, no host fallback). The bound
resolver now maps this memory mode to the existing architectural core
profile; its actual memory ingress remains 8 B/cycle, not HBM peak.
ArchBench selftest passes all tiny/small checks with HBM bounds included.
See native_hbm_mlp_bound_fix.* and hbm_bounds_selftest.log. This is RTL
execution and does not close any physical-board or performance gate.

2026-10-12 timing checkpoint: the fresh OOC-only-clock link still fails
100 MHz (WNS -0.209 ns, 2446 failing setup endpoints); automatic scaling
to 97.9 MHz is rejected. See hbm_100mhz_failed_timing.log. The optimized
descriptor row predicate is formally equivalent to the original 64-bit
wrapping sum/sign/bounds contract for all 160 input bits (Yosys 0.33),
with no added cycles. See tensor_bounds_formal.log and the three passing
transposed-transfer RTL cases in tensor_bounds_xpose.log. A new physical
implementation is required; this proof does not establish timing closure.

The current compiler RTL suite passes 68/68 after the bounds change;
see tensor_bounds_compiler_rtl_current.log. The initial isolated snapshot
had the superseded stride-refusal test, which failed as expected for a
now-supported legal stride; current tests retain both numerical stride
validation and invalid-rank fault propagation.

2026-10-12: bounds-optimized image still failed its unchanged 100MHz
link. bounds_link_timing.txt extracts the report summary and worst path;
bounds_link_timing.json records its full-report hash and source commit.
WNS -0.336ns, TNS -226.125ns, 1905 setup failures; automatic 96MHz
rejected. No image was programmed. The worst path is now LSU request
selection through local RAM write enable; follow-up pipelining is pending
functional validation and a new timing/CDC implementation.
Standalone hardware export also passes 3/3 after declaring the optional
AXI dependencies (export_axi_dependencies.log).

Local RAM pipeline checkpoint: 49/49 bridge cases (seven geometries),
13/13 semantic mutations detected including queued-tag and FIFO-pop
mutations, 8/8 FPGA wrapper tests and 4/4 representative native device
cases. Validation used ~/tritonchip-r40-ci on gx10-6646. Pipeline defaults
to zero; explicit image --local-pipeline requires an HBM burst bridge.
See localpipe_*.log. This adds local request latency; no speedup, timing
or bandwidth acceptance is claimed. New 100MHz implementation pending.

2026-10-13 RAM cache checkpoint: the local-pipeline image failed legal
routing (2270 unrouted signals, 2225 overlaps in line-data flops); no
image was programmed. localpipe_link_failure.json records that attempt.

An optional synchronous dual-port line RAM keeps held replies and their
tags stable, arbitrates cache-hit writes against reads, and uses byte
enables for subword updates. Defaults remain zero. The same 16-line
geometry maps to 2 RAMB36E2, 8 LUT2, no data flip-flops in Yosys XCU.
Bridge 70/70, semantic mutations 16/16, adverse native cases 4/4 and
FPGA wrapper/protocol 8/8 pass. See ram_cache_*.log. Cache-hit latency
increases by one cycle; no speedup or board timing is claimed.
