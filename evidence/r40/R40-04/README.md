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

2026-10-13 common-clock software checkpoint: opt-in `device_clock: true`
captures each program's sequence-checked MMIO start/completion endpoints.
Session and asynchronous launch logs retain one clock/session/epoch domain;
ArchBench timed samples add device-graph-makespan only with complete
endpoints. Existing serial-grid cycles keep their meaning. Counter wrap,
out-of-order completion, missing programs, reset/domain mismatch, old image
unavailability and timing-field separation pass unit 16/16.

The standard native Torch mm -> sum graph runs through fpga-u50-sim,
one physical core and two dependent kernels: single/batched both pass,
zero steady tensor transfers, active cycles 4550 vs graph spans 5646/5626.
This is functional RTL evidence with submission gaps, not a performance
comparison or protocol-2 measurement. Native initialization also verifies
word padding outside a whole Storage and dense-transpose writes without
reading uninitialized RAM; gapped/partial views preserve neighboring bytes.
Host copy units 4/4, native copy 15/15, runtime 7/7, backend 3/3 pass.
See device_clock_* and host_* logs. The current ac580b0 RAM image build
does not include these clock registers; no board clock evidence is claimed.

2026-10-13 first timing-closed HBM image: source ac580b0 (the optional
local request pipeline and synchronous RAM cache), single physical core,
SLR1, HBM[0:31], 100 MHz kernel/device, 450 MHz HBM. Post-route setup WNS
+0.016 ns / TNS 0; hold WHS +0.006 ns / THS 0; pulse-width violations 0,
no unconstrained internal endpoints. v++ retained both requested clocks.
The xclbin SHA256 is 62751b3ca79044655bcb617a805f7ca2d2360efb752bcf63280fb59f25417413.

The complete routed CDC report has 58 critical findings, all internal to
vendor platform PCIe/ERT/clock/HBM IP; no reported crossing involves the
TNCA kernel. This is an explicit scope review, not a claim that the platform
report has zero critical findings. The raw report, all critical endpoints
and source hierarchy classification are preserved in ram_image/. Full
reports are gzip-compressed; reports.json retains their uncompressed hashes.

The card was idle, had no open users and measured FPGA 32 C / HBM 27 C.
program.sh --check and --go passed; the loaded UUID is
e576c850-23ec-61c0-f728-74a1c19676ea. Standard native Torch window and address
sampling pass 2/2 on software 576b310: an 8 GiB MBG[0:31] BO, 32 distinct
pseudo-channels, first sample 0x21000, final sample 0x1f0021000, including
addresses above 4 GiB, correct host DMA and in-place device increment.
The separate 64 MiB capacity/96 MiB streaming audit is running. This does
not yet close R40-04 or establish its bandwidth/performance thresholds.
The image predates the new common-clock registers.
