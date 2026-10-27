# Compact ordered burst stream lane

## Implemented component and functional evidence

`rtl/fpga/tnca_stream_axi.v` implements a per-PC stream lane with three explicitly defined operations: byte-exact copy of complete 64-byte lines; writes of a repeated 32-bit bit pattern; and reduction of unsigned 32-bit words modulo 2^32. The reduction is integer arithmetic and does not replace any floating-point instruction. Existing instruction implementations are unchanged. Partial lines, type conversion, arbitrary striding and overlapping copies require the existing execution path.

The lane issues up to eight ordered, single-ID AXI bursts, with at most 16 beats per burst. Each beat is 512 bits. Bursts stop at 4 KiB boundaries and at the command's end. Read admission reserves at most 128 lines; a copy releases credit only as data leaves the output FIFO. Read responses and write responses have independent bookkeeping, and successful completion waits for write acknowledgements and the integer reduction pipeline to drain.

`tnca_stream_fifo.v` uses a synchronous memory read, a registered output and an empty-queue bypass. It has no data-memory reset, allowing block RAM inference. The stream lane does not use the generic adapter's per-line address CAM, full-line register array or response-tag multiplexing. Copy disallows overlap, and AXI ID zero preserves burst order.

`python3 tools/r41/hbm_stream_tests.py` passes 18 configurations. The raw `functional.log` covers continuous and independently stalled channels, addresses above 4 GiB and in PC31, single-line and multi-burst copies, 4 KiB splits, pattern writes, integer wraparound reduction, empty commands, invalid lengths/ranges/modes/base alignment and misaligned source refusal. Command inputs are poisoned after acceptance to check snapshot isolation. Completion must hold until acknowledged. Tests reach the 128-line credit bound and exercise stored FIFO data under backpressure.

Seven fault cases run in each channel-stall mode: read error, write error, malformed RLAST, wrong RID, wrong BID, unsolicited B and unsolicited R. Each must latch fault, refuse new commands, avoid a successful completion and recover only after a coordinated engine/slave reset. The following copy must then pass. These checks are functional RTL evidence, not board bandwidth evidence.

## First synthesis result

The free FPGA-host OOC screen synthesized one lane at a requested 10 ns clock:

| Component | LUTs | Registers | BRAM tiles |
|---|---:|---:|---:|
| Generic pipelined 32-slot adapter | 17,514 | 21,746 | 0 |
| Ordered burst stream lane | 1,560 | 1,651 | 7.5 |
| Arithmetic projection: 24 burst lanes | 37,440 | 39,624 | 180 |
| Arithmetic projection: 32 burst lanes | 49,920 | 52,832 | 240 |

The compact lane has a deeper data window and an explicitly narrower access contract. The table is not a replacement of arbitrary legacy memory accesses. Replication columns exclude command/control logic, platform interconnect, width converters, ownership logic, physical optimization and placement overhead. The raw synthesis reports and exact screened RTL are in `screen_v1/`.

The routed screen is constrained to SLR1 with a 10 ns clock, 0.2 ns uncertainty and 1 ns input/output delays. It completed successfully on the FPGA host under `~/r41_hbmbw_stream_v1/`: setup WNS +5.596 ns, hold WHS +0.026 ns, zero TNS/THS and exit status zero. These are isolated-lane results, not full-image timing. It uses the shared Vivado slot guard and does not access the card or consume a full implementation run.

## Next image integration

The candidate for integration is 32 compact lanes at the existing 100 MHz kernel clock, preserving the three-core instruction path as the general fallback. The algebraic one-direction interface ceiling is 204.8 GB/s, so 100 GB/s requires 48.8% utilization. The existing controller experiment supports independent-PC service capacity, but does not establish a complete-chip rate or real copy throughput.

A small contiguous BO range currently occupies only one or a few physical PCs. Therefore real tensor buffers need an explicit logical-to-physical layout, rather than more master names. A candidate layout interleaves 4 KiB pages over 32 PCs: logical page bits [16:12] select PC, logical bits [32:17] become the high part of the local 28-bit PC offset, and bits [11:0] stay unchanged. Core outlet addressing and host DMA must implement the identical permutation, advertised by a new image capability. The original image path must remain unchanged when that capability is absent. Full capacity needs a verified 8 GiB allocation spanning the PCs; partial/offset BO allocations must refuse the new layout unless their mapping is explicitly supported.

For the initial accelerated copy path, source and destination offsets must have the same page-to-PC phase, with nonoverlapping ranges. Large buffers can use 128 KiB alignment. Other copies, partial-line tails, conversions and strided operations retain existing semantics through the core path. The accelerator must be invoked by the actual native storage/tensor copy path, rather than being only a benchmark generator.

A separate stream CU in the same xclbin is an integration option: it keeps the existing three-master core wrapper and its instruction ABI intact, while exposing the stream lane masters and its own control interface. Runtime operations must share the session serialization lock, drain the core, perform logical-range cache writeback/invalidation, run the stream command, verify completion and fault state, and only then expose data to subsequent core work. Integer read-reduce and pattern-write operations need explicit product APIs and correctness checks; they must not silently substitute for floating-point reductions or casts.

## Physical risks to resolve before a full run

The 32-lane area projection plus the existing 927,245 routed LUTs is below total LUT capacity, but this alone does not prove fit. SLR0's CLB packing is 96.74%, and the existing SLR0-to-SLR1 SLL use is 14,263 of 23,040. Placing all wide stream paths in other SLRs would add too many wires if done naively: 32 pairs of 512-bit data buses alone contain 32,768 signals. Width-converter placement, command distribution and existing crossing use must be inspected in the actual platform topology.

Keeping stream lanes near HBM in SLR0 avoids those wide crossings, but increases its packing/congestion. Distributing lanes requires a demonstrated crossing budget. The next free screens must include replicated lanes and the existing tile/SLR occupancy; isolated lane timing cannot qualify the image. No full run should be allocated until layout semantics, runtime integration, area, crossing count and routed screens have concrete evidence.
