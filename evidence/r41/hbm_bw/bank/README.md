# Replicated stream bank and command binding

## Implemented behavior

`tnca_stream_bank.v` connects up to 32 real stream lanes to a flat 64-bit register interface. The image candidate uses 32 lanes, one per PC. Each outgoing AXI address has a fixed physical PC tag and a local 28-bit offset, so no lane can issue into another PC. All lanes retain the compact burst engine's operation semantics and credit bounds.

The bank validates the complete enabled descriptor set before asserting any lane command. It refuses unsupported modes, invalid source/destination alignment or range, overlapping copy ranges and any aperture other than base zero/full 8 GiB. A bad descriptor in a late lane prevents earlier lanes from starting. Disabled lanes are excluded from validation and aggregate counters.

Configuration writes are accepted only while idle and fault-free. A fresh monotonically increasing request sequence is required for each start; a stale doorbell cannot replay work. Commands remain pending independently until each lane accepts. Successful completion waits for every enabled lane to finish, sequentially sums actual lane results and read/write line counters, acknowledges every lane and then publishes the completed sequence. The measured cycle window includes dispatch and aggregation. Aggregating one lane per cycle avoids a large routed reduction tree.

Any lane fault blocks new lane admission and all later configuration/start writes. Faulted work never receives a successful completion. Recovery requires a coordinated lane/fabric reset. The runtime must still serialize this bank against the core CU and perform cache ownership handoff; this component does not itself make concurrent core/stream access coherent.

## Register contract

All offsets below address 64-bit registers. The future AXI-Lite wrapper must preserve these byte offsets.

| Offset | Meaning |
|---|---|
| 0x000 | Read-only identity `0x52343153424e4b01` |
| 0x008 | Capability: lane count [63:48], data width [47:32], burst beats [31:16], credit lines [15:0] |
| 0x010 | Requested sequence |
| 0x018 | Write bit 0 to start a fresh sequence |
| 0x020 | Status flags: busy bit 0, completed-sequence-present bit 1, fault bit 2 |
| 0x028 / 0x030 / 0x038 | Completed sequence / completion status / execution cycles |
| 0x040 | Mode [33:32], repeated bit pattern [31:0] |
| 0x048 / 0x050 | Enabled-lane mask / indirect lane selector |
| 0x058 / 0x060 / 0x068 | Selected lane source offset / destination offset / 64-byte line count |
| 0x070 / 0x078 / 0x080 | Aggregate uint32 sum / actual read lines / actual write lines |
| 0x088 / 0x090 | Declared backing aperture size / base |
| 0x098 / 0x0a0 | Selected lane result sum / fault-lane mask |
| 0x0a8 / 0x0b0 / 0x0b8 | Selected lane cycles / actual read lines / actual write lines |

Modes are 0 copy, 1 repeated-pattern write and 2 uint32 modulo-sum read-reduce. Completion status zero is success; three is command refusal. Fault must be checked separately from a previously completed sequence. Per-lane fields describe that lane's last command; global counters include only the current enabled lanes. Empty masks perform no memory transfers and are not valid bandwidth samples.

Descriptor and selector writes with unsupported high bits are ignored. Software must verify configuration readback before starting; it must not infer acceptance from a posted write alone. Busy writes, including sequence changes, are ignored. Completion sequence and counters remain stable until the next accepted start.

## Functional evidence and running screens

`tools/r41/hbm_bank_tests.py` instantiates real stream engines with independent AXI burst slaves. The two-, eight- and 32-lane configurations pass fill, copy and read-reduce data checks, per-PC addresses, line counters, inactive lanes, empty masks, all-lane rejection of a late invalid descriptor, ignored busy writes, stale doorbells, bank fault blocking and coordinated-reset recovery.

The initial 32-lane Icarus run exceeded the 120-second wall limit. Its timeout is retained in `functional_initial_timeout.log`. A diagnostic run showed advancing simulation clocks and successful fill completion, rather than a stopped simulation. The rerun uses unbuffered output and a 600-second bound. The complete rerun passes all three lane counts and all six command/recovery checks per configuration. `functional.log` ends with `HBM STREAM BANK FUNCTIONAL PASS 2/8/32 real lanes`. No RTL change was needed to pass the larger wall-time bound.

A free 32-lane routed OOC screen is live on the FPGA host at `~/r41_hbmbw_bank_v1/`, launcher PID 2612852. It is constrained to SLR0 at 10 ns, with 0.2 ns uncertainty and 1 ns I/O delay assumptions. It runs through the shared Vivado slot guard with a 20 GiB available-memory threshold. It does not access the card and does not consume the full-run budget. Its source and launch recipe are retained here; routed results remain pending.

No new image exists yet. The next work is the named AXI-master/AXI-Lite board wrapper and packaging, runtime command driver with cache ownership and native-copy dispatch, and combined tile/platform area and crossing screens.
