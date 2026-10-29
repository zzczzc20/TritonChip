# G4: a general full-link data path for memory-bound operators — design and bound

Status: workstream G4 of [`BANDWIDTH_PLAN.md`](BANDWIDTH_PLAN.md), 2026-10-28: phase 1 (design, §1–§7) and phase 2 (the one-lane path, §8). The run-2 candidate is prepared and not linked. **[`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md) is binding and takes precedence over this document.**

Inputs: [`LINK_BUDGET.md`](LINK_BUDGET.md) (G3), [`RESOURCES.md`](RESOURCES.md) (G2), [`AUDIT.md`](AUDIT.md) (G1). Branch `r41-g4`, made from `origin/r41-hbm-bw` with `origin/r41` merged in (the stream engine stays removed).

Domains: every number is labelled **board** (run-1 image, 100 MHz), **rtl** (RTL simulation), **model** (SystemC/TLM, uncalibrated), **screen** (out-of-context synthesis or place and route), or **analytic** (arithmetic on recorded numbers or on compiled code). An **estimate** is an engineering estimate with no run behind it, and is marked as such. Nothing here is a board measurement of a new design.

**Decision (2026-10-28).** Author decision (source: the author, in the lead's chat, 2026-10-28, answering the lead's question about the G4 phase-1 result): **option (c), build the one-lane 64 B path (about 19 GB/s analytic).** Phase 2 (§8) builds that path, on G2's reduced profile `core-vrf-sync-v5-x8-g2`. The 32 GB/s target is recorded as unreachable, with the evidence in §1 and §4–§5. The registered suite is unchanged, and every case is reported.

## 1. Result

**The target, as defined in BANDWIDTH_PLAN.md §1, cannot be reached with the current ISA and core, whatever the memory path does.** The plan claims 32 GB/s on the board, which is 106.7 B/cycle per core, for copy/clone, the element-wise binary operators and the unary activations of the registered suite. Below that the limit is the core's instruction issue, not memory:

- The core issues one instruction per cycle. A vector instruction works on 16 lanes of 32 bits.
- Vector floating point is binary32 only, and there are no packed half-precision instructions.
- The SFU has one slice by default.

The issue ceilings come from G3's compile census of the registered suite (`g4/issue_bound.json`, `tools/r41/g4_issue_bound.py`; analytic, on the compiled code). They are computed for an ideal lowering in which loads, stores, address arithmetic, constants and loop overhead cost nothing, and only the instructions that depend on loaded data remain:

| Class (large points) | Data-dependent instructions per 16 elements | Ideal issue ceiling, B/cycle/core (16 SFU slices) | Reaches 106.7? |
|---|---:|---:|---|
| copy, clone (all dtypes, all layouts) | 0 | unbounded by issue | yes |
| add/sub/mul fp32 (contig, strided / broadcast) | 1 | 192 / 128 | yes, before loop overhead (§3.2) |
| add/sub/mul fp16, bf16 | 4 (two widen, op, narrow) | 24 / 16 | **no** |
| div fp32 / fp16, bf16 | 7 / 10, one SFU op | 27.4 / 9.6 | **no** |
| relu fp32 / fp16, bf16 | 1 / 3 | 128 / 21.3 | fp32 only, before loop overhead (§3.2) |
| exp fp32 / fp16, bf16 | 7 / 9, one SFU op | 18.3 / 7.1 | **no** |
| silu, sigmoid fp32 / fp16, bf16 | 16 / 18, two SFU ops | 8.0 / 3.6 | **no** |
| gelu (erf) fp32 / fp16, bf16 | 53 / 55 | 2.4 / 1.2 | **no** |
| casts (fp32↔fp16/bf16, fp16↔bf16) | 1–2 | 32–96 | **no** |
| sum, mean, max, layernorm, rmsnorm, softmax | ≥ 1 per 16 *input* elements | ≤ 64 (fp32), ≤ 32 (fp16, bf16) | **no** |

29 of the 171 large registered points have an ideal issue ceiling at or above 106.7 B/cycle/core: copy and clone in every dtype and layout (18), fp32 add/sub/mul (9) and contiguous and strided fp32 relu (2). Of the target's own classes, all 18 copy/clone points can reach it in principle. Only 9 of the 36 binary points can, and 2 of the 45 activation points. With realistic tile overhead (§3.2), fp32 relu falls below the target (89–100 B/cycle/core), and fp32 add/sub/mul land at 103–112, so they reach it only with the lowest overhead.

Lifting the bound for the rest would take changes this workstream may not make: packed fp16/bf16 vector arithmetic or a wider vector unit (ISA and numerics, plus about 2,400 LUTs per added binary32 FMA lane), dual issue, or 16 SFU slices per core. All of these are the author's decisions (§7).

**For the classes the issue bound allows, the memory path is also out of reach within area and timing.** §4 sizes every stage for 106.7 B/cycle/core:
- Each stage needs two 64-byte lines per cycle per core, in the NoC, the home slice and the tensor-engine write port. The reason is that a binary operator reads 1.11 lines/cycle.
- About 160 lines must be in flight per core.
- Nine outlets are needed, with 32 lines outstanding each.

Area and timing for this:
- **Area.** The existing one-lane wide path alone measured +26,315 routed LUTs per tile (screen: tile 0 at 273,894 LUTs against 247,579). The second lane, the line-wide multi-command transfer engine and the outlets add an estimated 35–50k more per tile. Against that, G2's measured cuts free about 2.5k LUTs per tile, plus 16.7k in SLR0 from the outlets. The approved cuts (contexts 4→2, fewer barriers, narrower addresses) free an estimated 6–10k per tile more; G2 is still measuring them. SLR0 is at 96.74 % CLB packing (routed image).
- **Timing.** The one-lane wide tile already screens at only +0.508 ns in isolation. Run 1's screen-to-image calibration for RAM-capture paths in a ≥ 96 % SLR is −0.9 to −1.0 ns.

The model confirms the direction for the code the compiler produces today (§5; model domain). With the G4 memory parameters, the registered small points reach at most 3.1 B/cycle/core. With an idealised memory they reach at most 5.0. The run-1 geometry gives 0.9. The compiled code (one 16-lane vector per program) limits them long before memory does.

**Recommendation (phase 1, before the decision).** Per the brief, phase 2 (RTL) was not started. The decision went back to the lead and the author:
- (a) Restate the target for the classes that can reach it (copy and clone, and fp32 add/sub/mul with a thin margin), and report the issue bound for the rest.
- (b) Approve, or reject, an ISA or core change that lifts the issue bound (§7).
- (c) Accept a smaller, buildable memory path (§6). Its estimated chip ceiling is about 19 GB/s for copy and fp32 binary operators, analytic, and it is still worth building because every program benefits.

The compiler work in §3 (tensor-engine tiles with software pipelining) is general and needed under every option. It is the first thing to build, and it needs no new hardware to show its effect up to the 8 B/cycle core port.

## 2. Requirement per stage

32 GB/s at 100 MHz is 320 B/cycle for the chip, or **R = 106.7 B/cycle per core** of ArchBench demand (reads plus writes). Line = 64 B.

| Operator shape (contiguous) | Reads, B/cycle/core | Writes, B/cycle/core | Lines/cycle/core |
|---|---:|---:|---:|
| copy, clone, unary, same-size cast (1 : 1) | 53.3 | 53.3 | 0.83 read + 0.83 write |
| binary (2 : 1) | 71.1 | 35.6 | 1.11 read + 0.56 write |
| reduction (read only) | 106.7 | ≈ 0 | 1.67 read |

- A stride-2 view reads whole lines, so its link read traffic is twice its demand reads.
- A broadcast operand is a single row and adds almost no traffic.

Every stage below must therefore carry at least 1.11 read lines per cycle per core for binary operators, and 1.67 lines per cycle in total, with margin. One 64-byte line per cycle per direction is not enough: it caps a binary operator at 64 + 32 = 96 B/cycle/core.

## 3. Compiler and issue

### 3.1 What the compiler emits today
From G3's census and the compiler sources:
- Every element-wise kernel moves one 16-lane vector per program (`STAGE1_MAX_BLOCK = 16`, `python/tnca/inductor.py:72`). A wider `TNCA_MAX_BLOCK` is unrolled into separate 16-lane chunks by `register_blocks.lower` (`compiler/python/tnca_cc/register_blocks.py:204-334`), with no loop and no tensor engine.
- 1-D loads and stores always go to the LSU (`frontend.py:5782-5920`).
- `descriptors.load` uses the tensor transfer engine (TTE) only for multi-row boxes. These must be at most 1,024 B, and their consumers must be able to read a window (`descriptors.py:351-438`).
- The engine's use is synchronous: arm, `t.ld.async`, wait (`frontend.py:3926-3953`).
- The only software pipelining pass (`single_role_pipeline.py`) requires a DMA→MMA loop.

### 3.2 Proposed general lowering (no hardware change needed to start)
1. **Persistent tiled programs for 1-D element-wise kernels.** One program loops over a contiguous range of tiles: `x = pid; while x < ntiles: …; x += nprog`, as `rctx_spmd` already emits. This removes the per-program dispatch, argument loads and index vectors. The cost per program is about 38 cycles at ideal memory (model, `ideal` arm, fp32 add, 192 B per program at 5.0 B/cycle/core).
2. **Tensor-engine tiles.**
   - Each input is a rank-1 or rank-2 box of T registers, moved by `t.ld.async` into a window.
   - The output is computed in place into a window and leaves by `t.st`.
   - The conditions are generic: the dtype is TTE-transferable, the inner stride is the element size, rows are 8-byte multiples, and the base is 8-byte aligned. The legal tile is the largest T that fits two stages in the context's 64 registers.
3. **Software pipelining.** Load tile *i+1* while computing tile *i* and storing tile *i−1*: two barrier sets, with `mb.wait` only on the tile being consumed. This generalises `single_role_pipeline` from DMA→MMA to DMA→vector→DMA loops.

**The 16-bit types need one more step:**
- TTE windows hold packed bytes (32 fp16 per register), while vector arithmetic needs one value per 32-bit lane.
- `v.unpack.f16`/`v.pack.f16` are reserved and not implemented (ISA §8.4). The packed-window unpack today is a default-off knob (`TNCA_PACKED_WINDOW_UNPACK`).
- So fp16/bf16 element-wise kernels can use the TTE only through that unpack sequence, or by keeping the LSU's widening loads (`v.ld.h`).

**Tile overhead** (analytic):
- Per tile, the core issues T × k data instructions plus about c = 8–12 others: one `t.ld.async` per input, one `t.st`, one or two `mb.wait`, the coordinate update and the loop branch.
- The core's 64 logical registers per context bound T. Two stages of two fp32 inputs computed in place give T ≤ 14.
- **fp32 add:** 192·T/(T + c) = 103–112 B/cycle/core at T = 14 and c = 12–10. It reaches the target only at the low end of the overhead, with at most 5 % margin.
- **fp32 relu** (one input, T ≤ 28): 128·T/(T + c) = 89–100. This is below the target.
- **copy/clone** (k = 0): 128·T/c, so issue never binds.

### 3.3 Reductions and normalisations
A reduction needs at least one vector operation per 16 input elements. That limits it to 64 B/cycle/core at fp32 and 32 at fp16/bf16, with a widening convert. Normalisations need several. No reduction or normalisation can reach the target on this core.

## 4. Memory path sized for 106.7 B/cycle/core

Ceilings are per core and assume perfect behaviour of the other stages; margin is the ceiling divided by the requirement. Area is per tile unless stated otherwise. "Measured" area comes from the named screen; every other area figure is an estimate.

| Stage | Today (G3, rtl unless stated) | Design | Throughput math | Ceiling vs need | Area | Timing risk at 100 MHz |
|---|---|---|---|---|---|---|
| Issue | 3.7 B/cycle (fp16 add as compiled) | TTE tiles, §3.2 | 192·T/(T+c) for fp32 binary | 103–112 vs 106.7 (fp32 binary); see §1 for others | none (compiler) | none |
| Tensor transfer engine | 8 B beats, one command at a time, 16 tags (1.75 B/cycle analytic) | 64 B line beats, a queue of 4 commands, 160 line tags, two request issues per cycle | lines in flight ≥ 1.67 lines/cycle × 73-cycle round trip (rtl port round trip) = 122; 160 gives a margin of 1.3 | 2 lines/cycle vs 1.67 | estimate +8–12k LUTs (TTE today 14,494 LUTs, synthesis), tag table in LUTRAM | medium: 512-bit store-read and write-port muxes in the core |
| VRF window port | engine writer: 8 B per cycle; 1 register read per cycle | two engine write ports (writer 1 and a new writer 2, any bank); one store-read port | binary: 1.11 window writes/cycle + 0.56 ALU writes; four 1R1W banks | 2 engine writes/cycle vs 1.11 | estimate +2–3k LUTs (write muxes on four 512-bit banks) | medium: bank write mux into BRAM |
| Core LSU (non-TTE traffic) | hybrid LSU, 8 line slots, 1 line/cycle | keep the hybrid LSU (G1: valid) | not on the streaming path once §3.2 lands | – | measured +5.0k (hybrid 30,429 routed, against the pair's 25,435 synthesised) | low (+1.271 ns, screen) |
| Core NI and mesh | 64-bit payload, 1 flit/cycle per VN (8 B/cycle) | two memory lanes (lane = line address bit 6), each a 720-bit flit per cycle on two VNs | worst link (0→1 east): 2/3 × 1.67 = 1.11 lines/cycle per direction | 2 vs 1.11 per link and VN | measured for one lane: part of the +26.3k tile delta; second lane estimate +8–12k (NIs, routers) plus about 5.8k SLLs per SLR boundary per link | high: every mesh link crosses an SLR boundary at 720 bits × 2 lanes × 2 VNs |
| NUCA home slice | serial engine, 3–7 cycles per word; stores read their line first (1.5–3× read amplification) | two lanes per home, each a pipelined tag/data engine accepting one line per cycle; full-line write without fill; 48 MSHRs per lane with BRAM fill buffers; fast fill | each home serves a third of all cores' traffic, 1.67 lines/cycle; read misses 1.11 lines/cycle × (home→outlet→home ≈ 68 cycles: rtl outlet round trip 51 + home/link stages) ≈ 75 MSHRs per home | 2 lines/cycle vs 1.67; 96 MSHRs vs 75 | measured for one lane: wide home 41,279 routed LUTs against 30,647 narrow (synthesis); two lanes and 96 MSHRs estimate +25–35k LUTs and 8 RAMB36 per home | high: the one-lane slice screens +1.774 ns alone; two pipelined lanes add a tag/MSHR CAM per lane |
| Outlets | three 256-bit outlets, 8 lines each, issue every 2 cycles (28.6 B/cycle, rtl) | nine 256-bit outlets (three per home), pipelined issue, 32 lines outstanding each, LUTRAM line store | per home: read 71 B/cycle against 3 × 32 B/cycle; outstanding 0.37 lines/cycle × 51 cycles (rtl) ≈ 19, so 32 | 96 vs 71 B/cycle read per home | 32-slot generic outlet measured at 17,443 routed LUTs; with `AXI_LINE_LUTRAM` an estimate of 4–6k each (its same-line CAM remains), so about +20–30k in SLR0 against the 3 × 7.4k today | medium: 32 slots routed +1.658 ns alone (screen); SLR0 packing |
| HBM pseudo-channels | a 64 MiB tensor in 1–2 PCs (contiguous layout) | stripe 4 KiB pages over 32 PCs (`HBM_STRIPED`, layout `pc32-page4k-v1`, exists, default off); outlet chosen by PC | 32 GB/s / 6.2 GB/s per PC (model, Ramulator2, 64 in flight) ≥ 6 PCs busy | 32 PCs vs ≥ 6 | none in the kernel; platform HBM switch | high, unmeasured: with global addressing, every outlet reaches every PC through the HBM switch, whose lateral bandwidth between switch groups is limited |

### 4.1 Area balance per SLR

Per tile (SLR1 and SLR2, and tile 0 in SLR0):

| Item | LUTs | Basis |
|---|---:|---|
| One-lane wide path (hybrid LSU, 720-bit NI/mesh, wide home with chain, background reads, also 4 SFU slices) | **+26,315 (measured)** | screen: wide tile 0 in SLR1 at 273,894 LUTs (`kernel_wide/tile0_slr1/`) against the base tile-0 screen at 247,579 (RESOURCES.md §4.1) |
| Second NoC/home lane, MSHRs | +33–47k (estimate) | wide home and NI scaling above |
| Line-wide multi-command TTE | +8–12k (estimate) | §4 table |
| VRF second engine writer | +2–3k (estimate) | §4 table |
| `NUCA_SHARED_DATA_READ 2` | −2,478 (measured) | RESOURCES.md §4.1, tile screen |
| Barriers 32→16 (approved) | about −3.7k (estimate: half of `u_bar`'s 7,450 LUTs) | G2 measuring |
| Contexts 4→2 (approved) | about −1–3k LUTs (estimate: half of the SRF's 2,072 LUTRAM, plus context state) and −16 RAMB36 per core | G2 measuring |
| Narrower physical addresses (approved) | −2–5k (estimate) | G2 measuring |
| **Net per tile** | **about +60k to +80k** | |

In SLR0, the outlets also count:
- **Today:** three outlets at about 7.4k LUTs each, or about 1.4k each with `AXI_LINE_LUTRAM` (measured).
- **Design:** nine 32-slot outlets at an estimated 4–6k each.
- **Net:** about +20–30k LUTs against the LUTRAM baseline.

SLR0 is at 96.74 % CLB packing in the routed run-1 image, and SLR1 at 88.67 %. A net +60–80k LUTs per tile is 14–18 % of an SLR's 432k LUTs. **This does not fit in SLR0, and is unlikely to fit in SLR1**, without removing a capability. The author has rejected the two large candidates: sharing MACs and a smaller FMA.

### 4.2 Timing
The one-lane wide tile already passes its isolated SLR1 screen with only +0.508 ns of slack (`kernel_wide/tile0_slr1/`). Run 1's calibration from screen to image for RAM-capture families in a ≥ 96 % SLR is −0.9 to −1.0 ns (`stage2/run1/README.md`). So the one-lane path would probably fail at 100 MHz in SLR0. A second lane and a pipelined home engine with per-lane MSHR CAMs make this worse. The NoC's 2 × 2 × 720-bit links cross every SLR boundary, and so do the home-to-outlet links from SLR1 and SLR2 into SLR0. Each needs registered crossings (`NOC_LINK_STAGES`, `LINK_STAGES`), which lengthen the round trip that §4 sizes the in-flight tags against.

## 5. Model validation (model-prediction-uncalibrated)

Tool: `tools/r41/g4_model_suite.py` runs the registered suite through `tools/performance_harness/systemc/run_graphs.py`. The path is the standard one: `torch.compile`, Inductor, Triton, the TNCA compiler and a Session, on three cores, with the compiler's default settings. The model is `tnca-chip`, built from this branch, sha256 `a09d1e09…fdac5`, on GB10.

Three memory configurations, all at 100 MHz (`g4/model_small/*.cfg`):
- **`run1`:** the run-1 geometry, from `configs/r41-hbm8-device-service.cfg`. The SFU occupancy is 22/23, as corrected in `vector_schedule/README.md`.
- **`g4`:** the parameters of §4, as far as the model has keys for them:
  - 64-byte core beat and NoC payload, and two memory lanes;
  - every home occupancy set to 1 cycle per line;
  - 128 MSHRs and 256 targets per home, and 64 LSU slots;
  - 6 outlets with 64 outstanding and single-cycle issue;
  - 4 KiB striping over 32 PCs.
- **`ideal`:** `g4` with every memory latency at its minimum, 8 lanes and 32 outlets. It isolates what the compiled code can do on the core.

The model has no key for a full-line write without fill, for a line-wide tensor engine, or for the VRF window ports. Those parts of the design are not represented.

Small points (64 KiB per tensor), all 171, every one passing its numerical check under all three configurations. The full table is generated by `python3 tools/r41/g4_report.py`; contiguous rows:

| Family | dtype | Layout | run1 B/cycle/core | g4 B/cycle/core | ideal B/cycle/core |
|---|---|---|---:|---:|---:|
| copy | bf16 | contig | 0.7 | 1.2 | 2.0 |
| copy | fp16 | contig | 0.7 | 1.2 | 2.0 |
| copy | fp32 | contig | 0.9 | 2.4 | 4.0 |
| clone | bf16 | contig | 0.7 | 1.2 | 2.0 |
| clone | fp16 | contig | 0.7 | 1.2 | 2.0 |
| clone | fp32 | contig | 0.9 | 2.4 | 4.0 |
| add | bf16 | contig | 0.7 | 1.4 | 2.3 |
| add | fp16 | contig | 0.7 | 1.4 | 2.3 |
| add | fp32 | contig | 0.9 | 3.1 | 5.0 |
| sub | bf16 | contig | 0.7 | 1.4 | 2.3 |
| sub | fp16 | contig | 0.7 | 1.4 | 2.3 |
| sub | fp32 | contig | 0.9 | 3.1 | 5.0 |
| mul | bf16 | contig | 0.7 | 1.4 | 2.3 |
| mul | fp16 | contig | 0.7 | 1.4 | 2.3 |
| mul | fp32 | contig | 0.9 | 3.1 | 5.0 |
| div | bf16 | contig | 0.6 | 0.9 | 1.4 |
| div | fp16 | contig | 0.6 | 0.9 | 1.4 |
| div | fp32 | contig | 0.9 | 2.3 | 3.1 |
| relu | bf16 | contig | 0.7 | 1.1 | 1.7 |
| relu | fp16 | contig | 0.7 | 1.1 | 1.7 |
| relu | fp32 | contig | 0.9 | 2.3 | 3.8 |
| silu | bf16 | contig | 0.4 | 0.5 | 0.6 |
| silu | fp16 | contig | 0.4 | 0.5 | 0.6 |
| silu | fp32 | contig | 0.7 | 1.0 | 1.3 |
| gelu | bf16 | contig | 0.2 | 0.2 | 0.2 |
| gelu | fp16 | contig | 0.2 | 0.2 | 0.2 |
| gelu | fp32 | contig | 0.4 | 0.4 | 0.5 |
| exp | bf16 | contig | 0.6 | 0.7 | 1.0 |
| exp | fp16 | contig | 0.6 | 0.7 | 1.0 |
| exp | fp32 | contig | 0.8 | 1.6 | 2.1 |
| sigmoid | bf16 | contig | 0.4 | 0.5 | 0.6 |
| sigmoid | fp16 | contig | 0.4 | 0.5 | 0.6 |
| sigmoid | fp32 | contig | 0.7 | 1.0 | 1.3 |
| cast | bf16→fp16 | contig | 0.7 | 1.1 | 1.8 |
| cast | bf16→fp32 | contig | 0.8 | 1.7 | 2.8 |
| cast | fp16→bf16 | contig | 0.7 | 1.1 | 1.8 |
| cast | fp16→fp32 | contig | 0.8 | 1.7 | 2.8 |
| cast | fp32→bf16 | contig | 0.9 | 1.8 | 2.8 |
| cast | fp32→fp16 | contig | 0.9 | 1.8 | 2.8 |
| sum | bf16 | contig | 0.8 | 0.9 | 1.1 |
| sum | fp16 | contig | 0.8 | 0.9 | 1.1 |
| sum | fp32 | contig | 0.9 | 1.3 | 1.7 |
| mean | bf16 | contig | 0.8 | 0.9 | 1.1 |
| mean | fp16 | contig | 0.8 | 0.9 | 1.1 |
| mean | fp32 | contig | 0.9 | 1.3 | 1.7 |
| max | bf16 | contig | 0.8 | 0.9 | 1.1 |
| max | fp16 | contig | 0.8 | 0.9 | 1.1 |
| max | fp32 | contig | 0.8 | 1.3 | 1.7 |
| layernorm | bf16 | contig | 0.3 | 0.4 | 0.4 |
| layernorm | fp16 | contig | 0.3 | 0.4 | 0.4 |
| layernorm | fp32 | contig | 0.4 | 0.6 | 0.7 |
| rmsnorm | bf16 | contig | 0.4 | 0.5 | 0.5 |
| rmsnorm | fp16 | contig | 0.4 | 0.5 | 0.5 |
| rmsnorm | fp32 | contig | 0.5 | 0.8 | 0.8 |
| softmax | bf16 | contig | 0.2 | 0.3 | 0.3 |
| softmax | fp16 | contig | 0.2 | 0.3 | 0.3 |
| softmax | fp32 | contig | 0.4 | 0.5 | 0.6 |
Status counts: run1: pass 171; g4: pass 171; ideal: pass 171
Highest B/cycle/core of any point: run1 0.9; g4 3.1; ideal 5.0
<!-- /G4_MODEL_SMALL -->

What this shows:
- The compiled code is the limit. The highest prediction for any point is 0.9 (`run1`), 3.1 (`g4`) and 5.0 B/cycle/core (`ideal`).
- With an idealised memory, a fp32 add program still costs about 38 cycles for 16 elements: dispatch, argument loads, index vector and dependency latency.
- The memory-side changes alone cannot reach the target.
- The `run1` arm predicts fp16 copy at 0.7 B/cycle/core. G3 measured 0.449 (rtl) and the board ran fp16 clone at 0.543, so the model is within the same order and uncalibrated.

**Model, medium points, model_medium (model-prediction-uncalibrated)**, contiguous rows:

| Family | dtype | Layout | run1 B/cycle/core | g4 B/cycle/core | ideal B/cycle/core |
|---|---|---|---:|---:|---:|
| copy | bf16 | contig | 0.7 | 1.2 | 2.0 |
| copy | fp16 | contig | 0.7 | 1.2 | 2.0 |
| copy | fp32 | contig | 0.9 | 2.4 | 4.0 |
| clone | bf16 | contig | 0.7 | 1.2 | 2.0 |
| clone | fp16 | contig | 0.7 | 1.2 | 2.0 |
| clone | fp32 | contig | 0.9 | 2.4 | 4.0 |
| add | bf16 | contig | 0.7 | 1.4 | 2.3 |
| add | fp16 | contig | 0.7 | 1.4 | 2.3 |
| add | fp32 | contig | 1.0 | 3.1 | 5.0 |
| sub | bf16 | contig | 0.7 | 1.4 | 2.3 |
| sub | fp16 | contig | 0.7 | 1.4 | 2.3 |
| sub | fp32 | contig | 1.0 | 3.1 | 5.0 |
| mul | bf16 | contig | 0.7 | 1.4 | 2.3 |
| mul | fp16 | contig | 0.7 | 1.4 | 2.3 |
| mul | fp32 | contig | 1.0 | 3.1 | 5.0 |
| div | bf16 | contig | 0.6 | 1.0 | 1.4 |
| div | fp16 | contig | 0.6 | 1.0 | 1.4 |
| div | fp32 | contig | 0.9 | 2.3 | 3.2 |
| relu | bf16 | contig | 0.7 | 1.1 | 1.7 |
| relu | fp16 | contig | 0.7 | 1.1 | 1.7 |
| relu | fp32 | contig | 0.9 | 2.3 | 3.8 |
| silu | bf16 | contig | 0.4 | 0.5 | 0.6 |
| silu | fp16 | contig | 0.4 | 0.5 | 0.6 |
| silu | fp32 | contig | 0.7 | 1.0 | 1.3 |
| gelu | bf16 | contig | 0.2 | 0.2 | 0.2 |
| gelu | fp16 | contig | 0.2 | 0.2 | 0.2 |
| gelu | fp32 | contig | 0.4 | 0.4 | 0.5 |
| exp | bf16 | contig | 0.6 | 0.7 | 1.0 |
| exp | fp16 | contig | 0.6 | 0.7 | 1.0 |
| exp | fp32 | contig | 0.8 | 1.6 | 2.1 |
| sigmoid | bf16 | contig | 0.4 | 0.5 | 0.6 |
| sigmoid | fp16 | contig | 0.4 | 0.5 | 0.6 |
| sigmoid | fp32 | contig | 0.7 | 1.0 | 1.3 |
| cast | bf16→fp16 | contig | 0.7 | 1.1 | 1.8 |
| cast | bf16→fp32 | contig | 0.8 | 1.7 | 2.8 |
| cast | fp16→bf16 | contig | 0.7 | 1.1 | 1.8 |
| cast | fp16→fp32 | contig | 0.8 | 1.7 | 2.8 |
| cast | fp32→bf16 | contig | 0.9 | 1.8 | 2.8 |
| cast | fp32→fp16 | contig | 0.9 | 1.8 | 2.8 |
| sum | bf16 | contig | 0.9 | 1.0 | 1.2 |
| sum | fp16 | contig | 0.9 | 1.0 | 1.2 |
| sum | fp32 | contig | 1.0 | 1.9 | 2.5 |
| mean | bf16 | contig | 0.9 | 1.0 | 1.2 |
| mean | fp16 | contig | 0.9 | 1.0 | 1.2 |
| mean | fp32 | contig | 1.0 | 1.9 | 2.5 |
| max | bf16 | contig | 0.9 | 1.0 | 1.2 |
| max | fp16 | contig | 0.9 | 1.0 | 1.2 |
| max | fp32 | contig | 1.0 | 1.9 | 2.5 |
| layernorm | bf16 | contig | 0.4 | 0.4 | 0.5 |
| layernorm | fp16 | contig | 0.4 | 0.4 | 0.5 |
| layernorm | fp32 | contig | 0.4 | 0.9 | 1.0 |
| rmsnorm | bf16 | contig | 0.5 | 0.5 | 0.6 |
| rmsnorm | fp16 | contig | 0.5 | 0.5 | 0.6 |
| rmsnorm | fp32 | contig | 0.6 | 1.1 | 1.2 |
| softmax | bf16 | contig | 0.2 | 0.3 | 0.3 |
| softmax | fp16 | contig | 0.2 | 0.3 | 0.3 |
| softmax | fp32 | contig | 0.4 | 0.6 | 0.7 |
Status counts: run1: pass 171; g4: pass 171; ideal: pass 171
Highest B/cycle/core of any point: run1 1.0; g4 3.1; ideal 5.0
<!-- /G4_MODEL_MEDIUM -->

**Model, small points, model_small_block128 (model-prediction-uncalibrated, NON-DEFAULT TNCA_MAX_BLOCK=128: does not count for any target)**, contiguous rows:

| Family | dtype | Layout | g4 B/cycle/core | ideal B/cycle/core |
|---|---|---|---:|---:|
| copy | bf16 | contig | 3.2 | 4.1 |
| copy | fp16 | contig | 3.2 | 4.1 |
| copy | fp32 | contig | 6.0 | 8.0 |
| clone | bf16 | contig | 3.2 | 4.1 |
| clone | fp16 | contig | 3.2 | 4.1 |
| clone | fp32 | contig | 6.1 | 8.0 |
| add | bf16 | contig | 2.8 | 4.0 |
| add | fp16 | contig | 2.8 | 4.0 |
| add | fp32 | contig | 6.0 | 9.0 |
| sub | bf16 | contig | 2.8 | 4.0 |
| sub | fp16 | contig | 2.8 | 4.0 |
| sub | fp32 | contig | 6.0 | 9.0 |
| mul | bf16 | contig | 2.8 | 4.0 |
| mul | fp16 | contig | 2.8 | 4.0 |
| mul | fp32 | contig | 6.0 | 9.0 |
| div | bf16 | contig | 2.1 | 2.6 |
| div | fp16 | contig | 2.1 | 2.6 |
| div | fp32 | contig | 4.7 | 5.5 |
| relu | bf16 | contig | 2.6 | 3.2 |
| relu | fp16 | contig | 2.6 | 3.2 |
| relu | fp32 | contig | 5.0 | 6.9 |
| silu | bf16 | contig | 1.0 | 1.1 |
| silu | fp16 | contig | 1.0 | 1.1 |
| silu | fp32 | contig | 2.1 | 2.3 |
| gelu | bf16 | contig | 0.3 | 0.4 |
| gelu | fp16 | contig | 0.3 | 0.4 |
| gelu | fp32 | contig | 0.7 | 0.7 |
| exp | bf16 | contig | 1.7 | 1.9 |
| exp | fp16 | contig | 1.7 | 1.9 |
| exp | fp32 | contig | 3.6 | 4.1 |
| sigmoid | bf16 | contig | 1.0 | 1.1 |
| sigmoid | fp16 | contig | 1.0 | 1.1 |
| sigmoid | fp32 | contig | 2.2 | 2.4 |
| cast | bf16→fp16 | contig | 2.7 | 3.6 |
| cast | bf16→fp32 | contig | 4.1 | 5.3 |
| cast | fp16→bf16 | contig | 2.7 | 3.6 |
| cast | fp16→fp32 | contig | 4.1 | 5.3 |
| cast | fp32→bf16 | contig | 3.8 | 5.2 |
| cast | fp32→fp16 | contig | 3.8 | 5.2 |
| sum | bf16 | contig | 1.2 | 1.3 |
| sum | fp16 | contig | 1.2 | 1.3 |
| sum | fp32 | contig | 2.0 | 2.0 |
| mean | bf16 | contig | 1.2 | 1.3 |
| mean | fp16 | contig | 1.2 | 1.3 |
| mean | fp32 | contig | 1.9 | 2.0 |
| max | bf16 | contig | 1.1 | 1.2 |
| max | fp16 | contig | 1.1 | 1.2 |
| max | fp32 | contig | 1.8 | 1.9 |
| layernorm | bf16 | contig | 0.6 | 0.7 |
| layernorm | fp16 | contig | 0.6 | 0.7 |
| layernorm | fp32 | contig | 1.1 | 1.1 |
| rmsnorm | bf16 | contig | 1.0 | 1.0 |
| rmsnorm | fp16 | contig | 1.0 | 1.0 |
| rmsnorm | fp32 | contig | 1.6 | 1.7 |
| softmax | bf16 | contig | 0.3 | 0.4 |
| softmax | fp16 | contig | 0.3 | 0.4 |
| softmax | fp32 | contig | 0.7 | 0.8 |
Status counts: g4: pass 171; ideal: pass 171
Highest B/cycle/core of any point: g4 6.1; ideal 9.0
<!-- /G4_MODEL_BLOCK128 -->

## 6. A smaller memory path that fits (option c)

If the target is restated, a one-line-per-cycle path is the largest general improvement that has a plausible area and timing story. It consists of:
- the hybrid LSU;
- one 720-bit NoC lane;
- the wide home with a pipelined single lane, full-line write without fill and 32 MSHRs;
- a line-wide TTE with 64 tags;
- six outlets with 16–32 lines each;
- striping.

Its ceiling is set by the single home lane. Each home accepts one line per cycle and serves a third of the chip's traffic, which is 64 B/cycle per core of reads and writes together. For copy and for fp32 add alike, that is 19.2 GB/s for the chip (analytic).

Both depend on §3.2 being in place. Its added area is the measured +26.3k per tile of the one-lane wide path, plus an estimated 10–15k. Its timing is the +0.508 ns screen above, which still needs work in SLR0. It is not proposed here as a run-2 candidate. It is the honest upper end of what this workstream could build.

## 7. What would lift the issue bound (author's decisions)

| Change | Effect on the bound (analytic) | Cost / rule |
|---|---|---|
| Packed fp16/bf16 vector arithmetic (two values per 32-bit lane) | fp16 add 24 → at most 192 B/cycle/core (one instruction per 32 elements, if no converts remain) | ISA extension and a numerics change in the FMA datapath; needs the author's approval (discipline §2.2.5, §2.5.3) |
| Implement the reserved `v.unpack.f16`/`v.pack.f16` | lets the TTE feed fp16 kernels; issue does not improve | ISA: reserved encodings; author |
| Dual issue (vector + TTE/scalar) | removes the loop overhead c from §3.2: fp32 relu up to 128 | a core redesign |
| 16 SFU slices per core | exp fp32 8 → 18 B/cycle/core; silu 4 → 8 | the 4-slice SFU routes at 8,748 LUTs (screen); the result stays far below the target |
| 32 vector lanes | doubles every element-wise bound | 16 more FMAs per core, at about 2,400 LUTs each (RESOURCES.md §3); does not fit in SLR0 |

## 8. Phase 2: the one-lane path (option c)

History:
- The brief stopped phase 2 after the design (§1, §4).
- The lead then asked for the compiler step that is valid under every option (§8.1).
- The author chose option (c) (decision above), which authorised the RTL steps (§8.2 onwards).

### 8.0 Summary of phase 2

Everything in this section is default-off. The delivered machine, the compiler's defaults and the run-1 image are unchanged. Nothing was loaded onto a card, and no full link was started.

| Step | What it is | Where it stands | Best measured effect |
|---|---|---|---|
| Tiled lowering (§8.1) | `TNCA_TILED_POINTWISE`: persistent tiled programs for 1-D element-wise kernels, two modes | built, tested, gated default-off; a default change is proposed as a patch | rtl, run-1 image, 20 points: mode 2 geometric mean 1.489x, none slower |
| Full-line write without fill (§8.2) | `HOME_LINE_WRITE_NOFILL` | built, 50 bench cases, mutant caught | rtl bench: a streaming full-line store needs 0 memory reads instead of 2,048 |
| Pipelined home (§8.2) | `HOME_LINE_PIPE`: three-stage line home, one line per cycle when it does not stall | built, 189 bench cases | part of s4 (§8.4) |
| Lines in flight (§8.2) | `HOME_MSHR_N` (memory tags), `LINE_SLOTS` (line LSU slots and NoC tags) up to 64 | built; tag benches §8.2 | part of s4; mostly unused (§8.5) |
| Pipelined outlets (§8.2) | `AXI_OUT_N` 32 with `AXI_PIPELINED_ISSUE` (existing RTL, new geometry) | board-top bench passes | part of s4 (§8.4) |
| PC interleave | `HBM_STRIPED` (existing RTL, R41) | in the run-2 candidate only | not measurable in RTL simulation (needs the 8 GiB window) |

**The measured result is far below the 19 GB/s of §6.** The best registered point on the candidate's parameters, clone fp32 at 64 KiB with mode 2, reaches 11.2 B/cycle/core, which is 3.4 GB/s for the chip (rtl, testbench memory, not board). That is 15× the run-1 image under the default compiler, but 18 % of the one-lane ceiling.

**Under the default compiler**, the candidate is 1.15× the run-1 image (geometric mean over 20 points, rtl), and 6 of the 20 points are slower: sum 0.72×, strided add 0.82×, gelu 0.86×, strided clone 0.87×, silu 0.90×, exp 0.94×. For the five such points that were checked, controls on G2's image without the G4 steps (§8.4) show that most of the loss comes from G2's reduced profile; gelu was not checked. The strided points also lose about 10 % to the G4 steps.

The limit is no longer the memory path. It is the number of lines a core keeps in flight (§8.5): the vector LSU path can only hold as many lines as the compiled code has registers for, and the line-beat transfer-engine path that would hold more was not built in this phase.

### 8.1 Compiler step: tiled element-wise lowering

`compiler/python/tnca_cc/pointwise_tiles.py`, option `TNCA_TILED_POINTWISE` (default 0), plumbed through the ahead-of-time options, the R1 adapter, the Triton backend and the Inductor integration (`TILED_POINTWISE_BLOCK=1024`, at least 8 programs). It applies to any 1-D element-wise Triton program whose loads and stores are contiguous in the program index; there is no operator list and no shape test.
- **Mode 1** keeps the tensor-engine tiles of §3.2 with two-tile software pipelining in each iteration. The descriptor base is rounded down to 8 bytes and the coordinate shifted.
- **Mode 2** uses the vector LSU with one tile base and immediate offsets (new `offset` attribute on load and store, encoded as `#imm`). Up to four lane groups issue the next groups' loads before the current group computes. A register estimate picks the group count; a copy needs no result register; if registers run out, the lowering retries with one group.
- The frontend hook refuses an untileable program wider than 128 lanes rather than emitting a slow one. A dead pure-integer vector pass (`drop_dead_pure`) removes the index arithmetic the tiles no longer use.

Tests: `compiler/test/test_pointwise_tiles.py` (7 tests, including RTL runs and a grouped-load test), in `TRITON_SUITES` and `make triton-e2e` as `G4-TILED-POINTWISE`. Mutants of the lowering are caught (`g4/test_mutants.log`).

Results (all mode 2 unless stated; NON-DEFAULT option, recorded in every record):

| Domain | Scope | Result | Evidence |
|---|---|---|---|
| model | 171 small points, run-1 configuration | geometric mean 1.268x, no point slower; mode 1 slower on 35–36 points | `g4/model_small_tiled{1,2}` |
| model | 171 small points, `g4` configuration | geometric mean 1.812x, no point slower | same |
| model | the 113-point non-regression set, run-1 configuration | 110 pass in both arms (3 exceed capacity in both), geometric mean 1.083, min 0.991, none slower by more than 2 % | `g4/model_nonreg_m{0,2}` |
| rtl | run-1 image, 20 registered 64 KiB points | mode 2: geometric mean 1.489x, min 1.028, max 1.950, none slower; mode 1: 1.297x, 5 points slower | `g4/rtl/rtl_table.md` |

**Gate for the default-off step** (`g4/gate_compiler/gate-logs.tgz`, sha256 `7d7cc682…`): the 113-point audit unchanged, `make compiler` 157 unit tests and 6 IR files, every RTL lane identical except `triton-e2e` (+7 G4 lines).

**Proposed default change** (`g4/default_change_tiled_pointwise.patch`, not applied): default 2 in `inductor.py` and `triton_backend/compiler.py`. Its gate (`g4/gate_default_change/`, sha256 `b8aefd0e…`): base `2a134be` against base plus the patch.
- Every lane exits 0 in both trees, with the same pass/fail lines.
- `make compiler` passes 157 unit tests and 6 IR files; `test_pointwise_tiles.py` passes 7 tests.
- Compile audit of the 113 frozen points: 128 pointwise kernels get the tiled code. The other 65 pointwise kernels, all 152 reductions and all 152 templates keep identical code, and no Triton source changes.
- The change has no board measurement, and on the candidate image mode 2 is measured only on the 20 points of §8.4.

Applying it is the lead's and the author's decision.

### 8.2 RTL steps

All are parameters of the existing RTL with the delivered behaviour as default.

1. **Full-line write without fill** (`HOME_LINE_WRITE_NOFILL`, `rtl/nuca/tnca_nuca_slice.v`). A store that writes all 64 bytes of a missing line allocates it without reading memory. `tools/r41/g4_home_tests.py`: 50 cases; a streaming full-line store pass needs 0 memory reads instead of 2,048 (`g4/home_nofill_tests.log`, mutant `g4/home_nofill_mutant.log`). Partial-line stores still fill: a fp16 copy whose 16-lane store covers half a line reads 2.01 lines per demand read (rtl counter, §8.4).
2. **Pipelined line home** (`HOME_LINE_PIPE`, `rtl/nuca/tnca_home_pipe.v`, with `HYBRID_LSU`). Three stages: S1 registered tag read; S2 hit/miss decision and metadata update; S3 data array, response and writeback queue. S3 holds the pipe when it cannot finish. The fill unit gives S3 priority, does not overwrite a way whose victim writeback S3 still holds, and issues no read while S3 holds that line's writeback. Memory issue is locked while stalled; replacement is invalid-first, then round-robin. The pipe ignores the DCO placement policies (they change only where a line lives, not what it holds) and sends no tile-last-word notices; both remain available with `HOME_LINE_PIPE` 0. Benches `tb/nuca/tb_home_pipe.v` (with a maintenance image check) and `tb_home_pipe_special.v`; `tools/r41/g4_home_pipe_tests.py`: 189 of 189 cases pass after the S3 split (`g4/step2/home_pipe_s3_tests.log`). The split came from the s3 screen (§8.3), whose worst path was 31 levels in the two-stage pipe's S2.
3. **Lines in flight.** `HOME_MSHR_N` widens the home's memory tag through the shell, the AXI adapter and the board top. `LINE_SLOTS` widens the line LSU slots (up to 64, in LUTRAM), the NoC payload (720 + TAG_W − 5 bits), the NI tags and the core's memory tag; the hybrid LSU's beat tag goes to at most 7 bits. `AXI_ID_W` is derived from `AXI_OUT_N`. Tag benches, written by the Kimi Code agent and corrected here (`g4/tag_tests/README.md`): line LSU 48 of 48 configurations, wide NI 12 of 12 (`TAG_W` 7, 722-bit payload), hybrid LSU 48 of 48 (`BEAT_W` 4–7, fast-slot peak 64 at `BEAT_W` 7). Two failed hybrid attempts are kept: a hang bound too short for the 1,024-cycle latency of the wide configurations, and a stimulus that could not fill 64 slots.
4. **Pipelined outlets.** `AXI_OUT_N` 32 with `AXI_PIPELINED_ISSUE` 1 (existing adapter). The board-top bench now forwards both (Kimi Code agent; `g4/outlet_bench/README.md`): the default geometry and the 32-outlet pipelined geometry pass.
5. **Board-top geometry at the candidate's parameters** (`g4/geometry/`). `tools/r40/image_geometry_tests.py` now forwards the G4 parameters. At the s4 parameters (the candidate without striping, which the bench's 1 MiB window cannot hold), it runs two grids, the memory controls 1/1, 1/32, 3/1 and 3/32, the coherent window and the ownership handoff.
   - The first run failed: one result line read back as zero under 1/32.
   - A trace of that line showed the device correct: the writeback was issued, and AW/W/B completed before the handoff finished.
   - The cause was the bench's shared-BO mirror (`tb/fpga/tnca_multicore_shared_hbm.vh`), which sliced the B ID as 3 bits. With 32 outlets, slot 8 mirrored slot 0's stale line. The same defect let the earlier 32-outlet bench pass by chance.
   - After the fix, s4 and the run-1 integration image pass, and so do `make r40-multicore-dma` and `make r40-multicore-mmio`.
   - The failures, the bisection (both `HOME_LINE_PIPE=0` and `AXI_PIPELINED_ISSUE=0` passed) and the trace are kept.

### 8.3 Candidate images and screens

`g4/images/image_{g2,s0..s4}.tcl` add the steps one at a time to G2's combined image (`resources/screens/image_g2_all.tcl`, profile `core-vrf-sync-v5-x8-g2`):

| Image | Adds |
|---|---|
| g2 | G2 combined image (screen: tile 0 72.73 % of SLR0 CLBs, WNS +0.603 ns) |
| s0 | hybrid LSU |
| s1 | + background reads, full-line write without fill |
| s2 | + pipelined home |
| s3 | + 32 MSHRs, 32 line slots |
| s4 | + 32 outlets with pipelined issue |
| run-2 candidate (`g4/run2/image.tcl`) | s4 + `HBM_STRIPED` 1 |

Screens on the x86 host, 10 ns, kernel synthesis then tile 0 placed and routed out of context in SLR0 (domain: screen):

| Image (tree) | Kernel synthesis | Tile 0 LUTs | Tile 0 CLBs (% of SLR0) | WNS | Worst path | Evidence |
|---|---|---:|---:|---:|---|---|
| g2 (G2's screen, RESOURCES.md §9) | — | 232,925 | 39,974 (72.73 %) | +0.603 ns | — | G2 |
| s3 (`caa25fb`, two-stage pipe) | passes | 247,738 | 41,967 (76.36 %) | +0.359 ns | home pipe S2 data path, 31 levels | `g4/screens/s3_caa25fb0a1d4/` |
| s4 (`64e3c71`, three-stage pipe; = run-2 candidate without striping) | passes | 246,873 | 41,415 (75.35 %) | +0.423 ns | vector FPU accumulator (`u_core/u_vfpu/acc_reg`), 4 levels | `g4/screens/s4_64e3c71d7f39/` |

The screens are out of context. Run 1's calibration for RAM-capture paths in a ≥ 96 % SLR is −0.9 to −1.0 ns from screen to image (§4.2). The one-lane path costs about 14k routed LUTs and 1.4–2.0k CLBs in tile 0 over G2 (screen). The outlets of s4 lie outside the tile, in SLR0, and are not in these rows. After the S3 split, the worst tile path is no longer in the home. A tile margin of +0.423 ns does not by itself predict closure of the full image at 100 MHz.

### 8.4 Measurements

RTL simulation, 64 KiB registered points, three cores, the testbench memory model (domain: rtl; not board). B/cycle/core is ArchBench demand bytes per cycle per core; at 100 MHz, 1 B/cycle/core is 0.3 GB/s for the chip.

Arms: `run-1 image` = G3's and §8.1's records; `s4` = `g4/images/image_s4.tcl` (the candidate without striping) on profile `core-vrf-sync-v5-x8-g2`. Default = `TNCA_TILED_POINTWISE` 0; mode 2 is NON-DEFAULT. The s4 mode-2 arm uses the grouped lowering (software tree `21fbf302…`, commit `2a134be`), and the run-1 mode-2 arm uses §8.1's records (software tree `8fd94a41…`).

| Point (rtl, 64 KiB) | run-1 image, default: cycles | s4, default: cycles | s4 / run-1, default | run-1 image, mode 2: cycles | s4, mode 2: cycles | s4 mode 2: B/cycle/core | s4 mode 2: GB/s, chip at 100 MHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| add bf16-small-contig | 138,001 | 100,718 | 1.37x | 77,466 | 43,900 | 1.49 | 0.45 |
| add fp16-small-broadcast | 119,521 | 106,033 | 1.13x | 75,203 | 42,796 | 1.04 | 0.31 |
| add fp16-small-contig | 137,923 | 100,789 | 1.37x | 78,090 | 43,761 | 1.50 | 0.45 |
| add fp16-small-strided | 153,050 | 186,527 | 0.82x | 118,979 | 129,567 | 0.51 | 0.15 |
| add fp32-small-contig | 85,707 | 56,235 | 1.52x | 72,235 | 13,809 | 4.75 | 1.42 |
| cast fp16-to-fp32-small-contig | 108,392 | 67,472 | 1.61x | 75,437 | 22,448 | 2.92 | 0.88 |
| cast fp32-to-fp16-small-contig | 102,461 | 91,660 | 1.12x | 73,238 | 25,089 | 2.61 | 0.78 |
| clone fp16-small-broadcast | 79,457 | 70,189 | 1.13x | 49,654 | 26,509 | 0.85 | 0.25 |
| clone fp16-small-contig | 97,583 | 85,939 | 1.14x | 52,246 | 9,032 | 4.84 | 1.45 |
| clone fp16-small-strided | 104,441 | 120,242 | 0.87x | 74,389 | 69,780 | 0.63 | 0.19 |
| clone fp32-small-contig | 59,644 | 34,981 | 1.71x | 49,684 | 3,889 | 11.23 | 3.37 |
| copy fp16-small-contig | 97,339 | 85,926 | 1.13x | 52,045 | 8,943 | 4.89 | 1.47 |
| copy fp32-small-contig | 59,282 | 34,971 | 1.70x | 48,525 | 3,924 | 11.13 | 3.34 |
| div fp16-small-contig | 146,848 | 142,650 | 1.03x | 91,889 | 43,671 | 1.50 | 0.45 |
| exp fp16-small-contig | 104,672 | 111,150 | 0.94x | 66,514 | 33,357 | 1.31 | 0.39 |
| gelu fp16-small-contig | 266,587 | 309,104 | 0.86x | 136,701 | 119,750 | 0.36 | 0.11 |
| relu fp16-small-contig | 97,566 | 93,164 | 1.05x | 51,874 | 22,192 | 1.97 | 0.59 |
| relu fp32-small-contig | 59,204 | 36,695 | 1.61x | 48,498 | 12,370 | 3.53 | 1.06 |
| silu fp16-small-contig | 133,177 | 147,818 | 0.90x | 94,523 | 73,534 | 0.59 | 0.18 |
| sum fp16-small-contig | 34,619 | 48,075 | 0.72x | 33,669 | 48,112 | 0.45 | 0.14 |

Summary (rtl, 20 points):

| Comparison | Geometric mean | Min | Max | Slower by > 2 % |
|---|---:|---:|---:|---:|
| s4 vs run-1 image, both default | 1.148 | 0.72 | 1.71 | 6 |
| s4 vs run-1 image, both mode 2 | 2.523 | 0.70 | 12.78 | 2 (sum, strided add) |
| s4 mode 2 vs run-1 image default | 3.758 | 0.72 | 15.34 | 1 (sum) |

**Controls** (default compiler, the five points that were slower). The ratio is the run-1 image's cycles over the image's cycles, so values above 1 are faster:

| Point | G2 image | s1 (hybrid, background reads, no-fill) | s4 |
|---|---:|---:|---:|
| sum fp16 | 0.64 | 0.71 | 0.72 |
| strided add fp16 | 0.91 | 0.75 | 0.82 |
| strided clone fp16 | 0.95 | 0.80 | 0.87 |
| exp fp16 | 0.80 | 0.90 | 0.94 |
| silu fp16 | 0.81 | 0.87 | 0.90 |

G2's profile (two contexts, fewer registers and barriers) is slower than run 1 on all five. The G4 steps recover part of it for sum, exp and silu. For the two strided points, they cost about 10 % against G2 (s1 is worse still). The cause has not been isolated. The rest of the table (`g4/rtl_s4/rtl_table.md`) also has mode 1 for the fp32 points (0.60–0.63 B/cycle/core, slower than the default) and mode 2 with the earlier compiler.

The counters behind these numbers are in §8.5.

Model, `onelane` configuration (the §6 path as far as the model has keys), mode 2 (domain: model, uncalibrated): small points at most 10.1 B/cycle/core; medium points copy fp32 13.5, add fp32 13.2, about 4 GB/s for the chip (`g4/model_onelane_{small,medium}_m*`). The home queues in the model's counters.

### 8.5 Why the measured numbers are far below 19 GB/s

Little's law on the rtl counters: lines in flight per outlet = line rate × mean latency.

| Point (s4, mode 2, rtl) | B/cycle/core | Lines per cycle per outlet | Mean outlet latency (cycles) | Lines in flight per outlet (≈ per core) | Home busy | Outlet busy |
|---|---:|---:|---:|---:|---:|---:|
| clone fp32 | 11.23 | 0.187 | 48 | 9.0 | 0.26 | 0.41 |
| copy fp16 | 4.89 | 0.162 | 78 | 12.6 | 0.29 | 0.31 |
| add fp32 | 4.75 | 0.079 | 56 | 4.5 | 0.13 | 0.22 |

- No stage is saturated. Home and outlet are busy a quarter to two fifths of the time.
- The §6 ceiling of 64 B/cycle/core is one line per cycle per core. At the measured 48–78-cycle latency, that needs about 50–80 lines in flight per core.
- The vector path holds 4–13. Each line in flight is a vector load whose destination register cannot be reused until it returns. The context has 64 logical registers, and the tiled lowering uses at most four lane groups, chosen by its register estimate (§8.1).
- The wider tags (32 MSHRs, 32 slots, 32 outlet IDs) are therefore not used. Raising them further would not help the vector path.
- copy fp16 reads 2.01 lines per demand line, because its half-line stores still fill (§8.2 item 1). That halves its useful rate.
- add fp32 issues three instructions per 16 elements plus the loop. Its IPC is 0.15 with a 14 % LSU stall, so the core's issue and dependency waits, not memory, set its rate.

The model's `onelane` configuration (§8.4) gives the same picture from the other side: with the memory path's queues modelled, medium points reach 13.5 B/cycle/core, and its counters show queueing at the home. The model does not represent the line-wide transfer engine either.

**What would use the path:** a requester that keeps lines in flight without registers, which is the line-beat, multi-command transfer engine of §6 (§8.6 item 1). The memory side built here (tags, pipe, outlets) is sized for it.

### 8.6 What remains before the one-lane path can reach its ceiling

1. **A line-beat transfer-engine path.** The tensor engine is command-serial: one transfer command at a time per core, so mode 1 cannot keep many lines in flight. Its line-wide, multi-command form (§6) was not built. It is the only general way to hold 64–80 lines per core without spending vector registers.
2. **Write-validate for partial-line stores.** fp16/bf16 element-wise outputs store half lines, which still fill (2.01 line reads per demand read). Per-byte valid bits in the home would remove that read.
3. **DCO placement policies and tile-last-word notices in the pipelined home.** The pipe uses invalid-first, round-robin replacement and sends no notices, so a program that relies on a DCO policy for locality loses it when `HOME_LINE_PIPE` is 1.
4. **Memory-level parallelism of the vector path.** Limited by registers and latency (§8.5); the compiler's grouping is already at four groups.
5. **Timing closure** of the run-2 candidate in the full image (§8.3), within the Stage 2 budget of five full runs.

### 8.7 Run-2 candidate

`g4/run2/image.tcl` and `g4/run2/README.md`. Not linked: the lead starts the link. It is s4 plus `HBM_STRIPED` 1.
- XO packaging with it passes on the x86 host (exit 0, no critical warnings; `g4/run2/package_dryrun.log`).
- The README lists the parameters against run 1, the evidence above and the link command (`flow/fpga/run_r40_attempt01.sh ROOT evidence/r41/hbm_bw/g4/run2/image.tcl`, per `evidence/r41/stage2/x86_link/README.md`).
- A link uses one of the five Stage 2 runs.

Two cautions for the lead:
- (a) The candidate is not a performance claim. The measured gain over run 1 under the default compiler is small, and some points are slower (§8.4).
- (b) Most of the gain needs `TNCA_TILED_POINTWISE` 2, which is not the default.


## 9. After phase 2 (coordinator's next steps, 2026-10-29)

### 9.1 Which G2 cut causes the slowdown (rtl)
One cut at a time on the run-1 image, on the five points that were slower under G2 (`g4/g2_cuts/`). **The contexts cut (4 → 2 per core) accounts for all of it.** Its geometric mean is 0.807 against 0.808 for all of G2's cuts together, ranging from sum 0.62× to strided clone 0.94×. The barrier/epoch, address, shared-read and LUTRAM cuts each change these points by at most 1 %. The mechanism is fewer requests in flight per core at unchanged request latency (sum: 4.14 → 2.47 outstanding, IPC 0.497 → 0.307). This is a measured cost of an approved capability cut (discipline §2.4.2), for the author.

### 9.2 Line-wide, multi-command tensor engine: TLM first (model)
`g4/tte_tlm/README.md`. The engine was modelled with default-off keys:
- line requests that hold engine line buffers instead of vector registers;
- overlapped commands;
- a store's window lease that ends when its registers have been read;
- a line-wide descriptor snapshot;
- optional combining of partial lines;
- full-line writes without a fill.

It was used through a new lowering, `TNCA_TILED_POINTWISE=3` (NON-DEFAULT): tensor-engine tiles streamed through a ring of windows, which falls back to mode 2 for programs it cannot stream. The mode is correct on the existing RTL (9 unit tests).

On medium fp32 points (model, uncalibrated), it is 1.0–1.8× the vector path:
- with a 2-cycle home: copy 1.00×, add 1.20×, relu 1.67×;
- with a 1-cycle home: copy 1.27×, add 1.58×, relu 1.76×;
- with ideal memory: 1.29–1.50×.

The best point reaches 39 B/cycle/core (ideal memory) and 28.6 (1-cycle home). With the default compiler, the line engine and full-line no-fill make no point slower; fp32 element-wise points gain about 1.2× and bmm 1.26×.

It does not reach the one-lane ceiling of 64 B/cycle/core:
- Inside a program, the engine already runs at about 0.9 lines a cycle on the core's single 64-byte port, which is the ceiling.
- The average loses about a third to the gap between programs (descriptor build, fence, pipeline fill).
- The ring holds at most 48 lines per core, against the 50–80 the design asks for.
- With a 2-cycle home, the home binds copy.

The RTL is not started; the decision to build it waits on this result.

### 9.3 Author decisions of 2026-10-29
Author decisions (source: the author, in the lead's chat, 2026-10-29, answering the lead's questions about the G4 §9 results):
1. **Restore 4 hardware contexts.** The 4→2 cut is withdrawn because of its measured slowdown. Keep every other G2 cut.
2. **Cut the per-program gap first, then the TTE RTL**, in order:
   - (a) reduce the inter-program cost generically;
   - (b) calibrate the TLM home against the RTL home;
   - (c) build the engine RTL, default-off.

**Decision 1 is done** (`g4/g2c4/`, RESOURCES.md §9):
- The profile is `core-vrf-sync-v5-x8-g2c4`: v5-x8 with 8 barriers and 38-bit epochs, and four contexts with 256 physical vector registers.
- `make core-g2c4` passes 86 of 86 runs. `t39_pair` runs as one pair: two pairs need 10 barriers, and a barrier index beyond a context's slice aliases instead of faulting, a pre-existing property.
- The images are `image_g2c4.tcl` and `image_s4c4.tcl`, and the run-2 candidate now has four contexts. Board-top geometry at `image_s4c4.tcl` passes (`g4/g2c4/s4c4_geometry.log`).

On the five points (rtl, default compiler; run-1 cycles divided by the image's cycles, `g4/g2c4/rtl/`):

| Image | strided add | strided clone | exp | silu | sum | geometric mean |
|---|---:|---:|---:|---:|---:|---:|
| G2, 2 contexts | 0.91 | 0.95 | 0.80 | 0.81 | 0.62 | 0.808 |
| G2 cuts, 4 contexts (`g2c4`) | 0.99 | 0.99 | 0.89 | 0.99 | 1.00 | 0.971 |
| s4, 2 contexts | 0.82 | 0.87 | 0.94 | 0.90 | 0.70 | 0.841 |
| s4, 4 contexts (`s4c4`) | 0.94 | 1.04 | 1.34 | 1.35 | 1.08 | 1.136 |

Restoring the contexts recovers the slowdown. Two exceptions remain:
- **exp on `g2c4` (0.89×).** Each kept cut alone left exp at 1.00×.
  - The same `g2c4` image runs exp at 104,400 cycles (1.00×) when the code is compiled for the 32-barrier target `core-vrf-sync-v5-x8` (`g4/g2c4/rtl/x_g2c4_profile_x8_exp.jsonl`). The loss therefore comes from what the compiler does for the 8-barrier target with four contexts, not from the hardware.
  - Explained and fixed in §9.6: the resized profile took the flip-flop scheduling model.
- **Strided add on `s4c4` (0.94×).** Codex owns the strided points.

### 9.4 (a) The per-program cost
- **Longer programs, and the compile failure that blocked them.**
  - The 4096-lane failure ("no run of 1 consecutive scalar registers") was a code-generator allocation gap. A transfer's coordinate run did not take back the registers of tokens and events past their last use, which `_alloc` already did for single registers. A long unrolled program leaves one token and one event per command.
  - `codegen._scalar_run` now does the same. A kernel that already fit keeps exactly its assignment, because the reuse happens only on the failure path.
  - Mode 3 now compiles 8192-lane programs (`TILED_STREAM_BLOCK = 8192`, with mode 2's block as the next candidate and the narrow one last).
- **Effect, model, medium fp32, `ideal` / `pipe1` memory:**
  - 2048 → 8192 lanes: copy 71,527 → 56,239 cycles (39 → 50 B/cycle/core), add 110,353 → 89,488, relu 116,111 → 101,443 (ideal);
  - copy 97,713 → 62,688 (pipe1);
  - in a command trace, the gap between program pairs stays about 263 cycles, but it now recurs every ~2,560 cycles instead of every ~840, about 10 % of the time.
- **Descriptor reuse across programs of a context.** `rctx_spmd` already builds a launch-invariant TensorMap once per context when a context loops over its programs. Under the physical-chunk dispatch the model and the board use, each dispatch runs one program per context, so the map is rebuilt each time.
  - Reusing it across dispatches needs a way to know that the context's descriptor slot still holds this launch's map, such as a launch-scoped validity bit in the runtime or the engine. Nothing in the ISA provides that today.
  - Proposed, not built. Its remaining value is bounded by the 10 % above.
- **Fence and pipeline refill.** The fence before a snapshot orders the context's scalar stores into the map before the engine reads it; it is part of the same per-program cost. The refill is the ring's first loads.
  - Both shrink with program length; neither was changed.
- **Mode 3 no longer takes programs with special-function or division operations.** In the registered-suite model run (below), gelu, sigmoid, silu, exp and div fp32 were 0.36–0.88× mode 2. Mode 2's lane groups overlap the SFU's 22-cycle occupancy with memory; mode 3's ring does not. Such programs now take mode 2's lowering. This is an operation-class rule, not a benchmark rule.

**Registered suite, model** (171 medium points, profile `core-vrf-sync-v5-x8-g2c4`, `cal` / `cal_line` memory, `g4/model_c4/`):

| Arm | pass | max B/cycle/core | geometric mean |
|---|---:|---:|---:|
| default compiler | 171/171 | 3.9 | 0.64 |
| default compiler + line engine | 171/171 | 3.9 | 0.64 (1.001× the default; none slower) |
| mode 2 | 171/171 | 21.7 | 1.25 |
| mode 3 + line engine | 171/171 | 33.5 | 1.30 |

- Mode 3 + line engine against mode 2: geometric mean 1.036×, max 2.66× (fp32 add/sub/mul).
- Ten fp16/bf16 broadcast and cast points are 0.977–0.98×: they take mode 2's lowering at the longer block.
- Against the default compiler: 2.01×, none slower.
- The best points reach 33.5 B/cycle/core (fp32 add/sub/mul) and 27.8 (fp32 copy/clone), about 10 GB/s for the chip (model). fp16/bf16 element-wise points stay at mode 2's ≤ 8.9, because the engine path needs 4-byte elements.

**RTL probes on `image_s4c4.tcl`** (64 KiB, rtl, `g4/g2c4/rtl_mode3/`):
- Mode 2: copy fp32 3,583 cycles (12.3 B/cycle/core), clone 3,581, add 9,506, relu 8,169.
- Mode 3 on the existing tensor engine: copy 51,627, add 105,797, relu 51,706, clone 51,294, 14× slower. That engine moves 8-byte units and runs one command at a time; mode 3 is for the line engine and must not be used without it.

### 9.5 (b) The home, calibrated against the RTL
`tools/r41/g4_home_calibrate.py` drives `tnca_home_pipe` (32 MSHRs, no memory stalls) at memory latencies of 10, 40 and 80 cycles. A new copy mode alternates streaming reads and full-line writes (`g4/home_cal/home_cal.json`; rtl, one slice). Cycles per request:

| Traffic | lat 10 | lat 40 | lat 80 |
|---|---:|---:|---:|
| hits | 1.02 | 1.03 | 1.05 |
| streaming read misses | 1.99 | 2.01 | 2.78 |
| streaming full-line writes, no fill (with victim writebacks) | 1.00 | 1.00 | 1.00 |
| copy (read + write alternating) | 1.26 | 1.27 | 1.54 |

- A read miss costs the home 2 cycles: the pipe's allocate, plus the fill unit's write of the data array.
- A full-line install costs 1, and a writeback costs nothing extra (its queue runs beside the pipe).
- Only at 80 cycles of latency do the 32 MSHRs bind reads.

The calibrated model occupancies, variant `cal`:
- hit 1, miss 1, fill 1;
- request 0, wake 0, writeback 0;
- install 0 (new key `tl_slice_install_occupancy`).

This lies between the 2-cycle (`pipe`) and 1-cycle (`pipe1`) brackets of §9.2: reads as `pipe`, writes as `pipe1`.

**The core side of the model is still uncalibrated.** The model is faster than RTL on the same image and settings:

| Point (s4c4, mode 2) | model `cal` | RTL | model faster by |
|---|---:|---:|---:|
| copy fp32 | 2,517 | 3,583 | 1.4× |
| add fp32 | 6,401 | 9,506 | 1.5× |
| relu fp32 | 4,718 | 8,169 | 1.7× |

Model gains are therefore upper bounds until the core pipeline is calibrated.

### 9.6 (c) The line-wide tensor engine in RTL, and the run-2 steps

Further decisions and directions of 2026-10-29:
- The author (course note): the goal is FP8 model performance. The engine must stream bytes for any element type, and it must serve the matrix engine's weight stream (X8 transposed transfers and matrix-operand loads).
- The coordinator: also cut the home read miss to one cycle a line and widen the outlets. Go to 64 lines in flight if the screens fit. Run 2 also carries the multi-precision matrix MAC lane (G2) and `FMA_IMPL=1`, so the screens must leave room for them.

**The engine** (`rtl/core/tnca_tensor.v` `LINE_ENGINE`, image parameter `TENSOR_LINE`, default 0; plumbed core → tile → top → shell → AXI → FPGA → XO and the board bench):
- A transfer whose rows are whole 64-byte lines moves as line requests through 32 line buffers (64 with `LINE_SLOTS 64`), with up to 8 commands in flight. A line must be inside the tensor, with line-aligned row starts and outer strides.
- The eligibility test has no element-size term. The engine moves bytes, so fp32, fp16/bf16 and fp8 lines take the same path.
- A TensorMap at a line-aligned address is read as four line requests.
- Every other transfer keeps the word walk, placed in a line of the same port. That includes the transposed X8 tile, a box with a partial line, and an unaligned map. The walk's register writes take the window port first and line writes take any other cycle, so a word-walk command no longer waits for line commands to drain (commit `4ed6c0d`).
- A store reads its next line's register while the current line waits (`7d645ec`).

**Verification (rtl):**
- `tb/core/tb_tensor_line.v` is a miter: the line engine and the word engine run the same programs, with random memory latency, request refusals and register-file refusals. Memories, register files, completions and faults must match.
- Slots carry fp32, fp16 and fp8 element sizes. Transposed 16-bit and 1-byte loads, some of them whole-line boxes, are interleaved with line commands.
- `tools/r41/tensor_line_tests.py`: 18 of 18 configurations pass (`g4/tte_rtl/miter_dtypes_xpose.log`).
- Mutants caught: register index, data, inside test, stride, open count, line count, descriptor data, two prefetch mutants, port arbitration (seeds 3 and 5), and the element size of the line count (`g4/tte_rtl/mutants_dtype.log`).
- Removing `!t_xpose` from the eligibility is an equivalent mutant: the transposed branch is taken first.

**Defects found on the way, and what they change by default:**

| Defect | Effect | Fix | Default behaviour |
|---|---|---|---|
| A mode-3 kernel not wrapped by `rctx_spmd` had no fence between its descriptor stores and `td.load` | the line snapshot read a stale word; runtime fault on copy | `rctx_spmd.publish_scratch_maps`, called by `aot._emit` | **changes default code**: one fence before the first snapshot of a map written since the last fence. Gate audit compare pending |
| With `HYBRID_LSU`, a fence or arrive could pass an access still in X | race on the line LSU path | fence and arrive also wait for `x_mem_claim` | only with `HYBRID_LSU` (off by default) |
| A resized profile (x8-g2, x8-g2c4) was scheduled with the flip-flop default model | exp 0.89× on `g2c4` (§9.3) | `schedule.Model` takes the longest listed target prefix; `tools/r39/schedule_tests.py` checks every banked target | **changes code** for those profiles only; g2c4 five-point geometric mean 0.971 → 0.996, exp 1.00× (rtl, `g4/g2c4/rtl_sched_fix/`) |
| `7b646be` did not elaborate in Vivado with `LINE_ENGINE 0` when the core's slot tag is wider than the port tag | the s4c4 kernel synthesis failed. Icarus accepted it, so the gates did not see it | zero-extend the response tag and data before slicing (`4ed6c0d`); Vivado elaborates both settings (`g4/tte_rtl/vivado_elab_le0_le1.log`) | none after the fix |

**Tried and rejected** (rtl, s5, copy fp32 64 KiB, mode 3):
- A ring sized by the context's barrier slice: 5,599 against 5,144 cycles.
- `TILED_MIN_PROGRAMS 16`: 5,984.

**RTL probes** (small 64 KiB registered points, profile g2c4, cycles; `g4/tte_rtl/probes/`):

| Point | s4c4 mode 0 | s4c4 mode 2 | s5 mode 3, `7b646be` | s5 mode 3, `247f43d` |
|---|---:|---:|---:|---:|
| add fp16-contig-r32 | 68,135 | 26,326 | 33,912 | 33,747 |
| add fp16-strided-r32 | 170,108 | 127,603 | 180,960 | 180,998 |
| add fp32-contig-r16 | 37,628 | 5,783 | 7,467 | 7,128 |
| cast fp16-to-fp32-contig-r32 | — | 14,013 | — | 18,371 |
| cast fp32-to-fp16-contig-r32 | 57,414 | 15,502 | 19,129 | 19,096 |
| clone fp16-contig-r32 | 55,937 | 7,331 | 7,427 | 7,433 |
| clone fp16-strided-r32 | 100,050 | 68,289 | 96,182 | 96,176 |
| clone fp32-contig-r16 | 23,522 | — | 5,141 | 5,105 |
| copy fp16-contig-r32 | 55,934 | 7,340 | 7,445 | 7,446 |
| copy fp32-contig-r16 | — | — | — | 5,017 |
| div fp16-contig-r32 | 93,416 | 37,655 | 51,207 | 48,306 |
| exp fp16-contig-r32 | 85,952 | 28,675 | 37,572 | 37,986 |
| silu fp16-contig-r32 | 111,976 | 53,360 | 65,576 | 64,768 |

The run-2 steps on the same small points (rtl, cycles). These check function; the points are too small to be bandwidth-bound:

| Point | s5 | s6 (chained fills) | s5w (512-bit outlets) | s6w | s7 (64 in flight) |
|---|---:|---:|---:|---:|---:|
| sum fp16-contig-r32 (mode 0) | 31,287 | 31,479 | 31,008 | 31,124 | 31,664 |
| clone fp16-contig-r32 (mode 2) | 7,337 | 7,357 | 7,318 | 7,295 | 7,290 |
| copy fp32-contig-r16 (mode 3) | 5,017 | 4,947 | 4,900 | 4,903 | 4,892 |

- fp16 points take mode 2's lowering under mode 3, because the compiler's tensor-engine accesses still require 4-byte elements. The register layout of 16- and 8-bit vectors, not the engine, sets that limit.
- Programs that mode 3 hands to mode 2's lowering run at its 8192-lane block, and that is slower than mode 2 at its own block (rtl, s5 mode 3 vs s4c4 mode 2): fp16 add 0.78×, exp 0.76×, div 0.78×, silu 0.82×, strided clone 0.71×. The fallback should use mode 2's block. This affects only the non-default compiler mode, and it is open.

**Home read miss at one cycle a line** (`HOME_LINE_PIPE 2` → `tnca_home_pipe` `FILL_CHAIN`):
- A read miss cost two cycles because the fill unit idled one cycle between MSHRs. It now starts the next filled MSHR in the cycle it finishes one.
- `tools/r41/g4_home_pipe_tests.py --fill-chain 1`: 189 of 189 cases pass.
- Streaming read misses, image slice geometry, cycles per line (rtl, one slice, `g4/home_chain/`):

| MSHRs / requester tags | fill chain | lat 10 | lat 40 | lat 80 |
|---|---|---:|---:|---:|
| 32 / 32 | off | 2.01 | 2.02 | 2.78 |
| 32 / 32 | on | 1.01 | 1.52 | 2.77 |
| 64 / 128 | off | 2.01 | 2.02 | 2.04 |
| 64 / 128 | on | 1.01 | 1.02 | 1.38 |

- One line a cycle needs both the chain and about one line in flight per cycle of round trip. The board bench's round trip is 25–63 cycles, so 32 lines in flight bind first.
- A copy stays at 1.26–1.29 cycles a request. A fill and a full-line install share the data array's write port.
- The calibration bench's DRAM model silently dropped reads beyond 32 in flight. Its pool is now a parameter.

**Outlets.** `tnca_line_axi` already supports a 512-bit data width (one beat a line). The board bench now takes `AXI_DATA_W`. Screen, one outlet in SLR0 at 10 ns, `LINE_LUTRAM 1`, pipelined issue, striped (screen, `g4/outlet_screen/`):

| Outlet | LUTs | registers | CLBs | WNS |
|---|---:|---:|---:|---:|
| 256-bit, 32 slots (s5, s6) | 3,147 | 5,405 | 1,088 | +3.916 ns |
| 512-bit, 32 slots (s6w) | 2,436 | 5,367 | 997 | +3.039 ns |
| 512-bit, 64 slots (s7, run-2 candidate) | 4,387 | 9,808 | 1,956 | +3.565 ns |

A 512-bit outlet is smaller than a 256-bit one. Its cost is the platform's 512→256 width converter in front of each HBM port, which v++ inserts and an out-of-context screen does not see. Each 256-bit HBM port at 450 MHz carries 14.4 GB/s, above a 512-bit outlet's 6.4 GB/s at 100 MHz.

**Screens** (tile 0 placed and routed out of context in SLR0 at 10 ns; kernel synthesis LUTs for the whole kernel; screen domain, x86 host):

| Image | what it adds | kernel LUTs | tile LUTs | tile CLBs (% of SLR0) | WNS |
|---|---|---:|---:|---:|---:|
SCREEN_ROWS

**Run-2 candidate** (`g4/run2/image.tcl` = `image_s7.tcl` + `HBM_STRIPED 1`):
- four contexts;
- the line engine (`TENSOR_LINE 1`);
- chained fills (`HOME_LINE_PIPE 2`);
- 512-bit outlets;
- 64 lines in flight per core and per home (`LINE_SLOTS`, `HOME_MSHR_N`, `AXI_OUT_N 64`).

G2 adds `FMA_IMPL=1` and the matrix MAC lane to the same file. The new options are default-off in the RTL and stay off until the gate.

Not on the line path yet: the transposed X8 tile, which is the FP8 weight stream's corner turn.
- It still reads 8-byte units, one per source row per column block, so it moves at most 8 bytes a cycle.
- A line version reads 16 whole rows and turns 64 columns. Its buffer is 1 KiB instead of 128 bytes, and its output multiplexer is about 8× wider.
- Proposed as the next RTL step if the profile of FP8 decode shows transposed weight loads on its critical path. Non-transposed matrix-operand loads of whole lines already take the line path.

### 9.7 Semantic hardware prefetch into the home (evaluation, 2-hour time box)
The author's idea (source: the author, in the lead's chat, 2026-10-29), relayed by the coordinator: hardware prefetch through the home/TMU, using either (a) hints from the line TTE about the rest of a descriptor's box, or (b) a stream detector on sequential read misses. Prefetched lines are clean, non-reserving and droppable.

**Decision: no-go for run 2. The design is kept for later.** The reasons, with numbers:

1. **What prefetch buys is a smaller core window, and the run-2 candidate already has a large one.**
   - Prefetch hides the memory round trip from the core: a demand that finds its line in the home waits only for the core–home round trip.
   - The other route is enough lines in flight per core. Kernel synthesis, s5 → run-2 candidate, per tile:
     - 32 → 64 slots costs +957 LUTs in the tensor engine and +848 in the line LSU;
     - 32 → 64 MSHRs costs +4,367 in the home pipe;
     - the 512-bit 64-slot outlet costs +1,170 per outlet.
     The tile screen of the candidate is in the screens table.
   - A prefetcher is estimated at 2–5 k LUTs per tile (detector or TTE look-ahead, a request injector at the home arbiter, a target-less MSHR, a NoC request class for (a)). That buys no more than the 64 slots already do.
2. **The upper bound of the gain is measured.** Home bench, streaming reads with chained fills, rtl, one slice, cycles per line (§9.6):
   - 32 lines in flight with 32 MSHRs: 1.52 at 40 cycles of latency, 2.77 at 80 (42 and 23 B/cycle);
   - 128 in flight with 64 MSHRs: 1.02 and 1.38 (63 and 46 B/cycle).
   - Prefetch with a 32-line core window can at best reach the second row, and only if the home itself keeps about one line in flight per cycle of memory latency. Prefetch therefore needs the 64 MSHRs as well: it moves the window from the core to the home but does not shrink it.
   - With 64 core slots and the bench's 25–63-cycle round trips, a stream is already port-bound (64 lines ÷ ~60 cycles ≈ 1 line a cycle), so prefetch adds nothing there.
   - It would add bandwidth only if the board's round trip exceeds about 64 cycles. That round trip has not been measured (LINK_BUDGET.md).
3. **Schedule and risk.** Run 2 is due within two days, behind the gate. Prefetch adds:
   - a third request source to the home arbiter;
   - a target-less MSHR state in the fill unit;
   - interactions with the same-line ordering rule, the writeback-before-read rule, maintenance drain and DCO policy;
   - for (a), a new NoC request class.
   Each needs the randomized home regressions and the board-top geometry bench.

**Sketch for later** (all default-off; no recognition of names or shapes):
- **Home side, common to (a) and (b).** A prefetch request enters the arbiter only when no demand is eligible. It is dropped in S2 on a hit, on a merge, or when fewer than *k* MSHRs are free. Otherwise it allocates an MSHR with no target and claims a clean or written-back victim. The fill unit installs the line clean (no dirty bit, no response) and frees the MSHR. A demand that arrives first merges as an ordinary target. Maintenance drains prefetch MSHRs like any other.
- **(b) Stream detector per home.** Four entries of {line address, stride = the home interleave}. On a demand read miss that continues an entry, issue up to *k* prefetches at distance *d* ≤ 16 lines per stream. That bounds prefetched lines to 4 × 16 = 64 of the slice's 256 lines.
- **(a) TTE hints.** The line engine already walks the box's line addresses. A second walker runs up to *d* lines ahead of the issued window and sends prefetch requests (a request bit on the line NoC) at the lowest arbiter priority.
- **When to revisit** (coordinator, 2026-10-29): the decision depends on the board's HBM round trip, which is unmeasured. If the first run-2 board run shows more than about 64 cycles of latency at the cores, prefetch is the next step for run 3. It is also worth revisiting if the FP8 decode profile shows the vector (non-TTE) load path latency-bound.

## Reproduction

```sh
# analytic issue ceilings (GB10; torch needed for the registry)
PYTHONPATH=$PWD:$PWD/python $TORCH_PYTHON tools/r41/g4_issue_bound.py \
    --census evidence/r41/hbm_bw/link_budget/census/census.json \
    --kernels evidence/r41/hbm_bw/link_budget/census/census_kernels.tgz \
    --suite archbench/suites/r41_membound.json --out evidence/r41/hbm_bw/g4/issue_bound.json
# model (GB10; build: tools/performance_harness/systemc/README.md)
PYTHONPATH=$PWD:$PWD/python $TORCH_PYTHON tools/r41/g4_model_suite.py --binary BUILD/tnca-chip \
    --out OUT --size small --jobs 3
python3 tools/r41/g4_report.py      # every table above, from the recorded files
```
Raw model records: `g4/model_<size>/{run1,g4,ideal}.log`, `*/results.json`, `*/counters.json`, the configurations and the exact commands (`*.command.json`). Runs were made on GB10 from this branch's tree at 28342ce, plus the three `tools/r41/g4_*.py` scripts committed with this document.

Phase 2 (§8), on GB10 unless stated:
```sh
# one RTL probe: a registered point on a candidate image (rtl; records carry the software tree)
$TORCH_PYTHON tools/r41/link_budget_rtl.py --image evidence/r41/hbm_bw/g4/images/image_s4.tcl \
    --profile core-vrf-sync-v5-x8-g2 --point op.clone/membound/fp32-small-contig-r16-c1024 \
    [--set TNCA_TILED_POINTWISE=2] --out OUT.jsonl
python3 tools/r41/g4_rtl_report.py --arm "NAME=GLOB" ...   # the §8.1/§8.4 tables (g4/rtl_s4/rtl_table.md)
python3 tools/r41/g4_home_tests.py; python3 tools/r41/g4_home_pipe_tests.py      # home benches
python3 tools/r41/lsu_line_tests.py; python3 tools/r41/wide_noc_tests.py; python3 tools/r41/lsu_hybrid_tests.py
python3 tools/r40/image_geometry_tests.py --image-tcl evidence/r41/hbm_bw/g4/images/image_s4.tcl \
    --image-profile core-vrf-sync-v5-x8-g2
$TORCH_PYTHON compiler/test/test_pointwise_tiles.py -v
```
Batch scripts and job lists are kept next to the records (`g4/rtl/`, `g4/rtl_s4/*/worker.sh`, `jobs.txt`). Screens were run on the x86 host with the G2 screen chain (kernel synthesis, then tile 0 out of context in SLR0 at 10 ns). Their logs and reports are in `g4/screens/`.
