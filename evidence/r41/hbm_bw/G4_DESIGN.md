# G4: a general full-link data path for memory-bound operators — design and bound

Status: workstream G4 of [`BANDWIDTH_PLAN.md`](BANDWIDTH_PLAN.md), phase 1 (design), 2026-10-28. **[`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md) is binding and takes precedence over this document.**

Inputs: [`LINK_BUDGET.md`](LINK_BUDGET.md) (G3), [`RESOURCES.md`](RESOURCES.md) (G2), [`AUDIT.md`](AUDIT.md) (G1). Branch `r41-g4`, made from `origin/r41-hbm-bw` with `origin/r41` merged in (the stream engine stays removed).

Domains: every number is labelled **board** (run-1 image, 100 MHz), **rtl** (RTL simulation), **model** (SystemC/TLM, uncalibrated), **screen** (out-of-context synthesis or place and route), or **analytic** (arithmetic on recorded numbers or on compiled code). An **estimate** is an engineering estimate with no run behind it, and is marked as such. Nothing here is a board measurement of a new design.

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

**Recommendation.** Per the brief, phase 2 (RTL) is not started. The decision goes back to the lead and the author:
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

## 8. Phase 2

Not started. Under the brief, phase 2 continues only if the design shows the target is reachable within area and timing; §1 and §4 show that it is not. The recommended next step under any of options (a)–(c) is the compiler lowering of §3.2, default-off until the 113-point gate. It is general, it needs no new hardware, and it can be measured with G3's RTL link probes and the model.

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
