# R41 hbm8: an 8 GiB device window on the current three-core image

Scope: whether R41 FP8 model work (Qwen3-VL-2B-FP8, Qwen3.5-4B-FP8) can plan
on a ≥6 GiB, target 8 GiB, device memory window on the delivered B40 image
`a88ebdf5…` (three cores, 90 MHz, UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`).
Covered here: the RTL, RTL simulation, the runtime and backend, and the
registered board experiment. The TLM-model run and the compiler
address-arithmetic audit are separate agents (§6).

## Conclusion

**8 GiB is available on the current image, subject to the board run. No new
image is needed.**

* **The 4 GiB window was an allocation choice, not a limit.** R40 raised the
  runtime's 1 GiB default to 4 GiB because the frozen suite's large Llama
  needs 2,996,965,376 bytes of parameters. A 4 GiB XRT preflight passed and
  reported PC0..15, and the higher PCs were then labelled "outside this
  proposed exposed window"
  (`evidence/r40/v2/convergence/README.md`, "Repaired original GEMM and
  declared capacity", 2026-10-18). No size above 4 GiB was attempted on this
  image.
* **The image connects every outlet to all 32 PCs.** A static, read-only
  parse of the xclbin (`xclbin_hbm_map.json`) shows the following:
  * all three m_axi arguments resolve to the same group bank, `MBG[0:31]`
    (memory index 41, 8 GiB, base 0);
  * there are 96 plain connectivity rows (3 × HBM[0..31]);
  * an 8 GiB BO at base 0 maps contiguously onto PC0..PC31 (topology
    `e3cf166d…`).
* **The RTL address path is 64-bit end to end.** This was audited on the
  image's exact source, git 54249f7 (`audit_54249f7.md`). The path covers the
  LSU and TXF engine, the NoC flit, the home hash, the 53-bit full cache tags,
  MSHR and victim addresses, the outlets, `tnca_line_axi`, the MMIO window
  registers and range checks, the DMA handoff and the 64-bit m_axi ports.
  * Nothing narrower than 64 bits limits 8 GiB or can make two window
    addresses share data.
  * The TMU decode compares address bits [63:12], so only the single page
    0x7FFF0000 is MMIO, and the runtime already reserves it.
* **RTL simulation passes.** The image RTL ran in simulation with an 8 GiB
  window and the B40 cache geometry, on the board top with all three cores
  and outlets. Every standard-Torch case passes, both at BO base 0 (as on the
  board) and at BO base 4 GiB (§2).
* **The runtime needs no change.** Passing `hbm_bytes=8589934592` in the
  backend args is enough (§3).
* **A precedent exists on the same platform.** The single-core image
  `e576c850` already passed an 8 GiB BO at base 0 over PC0..PC31 on this
  host, with addresses above 4 GiB, DMA and in-place device updates
  (`evidence/r40/R40-04/ram_image/`).

What only the board can settle (registered, §4):

* XRT really allocates the 8 GiB group BO for this xclbin, with enough host
  memory.
* The platform interconnect really routes PC16..PC31 for all three masters.
* The >2 GiB single-storage kernels behave on the device as they do in
  simulation.

If the board run passes, the Stage 1 statement "a ≥6 GiB window means a new
image exposing more PCs, plus >4 GiB addressing"
(`evidence/r41/stage1/stage2_readiness.md`, S2-04) no longer holds for B40.

Planning facts for the FP8 models:

* **Usable heap.** The 8 GiB window gives 8,589,930,496 usable heap bytes.
* **Largest single storage.** The reserved TMU page splits the heap into
  0x20000–0x7FFF0000 (~2 GiB) and 0x7FFF1000–0x2_0002_0000 (~6 GiB). The
  largest single storage is therefore ~6 GiB. In the R40 4 GiB window it was
  ~2 GiB (a 2.25 GiB tensor is refused there).
* **Same-device eager copies stop at 2 GiB.** `torch_device/copy.py` refuses
  spans of 2^31 bytes or more, with the error "64-bit copy indexing is not
  implemented". `copy_storage` (allocator resize) also needs fewer than 2^31
  bytes. Larger storages are filled by host DMA (chunked) or by compiled
  kernels. The compiler agent's worktree is changing `copy.py` (§6).
* **Host transfers cost several times their size in host memory.** Transfers
  go through Python lists of 8-byte words (`Transport.upload`), so transient
  host memory is several times the transfer. The board cases stage in 64 MiB
  pieces.
* **The BO pins host RAM.** Each process that opens the 8 GiB window pins an
  8 GiB host-backed XRT BO. The FPGA host has 62 GB, of which 21–24 GB was
  available on 2026-10-22, so the queue requires at least 16 GiB
  MemAvailable.

## Why R40 used 4 GiB

`doc/r40_report.md`: "4 GiB exposed HBM window over PC0..15; PC16..31 not
exposed and not certified". The reason is in the convergence record above:

* The collectors' `--hbm-bytes` was raised from 1 GiB to the smallest window
  that held the frozen subset (measured peak 3,098,945,024 B, headroom 1.39).
* The 4 GiB BO is an XRT allocation of the first 4 GiB of the same 8 GiB
  group. Its base is 0, so it covers PC0–15.
* Nothing in the RTL, XRT or runtime refused a larger size. 8 GiB was simply
  never requested on this image.

## 1. RTL audit (image source 54249f7)

`audit_54249f7.md` has the full table, with file:line for every address
carrier.

**Same RTL as the image.** The 81 link-source files on the FPGA host
(`~/tnca_src_r40_attempt03_54249f7`) hash-equal `git archive 54249f7`.

**Frozen tree.** The frozen Stage 1 tree differs in 7 RTL files:

* `tnca_vector_fpu.v` (FPU only);
* six SLR-timing edits that register control copies.

None of them narrows an address. In the frozen `tnca_line_axi.v`,
`base_q`, `bound_q` and `offset` are all 64-bit.

Every field narrower than 64 bits, and whether it limits 8 GiB:

| Narrower field | Where (54249f7) | Limits 8 GiB? |
|---|---|---|
| per-lane gather/scatter index, 32-bit signed, added to a 64-bit base | `rtl/core/tnca_lsu.v:67, :404-406` | no — one indexed instruction spans ±2^31 elements around a 64-bit base |
| cache tag 53 bits = addr[63:11] (SET_W 5, FULL_LINE_TAG 1) | `rtl/nuca/tnca_nuca_slice.v:191, :212-213`; `rtl/tile/tnca_tile.v:336-337` | no — full tag |
| TMU MMIO page compare addr[63:12] == 0x7FFF0 | `tnca_nuca_slice.v:861`; `rtl/tnca_top.v:84` | no — one page; runtime reserves it (`python/tnca/runtime.py:58`) |
| DCO tile hint: 3-bit tensor index + 13 bits of addr>>lsb | `tnca_nuca_slice.v:320-330`; `rtl/nuca/tnca_tmu.v:273-275` | no — replacement/dead-block hint only |
| home hash 8 bits = XOR of all 8 bytes of the 64-bit line index, mod 3 | `rtl/noc/tnca_core_ni.v:83-91` | no — every bit enters |
| R_AXI_STAT local limit, 4 KiB pages in [63:32] | `rtl/fpga/tnca_multicore_mmio.v:74` | no |
| AXI-Lite 32-bit halves, assembled 64-bit write | `rtl/fpga/tnca_axi_lite.v:28-45, :145` | no |
| TXF box counters, register-file window offsets (not memory addresses) | `rtl/core/tnca_tensor.v:275, :285, :288, :429` | no |

Unit RTL check: `tb/fpga/tb_line_axi_hbm8.v` passes 35/35 on both the image
and the frozen `tnca_line_axi.v`. It checks:

* that legal lines at 0, 2, 4, 6 and 8 GiB−64 issue exactly `base + offset`;
* refusals at 8 GiB, 12 GiB and 16 GiB, below MEM_BASE, on unaligned
  addresses and on a 2^64 wrap.

It does this at BO bases 0, 4 GiB and 2^64−8 GiB. Three mutants are each
caught:

* the bus address cut to 33 bits (caught at base 4 GiB; it is invisible at
  base 0, where 8 GiB needs only bits 0–32);
* the limit check removed;
* the alignment check removed.

Logs: `unit/`.

## 2. RTL simulation above 4 GiB

**Stand-in.** `fpga-multicore-sparse-sim` (`tools/performance_harness/hbm8_sim.py`)
is the board top `tnca_multicore_hbm_xo` with three cores, three outlets,
DMA handoff and memory controls, run through the production
`MulticoreMMIOBackend`. It adds two things:

* a stand-in BO base: 0 as on the board, or 4 GiB;
* a sparse AXI memory (`tb/common/tnca_sparse_map.vh`). Only the ranges a
  case touches are backed; any access to another line of the 8 GiB stops the
  simulator with its address. A dropped or extra high bit therefore fails
  loudly instead of reading zero.

**Configuration.** The RTL is git 54249f7 (the image) unless marked. The
B40 cache geometry (32 sets × 8 ways), registered victim/rank/TMU options and
2-stage links come from `image.tcl`. Every case is standard Torch: tnca
tensors, `torch.compile(fullgraph=True)`, Inductor, Triton and the TNCA
device. The same tool, `hbm8_case.py`, is what the board runs.

| Case | What it executes | image RTL, base 0 | image RTL, base 4 GiB | frozen RTL, base 0 |
|---|---|---|---|---|
| window | 8 GiB window, registers 0xA8/0x100/0x108/0xB0/0xB8 readback, heap 0x20000–0x2_0002_0000, reserved page, PC0–31 map | pass 9/9 | — | — |
| samples | 34 fp16 probes: every PC boundary (31 crossings), both BO edges (device 0x20000 and the last 512 B before 0x2_0002_0000), virtual 4 GiB; DMA in, compiled `add_` on 3 cores, DMA out | pass 75/75 | WAVE3 | WAVE3 |
| alias | pairs differing in bit 30/31/32/32+31 and TMU page +2/+4/+6 GiB (0xFFFF0000, 0x1_7FFF_0000, 0x1_FFFF_0000): distinct values, device load first, then device store | pass 35/35 | WAVE3 | — |
| crossing | device store then device load across virtual 4 GiB, physical 4 GiB and the window end; DMA out of the stored lines, DMA in and device load again | pass 16/16 | WAVE3 | WAVE3 |
| bigoffset | one 2.25 GiB storage after the TMU page; Inductor kernels `in_ptr0 + (x0 + 16777216*x1)`, int32 element offsets up to 587 M, i.e. byte offsets up to 2.35 GB, read and write | pass 5/5 | — | — |
| dma | DMA in at the window top, device copy to 6 GiB, DMA out of both | pass 5/5 | — | — |
| stream | (sim size: 8 KiB, across physical 4 GiB) DMA in, device row reduction, DMA out | pass 6/6 | — | — |
| negatives | refusals at the 8 GiB end (allocator, host, MMIO range: past end / at end / bit 33 / 2^64 wrap / handoff); host-window write+read of the last word and DMA read of it | pass 9/9 | — | — |
| clamp | request 16 GiB: exposed 8 GiB, heap = 8 GiB | pass 2/2 | — | — |

**What every launch showed:**

* each launch ran programs on all three cores (e.g. 8/4/4 programs for a
  probe);
* the three outlets all carried lines;
* `R_AXI_STAT` stayed clear at the end of every case.

Reports are in `sim/` (one `report.json` per case, with its sparse map).

## 3. Runtime and backend readiness

No runtime or RTL change is needed. Checks are in
`tools/performance_harness/test_hbm8.py` (21 tests; no card, no torch). They
drive the production link and backend code with the B40 xclbin's real
topology:

* **u50 backend `hbm_bytes`.** `hbm_bytes=8589934592` allocates 8 GiB in
  memory index 41. `R_AXI_BASE` = 0 and `R_AXI_SIZE` = 0x2_0000_0000 are
  written and read back. The heap and `link.words` then end at
  0x2_0002_0000.
* **XRT BO in HBM[0:31].** `MulticoreXDMALink` requires all three arguments
  to resolve to one group (mismatched groups are refused). The allocation
  records PC0..PC31 contiguous, with PC16 at 4 GiB.
  * A 16 GiB request is clamped to the 8 GiB group and records requested
    vs exposed.
  * The R40 4 GiB request maps PC0..PC15.
* **Allocator.** In an 8 GiB heap the reserved page 0x7FFF0000–0x7FFF1000 is
  kept (capacity 8 GiB − 4 KiB) and a straddling allocation is moved past it.
  * A 2.25 GiB storage lands at 0x7FFF1000.
  * The largest single storage is window end − 0x7FFF1000 (~6 GiB).
  * A refusal leaves the allocator unchanged.
* **No high-bit truncation.** DMA offsets for device addresses
  0x1_0000_0000 − 8, 0x1_0000_0000, 0x1_0002_0000, 0x1_7FFF_0000 and the
  last line reach `bo_write`/`bo_read` as exactly `addr − 0x20000`. The
  ownership handoff covers the same range.
* **Out-of-bounds refusals.** Host transfers past the window are refused
  before any handoff or bus access. The XRT accessor's own BO bounds refuse
  an overrun, an end read and negative lengths or offsets.
* **Device-side out-of-bounds stays in RTL.** An access past the window
  latches a sticky AXI fault that only an image reset clears. It is tested in
  RTL (§1), never on the shared card.

## 4. Board experiment (registered, not run)

**Manifests.** There are two manifests (`spec/r41/hbm8/experiments/`),
generated by `tools/performance_harness/hbm8_manifest.py`; `--check` passes.
Splitting them means a failure in the >2 GiB compiler path cannot stop the
window qualification. Each fpga step runs `hbm8_case.py` once, in a fresh
process with its own session. `experiment.py` probes the loaded UUID, the
90 MHz clock counter and kdsstat before and after every step.

| Manifest | Steps | Est. |
|---|---|---|
| `board-hbm8` | window, negatives, samples, alias, crossing, dma (256 MiB in, device copy top→6 GiB, 2×256 MiB out), clamp, summary | 0.42 h |
| `board-hbm8-big` | bigoffset (2.25 GiB storage, strided read/write past 2^31 bytes); stream (2.25 GiB single storage at 0x7FFF1000–0x1_0FFF1000: 36×64 MiB DMA in, device row reduction over all of it, 36×64 MiB DMA out, exact checks), summary | 0.27 h |

**Board identity.** The board block is the Stage 1 B40 identity with
`hbm_bytes` 8 GiB (16 GiB for `clamp`) and memory map `PC0-PC31`, sourced
from `xclbin_hbm_map.json`.

**Twin manifests.** `hbm8_manifest.py --sim DIR` writes simulator twins of
both manifests: the same steps as kind `rtl` on the sparse stand-in. Their
runs are in `sim/twin/`.

**Launch script.** `launches/hbm8_queue.sh` is modelled on the Stage 1
`board_queue.sh`, but never loads anything. Its gates run in this order:

1. The Stage 1 queue has logged BOARD-QUEUE-DONE, or its process has exited.
2. The Stage 1 idle rule: 15 consecutive idle minutes in kdsstat.
3. The loaded UUID is B40 and the clock counter reads 90 MHz; otherwise the
   script stops.
4. MemAvailable is at least 16 GiB; the script waits up to 6 h for this,
   re-checking idle.
5. It unpacks the archive and runs both manifests.

The script was tested against a fake sysfs (`QueueScriptTests`):

* both manifests run after Stage 1;
* a Stage 1 exit without BOARD-QUEUE-DONE still needs B40;
* another image (`428c00e5…`) or a 245 MHz clock stops it with nothing run;
* low host memory stops it.

**Launch command.** See the final report, or run it as below with `C` set to
the 12-hex commit to run. It copies the archive and the script through GB10
and detaches the queue. It does not touch the card until the gates pass.

```sh
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
C=<commit>
git -C /home/ubuntu/TritonChip-hbm8 archive --format=tar.gz -o $S/r41hbm8_$C.tar.gz $C
rsync -e $S/gb_rsh $S/r41hbm8_$C.tar.gz /home/ubuntu/TritonChip-hbm8/evidence/r41/hbm8/launches/hbm8_queue.sh gb10:
$S/gb "scp -o BatchMode=yes r41hbm8_$C.tar.gz hbm8_queue.sh zzhouch@acf3019.ece.ust.hk: && ssh -o BatchMode=yes zzhouch@acf3019.ece.ust.hk 'setsid -f bash ~/hbm8_queue.sh $C > ~/r41hbm8_queue_$C.out 2>&1 < /dev/null'"
# progress: $S/fp "tail ~/r41hbm8_board_$C/queue.log"
```

## 5. Changes in this branch

* **No change to `python/tnca`, `rtl/`, `compiler/` or `xls/`.** No new
  image is required, so there is nothing for Stage 2 to add for 8 GiB. A
  Stage 2 image must keep `fpga/u280_multicore_hbm.cfg`
  (`hbm{0,1,2}:HBM[0:31]`).
* **Test benches (not in the product tree):**
  * `tb/common/tnca_axi_line_model.v` gains `WIN_BASE` and `SPARSE`; the
    defaults keep the dense 0x1_0000_0000 model;
  * `tb/fpga/tnca_multicore_board_regs.v` passes them through;
  * `tb/fpga/tnca_multicore_dma.vh` locates DMA lines through the same map;
  * `tb/fpga/tnca_multicore_regs.v` declares dense-only;
  * new: `tb/common/tnca_sparse_map.vh` and `tb/fpga/tb_line_axi_hbm8.v`.
  * Existing benches that use these files: REGRESSION.
* **Tools:** `tools/performance_harness/hbm8_{case,sim,manifest,summary}.py`
  and `test_hbm8.py`.
* **Spec:** `spec/r41/hbm8/experiments/board-hbm8{,-big}.json`. Adding them
  changes the `software_tree` digest, which covers `spec/`. The board record
  carries its own tree.

## 6. Parallel agents

PARALLEL

## Reproduce

```sh
# unit RTL (GB10, iverilog 12): image and frozen tnca_line_axi
iverilog -g2012 -o t.vvp tb/fpga/tb_line_axi_hbm8.v <rtl>/fpga/tnca_line_axi.v && vvp t.vvp
# sparse 8 GiB system simulation, one case (image RTL = git archive 54249f7 rtl)
python tools/performance_harness/hbm8_case.py --sim --case samples --rtl-root <54249f7 tree> [--hbm-base 0x100000000] --out DIR
# runtime/backend/manifest/queue checks (no card, no torch)
python3 tools/performance_harness/test_hbm8.py
python3 tools/performance_harness/hbm8_manifest.py --check
```
