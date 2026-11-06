# DCO on TNCA: Dynamic Cache Orchestration in the NUCA slices

This is the TNCA implementation of *DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management* ([`DCO.pdf`](DCO.pdf)). It covers the hardware (TMU and the slice's replacement, bypass and dead-block logic), the compiler pass that feeds it, and the measurements on the 2×2 system.

## Contents

1. [What DCO adds](#1-what-dco-adds)
2. [Hardware](#2-hardware)
3. [Compiler](#3-compiler)
4. [Measurements](#4-measurements)
5. [Verification](#5-verification)
6. [Differences from the paper](#6-differences-from-the-paper)
7. [How to run](#7-how-to-run)
8. [Next steps](#8-next-steps)

## 1. What DCO adds

The NUCA slice used to be invalid-first, round-robin, write-allocate with no policy ([`architecture_overview.md`](architecture_overview.md) §2). DCO adds anti-thrashing replacement alongside four optional mechanisms. Each optional mechanism is enabled by its own bit of the TMU's POLICY register, so each can be measured on its own:

| Mechanism | Paper | What it does here |
|---|---|---|
| Anti-thrashing (AT) | §IV-C | A victim comes from the lowest priority tier in the set. Priority is the tag's low `B_BITS` bits. LRU breaks ties. Only lines of registered tensors that are in the older half of the set's LRU stack are ranked; see §6 for why. |
| Dead-block prediction (DBP) | §IV-B | Software registers nAcc per tensor. The TMU counts reads of each tile, and a tile whose count reaches nAcc goes into a 16-entry dead FIFO. Dead lines are evicted first. |
| Dynamic bypass | §IV-D | A read miss of a registered tensor whose priority is below `B_GEAR` is not allocated. Each slice adapts `B_GEAR` to its eviction rate per window, between `bypass_lb` and `bypass_ub`. |
| `gqa_bypass` | §IV-E | Dynamic bypass applies only to requests from the slower core of a pair, the one with fewer committed instructions. |
| Tensor bypass | §IV-A | A tensor registered with nAcc = 1 is never allocated. |

Plain LRU is also available as a policy (`[1:0] = 1`), because the paper's results are stated against LRU.

None of these changes what a load returns. They decide only where a line lives. The slice's randomised checker runs under every policy and checks every read against a shadow memory (§5).

## 2. Hardware

### 2.1 Tensor Management Unit (`rtl/nuca/tnca_tmu.v`)

There is one TMU for the machine, instantiated in `rtl/tnca_top.v`. It holds:

- **Tensor table:** 8 entries of {base, size, nAcc, bypass, operand id, tile size}. An address belongs to the entry with the highest base not above it, within `size` (size 0 means "up to the next tensor").
- **Live tiles:** 256 direct-mapped counters, keyed by tile id = {tensor index, `addr >> lsb`}, where `2^lsb` is the tensor's own tile size. Tiles of different tensors never share an id, whatever their sizes.
- **Dead FIFO:** 16 tile ids. Every slice consults it when choosing a victim.
- **Policy registers** (below), shared by all slices.

Software reaches the TMU with ordinary scalar stores and loads to a 4 KiB window at `TMU_BASE = 0x7FFF0000`. The window is never cached. The home slice of the addressed line forwards the access to the TMU (`E_MMIO` state in the slice). A registration is three stores to one staging line. The TMU keeps one staging line per requester, so concurrent registrations from different cores do not mix. A line has one home slice, and the slice serves one line's requests in arrival order, so the three stores arrive in order.

| Offset | Register | Fields |
|---|---|---|
| 0x000 | POLICY | `[1:0]` 0 round-robin, 1 LRU, 2 AT; `[2]` DBP; `[3]` dynamic bypass; `[4]` gqa_bypass; `[5]` tensor bypass; `[6]` static gear |
| 0x008 | PARAMS | `[3:0]` B_BITS (≤ 3); `[27:24]` GEAR_INIT |
| 0x010 | GEAR | `[15:0]` window (cycles); `[31:16]` bypass_ub; `[47:32]` bypass_lb (evictions per window) |
| 0x018 | CLEAR | any write forgets every tensor, live tile and dead tile |
| 0x020–0x038 | STAT | registrations, tile-last-word updates, tiles declared dead, notices dropped |
| 0x040 | TILE | `[5:0]` D_LSB: the default tile granule, for a tensor registered without one |
| 0x100 + 64·k | staging | +0 BASE, +8 SIZE, +16 DESC (`[31:0]` nAcc, `[32]` bypass, `[39:36]` operand, `[45:40]` tile lsb); the write to DESC commits |

### 2.2 The slice (`rtl/nuca/tnca_nuca_slice.v`)

- **Set index.** The slice now indexes sets with the line-address bits *above* the two home bits. Before R38 it used the low line bits. Every line that reaches a slice has the same home bits (`line[1:0] ^ line[3:2]` = slice id), so three quarters of each slice's sets were unreachable. This fix stands on its own and also changes the baseline (§4.3).
- **Victim selection.** Invalid ways come first, as before. Under round-robin the old pointer is kept. Otherwise the victim is the way with the smallest key {not dead, priority, recency}, where priority is `tag & (2^B_BITS − 1)` under AT. A way is dead if it lies in a registered tensor and either that tensor is bypass-registered or the line's tile id (formed as the TMU forms it) is in the dead FIFO. Priority is the tag's low bits only for lines of registered tensors in the older half of the set's LRU stack. Every other line counts as the highest tier, so recency decides among them (§6).
- **LRU** is kept as per-set ages, a permutation updated on every hit and fill.
- **Bypass.** A read miss that bypasses still gets an MSHR, but the MSHR claims no way. The fill writes no array, and the targets are served from the MSHR's line buffer. A store or atomic that finds a bypassed miss to its line waits for that miss to retire, then allocates normally. Stores and atomics never bypass.
- **Gear.** The slice counts valid-line evictions per `window` cycles. Above `ub` the gear rises (bypass more); below `lb` it falls; the range is 0 … 2^B_BITS.
- **Tile-last-word notices.** When DBP is on, a read of the last 8-byte word of a tile of a registered tensor, in that tensor's granule, sends the address to the TMU through a one-entry buffer. A notice that finds the buffer full is dropped and counted (`STAT_DROP`). Dropping a notice only delays the moment a tile is declared dead.
- **Counters:** `perf_bypass`, `perf_evict`, `perf_dead_evict`, `perf_gear`.

The slice also had a latent bug, fixed here: the allocate path never set `tg_amo/aop/cmp` on the target it created. An atomic that missed in the cache could therefore be served as a plain access (or a plain access as an atomic) whenever the target slot had last held something else.

### 2.3 System wiring

- `tnca_tile` passes the DCO ports through to the slice.
- `tnca_top` instantiates the TMU and packs the policy fields into one 76-bit `dco_cfg` bus. It derives the gqa slow-core mask from `perf_issued` of tile pairs {0,1} and {2,3}.
- The DRAM model gained `INTERVAL`, at most one request per INTERVAL cycles. A policy that saves traffic saves time only when traffic costs time.
- `tb_tnca_top` gained:
  - `+data=` to preload tensors that do not fit in the program image;
  - `+dco_*=` plusargs, which stand in for the host's policy writes;
  - a summary line with DCO and DRAM statistics.

## 3. Compiler

`compiler/python/tnca_cc/dco.py`, enabled with `TTIRImporter.run(..., dco={'grid': (x, y, z)})`.

**What it computes.** The paper defines nAcc as the product of the trip counts of the loops, the grid among them, that do not appear in a tile's index map. The pass gets the same number by counting directly. It interprets the kernel's integer and address arithmetic for every program of the grid and every loop iteration, with nothing but Python integers. It then counts how many times each tile origin of each pointer argument is loaded. This agrees with the product rule on the loop nests that rule covers. Counting also handles what the rule does not: a flattened tile index split with `t // N` and `t % N`, or a loop strided by the program id (`range(pid, T, npid)`). Both appear in the benchmark kernels.

**What it emits.** At the top of the kernel, before any load, it emits scalar stores to the TMU window:

- one TILE write, the default granule;
- three stores per input tensor: BASE, SIZE (the extent the trace touched), and DESC. DESC carries nAcc, bypass, and the tensor's tile granule: the largest power of two within one contiguous run of its tile, at least one line. A contiguous 16×16 f16 tile is counted whole (512 B). A tile of 32-byte rows is counted per line, and a line's last word is read exactly once per pass over it.

**Bypass.** The compiler marks a tensor for bypass when nAcc = 1 *and* its tiles cover whole lines. A tile narrower than a line shares that line with its neighbour: in f16, `X[m, kb*16:(kb+1)*16]` is half a line. Such a tile is read once, but its line is read twice, and bypassing it would fetch the line twice. DBP still sees the line die after its last word is read.

**When it refuses.** It registers nothing if:

- a loop bound depends on loaded data;
- a load address cannot be computed;
- a branch depends on an unknown value;
- the trace exceeds 4M operations.

The kernel is still compiled, and `importer.dco_note` says why. Registration is a hint, so leaving it out gives the baseline, never a wrong answer.

**Scratch.** Kernels compiled for literal arguments can now be given `scratch_stride`, so that programs running at the same time (one per tile) build their TensorMaps and repacks in separate areas.

## 4. Measurements

All runs use the 2×2 system with:

- **Last level:** 8 sets × 4 ways × 64 B per slice, 8 KiB in total.
- **DRAM:** 40 cycles of latency, and one line every `DRAM_INTERVAL` cycles.

**The kernels.** Three Triton kernels move their 16×16 tiles with tensor descriptors and accumulate with `acc = tl.dot(a, b, acc)` (§7). Each kernel runs four programs, one per tile.

**Correctness, every run:**

- Every policy produced bit-identical output.
- The first run matched NumPy.
- The registration stores cost nothing measurable. RR with and without them differs by under 0.1 %.

### 4.1 Compute-bound: DRAM_INTERVAL = 8, first version of AT

**xw** (X 384×128 streamed through a 4 KiB W, reused 24×):

| Policy | Cycles | vs RR | DRAM reads | Dead evictions |
|---|---:|---:|---:|---:|
| RR | 133 762 | 1.00× | 4 248 | 0 |
| LRU | 134 073 | 1.00× | 4 260 | 0 |
| AT | 131 109 | 1.02× | 4 255 | 0 |
| AT + DBP | 129 005 | 1.04× | 3 994 | 770 |
| all | 129 005 | 1.04× | 3 994 | 770 |

**gemm** (B = 16 KiB, twice the cache): every policy is within 1 %. AT reads more (3 858 lines vs RR's 3 424).

**attn** (two heads; per head, K + V = 16 KiB):

| Policy | Cycles | vs RR | DRAM reads | Hits |
|---|---:|---:|---:|---:|
| RR | 400 449 | 1.00× | 3 816 | 123 368 |
| LRU | 400 648 | 1.00× | 3 828 | 123 271 |
| AT | 533 970 | **0.75×** | 15 253 | 50 066 |
| all | 504 996 | 0.79× | 12 959 | 64 939 |

What this run shows:

1. **These kernels are compute-bound at this bandwidth.** xw's DRAM traffic occupies about a quarter of its runtime. A policy can save lines, but it cannot turn them into time.
2. **The first AT was harmful when four cores share a 4-way set.** All four cores stream the same K/V tile of a head at nearly the same moment. A line in a low priority tier was the next victim as soon as it landed, before the other cores had read it, so each core fetched it again. The scratch buffers that `s.to(float16)` repacks through every iteration are not tensors either, and they were ranked by their address bits like everything else. Both effects are visible as the 4× rise in DRAM reads. The paper's AT relies on deprioritised data "staying in the cache for a while", which a 16-way, megabyte LLC provides and a 4-way set of 8 KiB does not.
3. **The dynamic gear never moved.** These kernels evict a few lines per slice per window, below the first threshold (24 per 512 cycles). The bypass count is Q alone, through tensor bypass (nAcc = 1).

**The fix:** AT ranks only lines of registered tensors in the older half of the LRU stack, and dynamic bypass applies only to registered tensors (§6). The gear thresholds were also lowered (4 and 1 per 1 024 cycles).

### 4.2 Memory-bound: DRAM_INTERVAL = 48, AT with young-line protection

One line every 48 cycles, so traffic now costs time. AT ranks only lines of registered tensors in the older half of the LRU stack. The gear thresholds are 4 and 1 evictions per 1 024 cycles, and `at+fix2` is the paper's static variant with B_GEAR = 2. Every policy's output was bit-identical.

**xw**, X streamed through a W that is reused 24×. This is DCO's case:

| Policy | Cycles | vs RR | vs LRU | DRAM reads | Hits | Bypassed | Dead evictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| RR | 291 666 | 1.00× | 1.00× | 4 242 | 11 053 | 0 | 0 |
| LRU | 292 637 | 1.00× | 1.00× | 4 278 | 10 548 | 0 | 0 |
| AT | 289 344 | 1.01× | 1.01× | 4 194 | 11 027 | 0 | 0 |
| AT + DBP | 274 377 | 1.06× | 1.07× | 3 991 | 12 049 | 0 | 900 |
| AT + dynamic bypass | 279 798 | 1.04× | 1.05× | 4 377 | 12 476 | 1 313 | 0 |
| AT + static gear 2 | 293 579 | 0.99× | 1.00× | 4 285 | 11 223 | 871 | 0 |
| **all** (AT + DBP + bypass) | **264 228** | **1.10×** | **1.11×** | 4 225 | 12 962 | 1 366 | 539 |
| all + gqa_bypass | 271 333 | 1.07× | 1.08× | 3 969 | 12 520 | 397 | 742 |

- **DBP:** X's lines die after their last word is read, so they become the first victims and W stays. DRAM reads fall 6 %, and the 1 536 tiles declared dead are exactly X's 1 536 lines (its granule is one line).
- **Bypass:** a bypassed X line takes no way and evicts nothing. The combination is worth more than either alone: 1.10× over RR.

**gemm**, all of B reused by every row of blocks, with B twice the cache:

| Policy | Cycles | vs RR | vs LRU | DRAM reads | Bypassed |
|---|---:|---:|---:|---:|---:|
| RR | 227 696 | 1.00× | 1.03× | 3 341 | 0 |
| LRU | 235 243 | 0.97× | 1.00× | 3 445 | 0 |
| AT | 240 978 | 0.94× | 0.98× | 3 397 | 0 |
| AT + DBP | 236 578 | 0.96× | 0.99× | 3 378 | 0 |
| AT + dynamic bypass | 273 112 | 0.83× | 0.86× | 4 136 | 1 158 |
| AT + static gear 2 | 301 335 | 0.76× | 0.78× | 4 777 | 2 096 |
| all | 264 696 | 0.86× | 0.89× | 3 992 | 1 240 |
| **all + gqa_bypass** | **233 771** | 0.97× | **1.01×** | 3 340 | 136 |

DCO gains nothing here, and priority-based bypass is harmful. In this kernel the four cores work on neighbouring blocks of one row, so a B line one core fetches is soon read by another. Bypassing it by priority throws that reuse away, and the static gear 2 variant reads 43 % more. This is the inter-core sharing case of the paper's §IV-E. `gqa_bypass`, which bypasses only for the slower core of a pair, keeps the sharing and brings the full policy back to LRU's level. That matches the paper's recommendation to use gqa_bypass under spatial sharing.

**attn**: two heads, each with 8 Q blocks; every Q block streams that head's K and V (16 KiB, twice the cache). Four cores work on one head at a time.

| Policy | Cycles | vs RR | vs LRU | DRAM reads | DRAM writes | Dead evictions |
|---|---:|---:|---:|---:|---:|---:|
| RR | 546 531 | 1.00× | 0.98× | 3 772 | 2 428 | 0 |
| LRU | 535 531 | 1.02× | 1.00× | 3 521 | 2 181 | 0 |
| AT | 529 462 | 1.03× | 1.01× | 3 488 | 2 097 | 0 |
| **AT + DBP** | **491 841** | **1.11×** | **1.09×** | 3 050 | 1 662 | 640 |
| AT + static gear 2 | 552 585 | 0.99× | 0.97× | 4 336 | 1 948 | 0 |
| all | 496 441 | 1.10× | 1.08× | 3 178 | 1 620 | 506 |

- **AT:** with young lines protected, AT is no longer harmful (1.03×, against 0.75× in §4.1).
- **DBP is the larger lever.** After a head's eight Q blocks, its K and V tiles reach nAcc = 8 and go into the dead FIFO, and the next head's tiles replace them instead of the cache keeping a stale half. DRAM reads fall 19 % and writes 32 %, for 1.11× over RR.
- **Bypass:** a static gear costs reads here as in gemm, for the same reason. The dynamic gear inside `all` stays close to AT + DBP.

### 4.3 What the measurements say

| Workload | Best policy | vs RR | vs LRU | Why |
|---|---|---:|---:|---|
| xw (stream through a reused weight) | all | 1.10× | 1.11× | DBP retires X's lines; bypass keeps them out |
| attn (two heads, K/V twice the cache) | AT + DBP | 1.11× | 1.09× | the finished head's K/V is dead, not "recently used" |
| gemm (B shared by four cores) | all + gqa | 0.97× | 1.01× | sharing: bypass must be gqa or it costs reads |

- **Gains need memory to be the bottleneck.** At DRAM_INTERVAL = 8, the same kernels gain at most 4 %.
- **The size of the gain fits the scale.** The paper reports up to 1.9× with megabyte LLCs and long sequences. Here a 4-way, 8 KiB last level and kernels that fit an hour of RTL simulation give 1.10–1.11× on the two cases DCO targets, and nothing on the one it does not.
- **No policy changed a single output bit**, across about 60 system runs.

## 5. Verification

- **`make nuca`.** The slice's randomised differential test, under the round-robin baseline, on the fixed set index. Its addresses are now home-consistent: a slice sees only lines it is home to, as on the mesh.
- **`make nuca-dco`.** The same test under POLICY 1, 2, 46, 62 and 74 with randomised DCO inputs:
  - a bypass-registered region;
  - a dead FIFO that is rewritten at random;
  - a flipping slow-core flag;
  - a random tile-notice back-pressure.

  Every read is checked against the shadow. All 27 runs (9 policies × 3 seeds) passed during development.
- **`make top` and `make e2e`.** The 2×2 system with the TMU in place and POLICY 0.
- **`make dco-compiler` (`compiler/test/test_dco.py`).** The pass's plans for the three benchmark kernels; that registration precedes every load; that the pass refuses a data-dependent loop.
- **`make dco-bench`.** Every policy must produce bit-identical results, and the first run is also checked against NumPy.

## 6. Differences from the paper

- **Tile identity** is {tensor, aligned chunk in that tensor's granule}, not a (tensor, tile-coordinate) pair. A tile that is one contiguous block, like the benchmarks' K/V and weight tiles, is exactly one chunk. A tile whose rows are not contiguous is one chunk per row (or per line), and each chunk is still read nAcc times.
- **The tile-last access** is the last 8-byte word, not the last line, because a slice request here is a word.
- **The paper's `set` instruction** is a store to the TMU window, so no ISA change was needed.
- **The live table** is direct-mapped and the dead FIFO has 16 entries. A collision or overflow loses a hint, never data.
- **AT protects young lines and ignores non-tensor data.** The paper ranks every line by its priority tier. Here, only lines of registered tensors are ranked, and only those in the older half of their set's LRU stack. A just-filled line therefore survives until the cores sharing it have read it, and scratch data is ranked by recency alone. Without this, a 4-way set evicted a low-tier line as soon as it was filled (§4.1).
- **Dynamic bypass only for registered tensors**, for the same reason.
- **Scale.** An 8 KiB last level against the paper's 1–8 MB, and kernels sized so that RTL simulation (about 300 cycles per second) finishes within an hour. What carries over is the *ratio* of working set to cache, not absolute sizes, and the 4-way sets are what made the young-line protection necessary.

## 7. How to run

```
make nuca-dco                                  # slice correctness under the policies
make dco-bench TRITON_PYTHON=...               # memory-bound (DRAM_INTERVAL 48)
make dco-bench DCO_DRAM_INTERVAL=8 TRITON_PYTHON=...   # compute-bound
python tools/r38/dco_bench.py --only attn --policies rr,lru,all --dram-interval 48
```

- **Memory:** `DCO_DRAM_INTERVAL` (default 48) selects `sim/build/tb_top_dco_i<n>.vvp`, and `DCO_DRAM_LAT` (40) sets the latency.
- **Output:** the bench writes `sim/build/r38/dco_bench_i<n>.json`.
- **Run time:** the four-tile RTL simulates at about 300 cycles a second. xw and gemm take about 25 minutes per policy, attn about an hour.

## 8. Next steps

1. **Scale.** A faster system simulator (or the board, once the 2×2 system is on it) would allow 16-way sets and megabyte LLCs. That is where the paper's AT works without the young-line protection and its bypass numbers apply.
2. **The gear.** The thresholds were set by hand for these kernels. The paper adapts them; a sweep (`--policies at+bypass` against window, ub and lb) would show how sensitive the result is.
3. **Launches through the runtime** ([`compiler_overview.md`](compiler_overview.md) §15.6): compute nAcc from launch arguments, and carry the TMU writes in the artifact.
4. **CLEAR between kernels.** Today the host clears the TMU at reset. A runtime that launches back-to-back kernels must write CLEAR, or stale dead tiles of the previous kernel steer the next one.
