# Line-wide, multi-command tensor transfer engine: TLM first

**Domain: model (SystemC/TLM, uncalibrated).** Nothing here is RTL or board. Every point passed its numerical check. Medium registered points, three cores; B/cycle/core = ArchBench demand bytes / device cycles / 3.

## What was modelled (default-off keys)
`tools/performance_harness/systemc/include/tnca_tensor_engine.hpp` and `tnca_timeline_memory.hpp`:

| Key | Meaning |
|---|---|
| `dma_line_engine=1` | The tensor engine moves whole 64-byte lines on the core's wide port, one line request a cycle. Each line in flight holds an engine **line buffer**, not a vector register: a load's buffer is held from its request to its register write, and a store's from its register read to its request. A command's decode overlaps the previous command's issue. The controller is held only for a command's issue phase, so commands overlap; a store's window read lease ends when its registers have been read. The descriptor snapshot is four line reads in flight together. The window port has a write side (loads) and a read side (stores), one 64-byte register a cycle each. |
| `dma_line_buffers` | line buffers per core (default 64) |
| `dma_line_combine=1` | a store's partial lines wait in their buffers for a following queued store that completes them. Only a full line is written without a fill. |
| `tl_full_line_nofill=1` | a full-line write that misses allocates without a memory read (`HOME_LINE_WRITE_NOFILL`); applies to the LSU's wide stores too |

`dma_queue` (commands admitted per core) is 8 in these runs.

Compiler: `TNCA_TILED_POINTWISE=3`, which is NON-DEFAULT, in `compiler/python/tnca_cc/pointwise_tiles.py`:
- It streams tensor-engine tiles. Every tile of the program is unrolled, and each transfer has a ring of R windows (R·transfers·tile ≤ 48 registers).
- Tile i waits only for its own loads, and for the store of tile i−R before refilling that store window. The loads of tile i+R are issued as soon as tile i has been computed.
- Mode 3 applies only when every load and store of the program can use the tensor engine (4-byte elements, contiguous). Any other tileable program gets mode 2's lowering.
- Inductor gives mode 3 a 2048-lane block, with mode 2's 1024 as the next candidate.
- It is correct on the existing RTL: `compiler/test/test_pointwise_tiles.py` now has 9 tests, passing with the core bench.

## Memory configurations
`base` is G4's `onelane` configuration (the §6 path as far as the model has keys; `g4/model_onelane_*`). The home engine's occupancies stand for the pipelined home and bracket it:
- `pipe`: miss 1, fill 1, write-back 1, request 0, wake 0, so 2 cycles per read miss;
- `pipe1`: miss 1, every other occupancy 0, so 1 cycle per request; this is optimistic;
- `ideal`: G4's `ideal` configuration (minimum latencies, 8 lanes, 32 outlets), which isolates the core and the engine.

## Results (medium points, model)

| Point | `pipe`: mode 2 + no-fill | `pipe`: mode 3 + line engine | gain | `pipe1`: mode 2 | `pipe1`: mode 3 + line engine | gain | `ideal`: mode 2 | `ideal`: mode 3 + line engine | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy fp32 | 145,485 (19.2) | 144,990 (19.3) | 1.00 | 124,366 (22.5) | 97,713 (28.6) | 1.27 | 95,595 (29.3) | 71,527 (39.1) | 1.34 |
| clone fp32 | 142,963 (19.6) | 144,745 (19.3) | 0.99 | 124,420 (22.5) | 97,749 (28.6) | 1.27 | | | |
| add fp32 | 249,529 (16.8) | 208,331 (20.1) | 1.20 | 236,921 (17.7) | 150,030 (28.0) | 1.58 | 165,109 (25.4) | 110,353 (38.0) | 1.50 |
| relu fp32 | 257,456 (10.9) | 153,716 (18.2) | 1.67 | 249,465 (11.2) | 141,444 (19.8) | 1.76 | 149,526 (18.7) | 116,111 (24.1) | 1.29 |
| add fp32, strided | 460,212 | 436,934 | 1.05 | 448,499 | 422,439 | 1.06 | | | |
| add fp16 (mode 2 under mode 3) | 1,043,188 | 1,023,767 | 1.02 | 1,023,560 | 1,003,038 | 1.02 | | | |
| copy fp16 (mode 2 under mode 3) | 337,667 | 327,633 | 1.03 | 328,919 | 316,644 | 1.04 | | | |

Device cycles (B/cycle/core in brackets).

**The default compiler with the line engine** (`pipe`, mode 0; `mx_pipe_base_m0` against `mx_pipe_line_m0`; the second arm also has `tl_full_line_nofill`):
- copy fp32: 1,143,941 → 911,536 cycles;
- add fp32: 1,351,940 → 1,123,017;
- relu fp32: 1,189,928 → 957,965;
- strided add: 1,392,853 → 1,160,015;
- the fp16 points are unchanged;
- bmm fp16, small: 511,630 → 405,305; medium: 1,005,219 → 793,545.

No point is slower.

`dma_line_combine=1` changes nothing on these points: the engine's stores are whole lines. The fp16 half-line stores come from the vector LSU (mode 2), which this mechanism does not cover.

Line buffers in use reached 48 per core (`core_*_dma_line_buffers_peak`). That is the ring's window registers (R·transfers·tile), so the compiled code, not the 64 buffers, limits lines in flight.

## What this says
- **The line engine helps.** With mode 3, its gain over the vector path is 1.0–1.8× on the fp32 points, and it is never slower, even under the default compiler.
- **It does not reach the one-lane ceiling of 64 B/cycle/core.** The best point is 39 B/cycle/core with ideal memory and 28.6 with the optimistic home.
- **A command trace** (`TLM_DEBUG_LINE=1`, copy fp32, `ideal`, core 0; `trace_ideal_copy.txt`):
  - Inside a pair of programs (the two contexts run in step), commands issue every 8–9 cycles at 8 lines each. That is about 0.9 lines a cycle on the core's 64-byte port, which loads and stores share; this is the one-lane bound (about 58 B/cycle/core while it lasts).
  - Between pairs there is a gap of about 270 cycles: the next programs' descriptor builds, their fences and snapshots, and the first loads' latency. The gap brings the average down to 39 B/cycle/core.
- **What binds, with ideal memory:**
  - inside a program, the core's one 64-byte port (the one-lane ceiling itself);
  - between programs, the descriptor build, fence and pipeline fill (the gap above);
  - lines in flight are capped by the ring's 48 window registers.
- **With the 2-cycle home:** the home engine binds copy. copy fp32 gains nothing in `pipe`.
- **For RTL to be worth building:** the measured RTL home (§8.5) should be closer to `pipe1` than to `pipe`, and the compiled code needs less per-program overhead (longer programs, or descriptors reused across a context's programs).

Raw records:
- `runs.tgz`: each run's log, `results.json` and per-point model counters, the configurations and the scripts;
- `ideal_final.tgz`.

The `mx_*`, `gx_*` and `idf_*` runs used the final model binary (`tnca-chip`, sha256 `d19facdd…`). The mode-3 rows were rerun after the last compiler change, which keeps mode 3 to programs with every access on the tensor engine. The other runs in `runs.tgz` are earlier diagnostic steps.

**Default keys unchanged.** With none of the new keys set, the new binary reproduces the earlier binary's cycles exactly: `base`, mode 2, copy fp32 211,129 and add fp32 317,771 (`default_unchanged.tgz` against `med_base_m2` in `runs.tgz`).
