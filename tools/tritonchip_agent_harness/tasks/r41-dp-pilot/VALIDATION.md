# r41-dp-pilot: what was validated before use

The task scores an agent's changes on R41's three-core machine
(`spec/devices/u280-3core-r41.json`, per-core profile
`core-vrf-sync-v5-x8-g2c4-dp-board`), simulated by the evaluator in device
containers from the candidate's own description, as `r40-multicore-pilot`
does. Base: the author's merge-freeze commit `ef273883` plus the harness's
adaptation commits (`e9d46089`, `f206c033`, `089ba6db`, `b476f3fb` on the
local branch `r41-harness-start`).

## The machine is the image's

`python -m tnca.devices params u280-3core-r41` reproduces every one of the
60 parameters of the corrected R41 candidate's `image.tcl`
(`evidence/r41/hbm_bw/g4/run2/image.tcl`, image 791b8af2, source 8d237223),
with no key missing, extra or differing. R40's description still generates
its 41. The profile variant states the three core-level values the image
set beyond the author's `-dp` profile (LSU_INFLIGHT 1, FMA_IMPL 1,
MTX_VRF_REG 1); the author's profile is untouched.

## The striped window (2026-11-06)

The first noop run failed every point before its first launch: the
runtime's striped-HBM layout requires a base-zero 8 GiB window, and the
board-top stand-in offered 16 MiB at 4 GiB. The stand-in now simulates that
window sparsely (`python/tnca/backends/multicore_mmio_sim.py`, after the
author's `tools/performance_harness/hbm8_sim.py`): it backs the runtime's
code region, the memory budget as heap and the device's reserved range,
scattered over the 32 PCs as the image's address path scatters them, and
host DMA goes through `StripedHBMLayout`. The second run failed because the
runtime asked for 1 GiB; `stand_in` now opens the described machine's HBM.

## Known-answer submissions

**noop** (third run, 2026-11-06, run
`r41-dp-pilot-noop-20261106T063718Z-28a6`): 8 of 8 points pass, validation
and final `speedup_geomean` 1.0, `fast_1` 0.0, run valid. The observer's
count exceeds the shell's by 6 cycles on every point, the handshake the
protocol describes (`archbench/PROTOCOL.md` §3.1).

| Point | Set | Observed cycles | Wall per point |
|---|---|---|---|
| op.add/tiny/fp16-n1000 | validation | 3015 | 57 s |
| op.layernorm/tiny/fp16-r4-d64 | validation | 3855 | 79 s |
| op.linear/tiny/fp16-m16-k64-n64 | validation | 6378 | 139 s |
| op.scaled_mm/tiny/e4m3-to-bf16-m16-k64-n64 | validation | 3947 | 115 s |
| op.transpose/tiny/fp32-r32-c32 | final | 2453 | 99 s |
| op.softmax/tiny/fp16-r4-c64 | final | 2873 | 77 s |
| op.rope/tiny/fp16-b1-h2-s16-d16 | final | 6825 | 115 s |
| op.w8a8_linear/tiny/e4m3-to-bf16-m16-k64-n64 | final | 8337 | 156 s |

The two FP8 points (registered 2026-11-06, `archbench/ops.py`) run through
the same chain as every other point and match torch's CPU `_scaled_mm`
exactly on this machine (`err=0.0`); their tolerance is the output dtype's.
The eight device containers elaborate and come up in under a minute; a
whole evaluation takes about three minutes on the evaluation host.

**known/lsu-inflight-2.patch** (the second load/store unit back, which the
author gave up for the SLR budget; run
`r41-dp-pilot-patch...-20261106T064449Z`): the gate takes
`spec/r1_profiles.json` only, `machine.changed` is `LSU_INFLIGHT: 1 -> 2`,
8 of 8 points pass, validation `speedup_geomean` 1.114 and final 1.228,
`fast_1` 1.0 on both, run valid, **certification pending**. This is the
cheap win the card has to refuse: in simulation every point gets faster
(rope 1.40x, transpose 1.29x, add 1.24x), and the score stays pending until
the slow tier has built and run it on the U280, where the second unit costs
about 36k LUTs the base machine does not have.

## Not validated here

- Certification on the card of a hardware-changing candidate: the slow
  tier's board stage for a described machine is not built; a candidate that
  changes the machine keeps `certification: pending`.
- The quick tier IS validated on the author's shared x86 Vivado machine
  (worker `x86vivado`, Vivado 2023.2, `unshare` sandbox): the out-of-context
  synthesis of the `-dp-board` core at 10 ns finished in 318 s on
  2026-11-06 with WNS +3.785 ns, 0 failing endpoints, 160.9 MHz estimated,
  211,788 LUTs, 61,278 FFs, 167 BRAMs, 173 DSPs. The task offers 45 minutes
  of it, 30 per request.
- The FP8 bit-exact suite IS run once on the base as a chain check
  (2026-11-06, `fp8_cases.py --quick --group scaled_mm --group linear`,
  backend `fpga-multicore-sim` on `u280-3core-r41`, profile `-dp-board`,
  in the toolchain container on the evaluation host): 8 of 8 cases pass
  with 0 bit mismatches against the per-K and the `m.mma.dp` references
  (E4M3 and E5M2 scaled_mm, tensorwise and rowwise, bf16/f16/f32 out,
  mixed E4M3 x E5M2, static W8A8 and weight-only FP8 linears at two
  shapes); every FP8 GEMM graph executes `m.mma.dp.async`. The e5m2
  rowwise f32 case differs from torch CPU's own `_scaled_mm` by up to 97
  ulp, as the author's records say it does (the reference order differs);
  it matches the scale-first emulation exactly. The full suite, casts and
  plumbing included, is a campaign-end check on the board, not part of
  the score.

## Known properties of the start machine (from the r41 branch review, 2026-11-06)

- With `HYBRID_LSU` 1 and `HOME_LINE_PIPE` 2 the tile instantiates
  `tnca_home_pipe` (`rtl/tile/tnca_tile.v`, `HOME_PIPE_ON`), whose parameter
  list has none of `NUCA_SHARED_DATA_READ`, `NUCA_CACHE_DATA_PIPELINE`,
  `NUCA_REGISTER_VICTIM_RANK`, `NUCA_REGISTER_VICTIM_SELECT`,
  `HOME_BACKGROUND_READS`, `DCO_REGISTERED_RANGES`; the description states
  them as the image did, and they are dead for this machine. The task text
  says so.
- The simulated top elaborates the board machine: `VALU_LATE` and
  `TENSOR_MUL_PIPE` take their board defaults in the three-core top; only the
  single-core legacy profiles pin `VALU_LATE` 0.
- Icarus and Vivado disagree on untyped parameter arithmetic (the run-2
  image's `MAC_LAT` synthesised to 0 before the `integer` fix now in the
  base). The quick tier's Vivado elaboration is the only check of that class
  on a candidate's RTL; the simulation cannot see it.
- `flow/fpga/link.sh` refuses any automatic frequency scaling other than
  rule 08 (exit 3, `LINK_FAILED freq-moved`); the base image itself scales
  `hbm_aclk`, so certification needs the declared platform exception (D4).

## The base image (re-linked 2026-11-06)

Linked from the base `b476f3fb` on the author's shared x86 Vivado machine
(Vitis 2023.2) through the FPGA harness pipeline (run
`20261106T072410Z-45ff9131`, target `u280-multicore-r41`, image.device
`u280-3core-r41`, about 7 h): `tnca_u280.xclbin` sha256
`7aa3e8500b6c7fbaa3186bdf3205089c69cfc352d08666233750f320d9b4d361`, UUID
`d6bd7770-9a0f-c5ba-77fe-f0ad9cee61e6`, DATA_CLK 100 MHz. The kernel clock
closes: WNS +0.111 ns, 0 of 1,145,149 endpoints failing, WHS +0.001 ns. The
platform's `hbm_aclk` auto-scaled 450 to 433 MHz with 22 failing setup
endpoints (WNS -0.085 ns, TNS -0.856 ns), so `link.sh` reports `LINK_FAILED
freq-moved` and the harness records TIMING_FAIL: these are the same figures
as the author's corrected candidate (791b8af2), which the author accepted by
waiver. The harness's certification needs the declared platform exception
(D4) to call such an image feasible; the pilot records it as this section.

## Board pilot (2026-11-06, smoke mode; driver `board/board_pilot.py`)

The image was programmed on the shared U280 after 902 s of continuous idle
(previous card image UUID 0bc441ad-2388-f291-e945-5dc6b994d4a1, a foreign
300 MHz design, recorded), `program.sh --check` and `--go` both 0, three
counter samples `[100, 500, 430, 0]`: kernel 100 MHz, HBM counter 430 as in
the author's capture. The chat tooling's four auto-enabled paths (EOS
guard, early stop, hybrid prefix checkpoint, static prefix rewind) were
pinned to 0, the campaign's loop; CPU references regenerated on acf3019
under the author rule with first-token references.

**Qwen3-VL-2B-FP8, registered prompt `fact` (reduced proof):** pass under
`author-20261101` (content, first token equal to the CPU's, containment),
warm repeat pass with bitwise-equal logits, scripted chat 3 of 3 turns
pass, no CPU numerical fallback, every parameter and buffer on `tnca:0`.

| | this base image | author's campaign |
|---|---|---|
| cold prompt to return | 854.7 s | 980.8 s |
| compile and warm-up | 557.9 s | 571.3 s |
| decode, 11 steps | 229.5 s (20.9 s per step) | 242.7 s |
| cold non-special tokens/s | 0.00117 | 0.00102 |
| warm repeat | 336.6 s, 0.00297 tokens/s | 358.5 s, 0.00279 |

The same loop (12 executed steps, post-EOS work counted) 6 to 13 percent
faster: ef273883's runtime changes, not a measurement difference.
