# Generated datapath: Google XLS

Part of the floating-point datapath is written in [DSLX](https://github.com/google/xls)
and compiled to Verilog by XLS rather than written by hand. This document says
which parts, why those and not others, and what the decision cost and bought.

## What is generated

| Module | From | What it does | Stages | Latency |
|---|---|---|---:|---:|
| `tnca_widen_xls` | `xls/tnca_cvt.x` | any narrow format → binary32 | comb. | 0 |
| `tnca_widen_pair_xls` | `xls/tnca_fpu.x` | two operands at once | comb. | 0 |
| `tnca_narrow_xls` | `xls/tnca_cvt.x` | binary32 → any narrow format, RNE | 3 | 2 |
| `tnca_cvt_xls` | `xls/tnca_cvt.x` | both, for any source/destination pair | 4 | 3 |
| `tnca_cvt_comb_xls` | `xls/tnca_cvt.x` | the same, unpipelined | comb. | 0 |
| `tnca_sfu_exp2_xls` | `xls/tnca_sfu.x` | `v.exp2.approx` | 7 | 6 |
| `tnca_sfu_log2_xls` | `xls/tnca_sfu.x` | `v.log2.approx` | 7 | 6 |
| `tnca_sfu_rcp_xls` | `xls/tnca_sfu.x` | `v.rcp.approx` | 7 | 6 |
| `tnca_sfu_rsqrt_xls` | `xls/tnca_sfu.x` | `v.rsqrt.approx` | 7 | 6 |
| `tnca_fma_xls` | `xls/tnca_fma.x` | binary32 FMA with subnormals (FP_LAT = 2) | 3 | 2 |
| `tnca_fma_l3_xls` | `xls/tnca_fma.x` | the same, one stage deeper (FP_LAT = 3) | 4 | 3 |

The two FMA instances are generated with `--flop_outputs=false`: N stages are
N−1 registers, and the result is combinational from the last stage, because
every consumer (the vector FP writeback, the accumulator file) registers it.
`rtl/core/tnca_fp32_fma.v` picks the instance for the core's `FP_LAT`.

The formats are binary32, bfloat16, binary16, OCP FP8 E5M2 and OCP FP8 E4M3.
FP4 is deliberately absent.

Latency is stages minus one throughout, because everything is generated with
`--flop_inputs=false`: the rising edge that accepts the input is also the edge
that fills stage 0. Each testbench *measures* its unit's latency rather than
assuming it — the numbers above are the ones that make the benches pass, and
the neighbouring values fail loudly.

## The binary32 FMA: generated since R38, without apfloat

**Until R38 the FMA was hand-written,** with two hand-placed register stages.
The reason was XLS's `apfloat` library, which flushes subnormals to zero. That
is measured, not assumed:

```
apfloat::add(1e-45, 0.0)            -> 0.0   (should be 1e-45)
apfloat::fma(+0.0, -0.0, 1e-45)     -> 0.0   (should be 1e-45)
```

It is a reasonable choice for a library aimed at ML kernels, and the library
says so by shipping `subnormals_to_zero` as a helper. It is not a choice this
ISA permits: §12.4 and §12.7 make FTZ an explicitly encoded mode and say in as
many words that flush-to-zero is never an implementation shortcut. Generating
a binary32 FMA from `apfloat` would have broken the default rounding mode, and
`rtl/core/tnca_fp32_fma.v` already handles subnormals and is checked against an
exact Fraction oracle over half a million vectors.

So the arithmetic stayed where it was verified, and XLS only did the format
work around it (`rtl/core/tnca_fpu_multi.v`: XLS widening in front of the FMA).

**What changed is where the incompatibility is taken.** It was never XLS: it
is one library. `xls/tnca_fma.x` writes the FMA in DSLX without `apfloat` --
the hand-written algorithm step for step, subnormal inputs and outputs, the
`ftz` mode, `mul_only`, `sub_c`, the NaN and signed-zero rules -- and XLS does
what it is good at, the schedule. The reason this matters is the clock: the
U280 link's worst path (60 logic levels, 23.9 ns at 37.5 MHz) ran through the
hand-placed normalise/round stage, and moving a hand-placed cut is a redesign,
while a generated one is a stage count in `tools/xls/config.json`. Two
instances are generated now, latency 2 (a drop-in: same `FP_LAT`, same cycles)
and latency 3 (for a faster board clock); a third is one more config entry.

The verification did not change: `tb_fp32_fma` and `make fma-deep` check both
instances against the same exact Fraction oracle (`tools/model/fp32.py`;
41 240 and 516 240 vectors, 0 failures), and `tb_fpu_multi` checks the
composition with the five formats. The DSLX file also carries its own tests
for the cases `apfloat` gets wrong.

## What is not generated, and why

**The SFU is the opposite case.** ISA §8.8 specifies those operations as
approximate, with a published error bound, and *requires* subnormal flushing.
There XLS's behaviour is the required behaviour, and the four functions are the
first implementation the design has had. `CAPABILITY0.SFU` read 0 precisely
because Stage 1a shipped without them; it reads 1 now, and the measurement
below is what earned that.

## Scheduling: one clock target, and what it cost

Every generated module is scheduled by XLS against the **asap7** delay model at
the same **1300 ps** target, set once in `tools/xls/config.json`. The stage
count per module is the smallest that meets it. `make xls-report` measures the
result — this table is its output:

| Module | Stages | Combinational | Pipelined | Speed-up | Floor |
|---|---:|---:|---:|---:|---:|
| `tnca_widen_xls` | 1 | 1005 ps | 1005 ps | 1.00× | 220 ps |
| `tnca_widen_pair_xls` | 1 | 1005 ps | 1005 ps | 1.00× | 220 ps |
| `tnca_narrow_xls` | 3 | 3476 ps | 1210 ps | **2.87×** | 358 ps |
| `tnca_cvt_xls` | 4 | 4516 ps | 1195 ps | **3.78×** | 358 ps |
| `tnca_sfu_exp2_xls` | 7 | 6688 ps | 1220 ps | **5.48×** | 1217 ps |
| `tnca_sfu_log2_xls` | 7 | 6664 ps | 1293 ps | **5.15×** | 1150 ps |
| `tnca_sfu_rcp_xls` | 7 | 5316 ps | 1151 ps | **4.62×** | 1150 ps |
| `tnca_sfu_rsqrt_xls` | 7 | 5263 ps | 1132 ps | **4.65×** | 1130 ps |

**Floor** is the shortest period *any* schedule of that module could reach.
XLS never splits a single IR node across stages, so the widest multiply sets
it. Read the last two columns together:

- `tnca_sfu_rcp_xls` lands at **1151 ps against a floor of 1150 ps**, and
  `rsqrt` at 1132 against 1130. Those units are at the bound set by their own
  arithmetic — there is no schedule of that DSLX that is faster, and the only
  way down is a narrower multiply, not more flops.
- `exp2` is 3 ps off its floor for the same reason. `log2` has 143 ps of slack
  left, which is the one place more stages would still buy something.
- `tnca_narrow_xls` and `tnca_cvt_xls` are nowhere near their 358 ps floor and
  are not meant to be: three and four stages reach the 1300 ps target, and
  stopping there costs fewer flops than chasing a floor nothing needs.

Two decisions came out of this table rather than out of taste.

**`widen` stays combinational.** Its whole path is 1005 ps, inside the target
with no flops at all, and that is what lets `rtl/core/tnca_fpu_multi.v` put
format widening in front of the existing binary32 FMA without adding a cycle to
the multiplier. The *narrowing* direction is the expensive one — rounding,
saturation and subnormal generation for four destination formats, 3476 ps
combinational — and that is the half that had to be pipelined. The two
directions are separate modules for exactly this reason; `tnca_cvt_xls` exists
for the ISA-level convert, not for the datapath.

**The SFU coefficient tables are 32-bit, not 64-bit.** The first version held
Q_.30 coefficients as `s64` and multiplied in 64 bits. Every coefficient has
magnitude below 2, so Q2.30 in `s32` is exact and loses nothing — the accuracy
bench reports the identical 16,692 vectors inside the bound either way. What
changed is everything else: XLS's optimiser narrows the products to 32×16 and
31×16, yosys goes from **not finishing a single module in six minutes** to 33
seconds, and the asap7 floor is set by a 32×16 multiply instead of a 64×64 one.
The wide version was correct and unshippable.

## What yosys says

`make synth` maps the same modules to generic gates with no PDK, so its numbers
are a different proxy from the asap7 periods above and should disagree with
them in detail — it has no carry-lookahead cells, so every addition is ripple
carry (see doc/synthesis.md for the calibration). It agrees on the direction,
which is the point of measuring twice:

| Module | Combinational | | | Scheduled | | |
|---|---:|---:|---:|---:|---:|---:|
| | gates | flops | depth | gates | flops | depth |
| `tnca_widen_xls` | 265 | 0 | 11 | *unchanged — stays combinational* | | |
| `tnca_widen_pair_xls` | 501 | 0 | 11 | *unchanged* | | |
| `tnca_narrow_xls` | 872 | 0 | **45** | 2,330 | 211 | **31** |
| `tnca_cvt_xls` | 1,097 | 0 | **54** | 2,848 | 253 | **29** |
| `tnca_sfu_rcp_xls` | 10,109 | 0 | **105** | 11,669 | 330 | **42** |
| `tnca_sfu_rsqrt_xls` | *yosys did not finish* | | | 20,223 | 428 | **41** |
| `tnca_sfu_exp2_xls` | *not measured* | | | 14,946 | 529 | 69 |
| `tnca_sfu_log2_xls` | *not measured* | | | 13,246 | 447 | 72 |

For scale, the calibration block in doc/synthesis.md puts a bare 32×32
multiplier at 5,377 gates and 65 levels in this library, and `tnca_fp32_fma` at
7,120 gates and 115 levels.

Three things in that table are worth saying out loud.

**`tnca_sfu_rcp_xls` costs less than its own arithmetic.** Two 32×32
multipliers alone are 10,754 gates here; the whole unit — two multiplies, a
256-entry three-coefficient table, binary32 decode, segment index, the
renormalisation and the pack — is 10,109 combinational. XLS narrowed the
products to 32×16 and 31×16 from the range it could prove, and what that buys
pays for everything around them.

**Pipelining bought depth at a fair price in flops.** `tnca_cvt_xls` went from
54 levels to 29 and `tnca_narrow_xls` from 45 to 31, for about 250 and 210
flops on a 32-bit datapath. Both now clear the project's 50-level flag, which
the combinational `tnca_cvt_xls` did not.

**`exp2` and `log2` are still over the flag, and are meant to be.** At 69 and 72
levels they sit at one 32×16 multiplier plus the exponent arithmetic, in a
library whose 32×32 multiplier is 65 levels on its own. Their asap7 numbers say
the same thing from the other side: `exp2` schedules at 1220 ps against a floor
of 1217. There is no structure left to remove; a real flow's parallel-prefix
adders are where the rest of it goes.

**And pipelining made them synthesisable at all.** Combinational
`tnca_sfu_rsqrt_xls` ran yosys past thirteen minutes without finishing; the
scheduled version completes in about one. That is not a PPA result, but it is
the difference between a module that can be measured on every commit and one
that cannot.

## Why E4M3 is written out rather than handed to `apfloat`

`apfloat` models an IEEE float, where the all-ones exponent means infinity or
NaN. OCP FP8 E4M3 does not work that way: it has **no infinity**, its all-ones
exponent is mostly ordinary numbers, and only `S.1111.111` is NaN. Its largest
finite value is 448, where an IEEE reading of the same bits would stop at 240.

Handing E4M3 to a generic IEEE routine silently loses the top of its range.
`xls/tnca_formats.x` therefore carries the format rules itself, with the
difference as a parameter (`HAS_INF`) rather than as a second code path, so
there is one rounding implementation to get right instead of two to keep
consistent. Overflow in E4M3 saturates to ±448; the OCP spec permits either
saturation or NaN, and the choice is stated there rather than left to be
discovered.

## Where they end up in the core

`rtl/core/tnca_fpu_multi.v` puts `tnca_widen_xls` in front of the hand-written
binary32 FMA, so FP8 and FP16 operands reach one verified multiplier. Because
widening is combinational, that costs no extra cycle. **It is the matrix
unit's MAC**: sixteen of them, with the format code from ISA §9C's `mod[3:2]`,
which is how `m.mma.async` came to take E5M2 and E4M3 fragments as well as
FP16 and BF16. An accumulator row op passes `f32`, and `widen(f32, x)` is `x`,
so one MAC serves both without a bypass.

That replaced the hand-written `tnca_cvt16_f32`, which did FP16 and BF16 to
binary32 only. It was deleted rather than kept: two conversion implementations
to keep in agreement is precisely what this document argues against, and the
generated one is checked exhaustively where the hand-written one was checked
over 131,072 vectors of two formats. Its last measurement — 140 gates and 14
levels — stays in doc/synthesis.md as the comparison point.

`tnca_cvt_comb_xls` is one per lane in `tnca_vector_fpu`, implementing
`vf.cvt.<src>.<dst>` (ISA §8.4 `func = 0x04`): all twenty-five pairs among the
five formats, lane-wise, with the narrow value right-aligned in its lane.

**Why the unpipelined instance there.** The vector FPU's clock is set by its
FMA at 115 gate levels; the convert chain is 54. A pipelined instance would
add 253 flops per lane — about four thousand — to buy timing the unit has no
way to use. The pipelined sibling exists for a context that does have a clock
of its own, and `tb_cvt_xls` checks both against the same oracle in the same
run, so neither can drift from the description or from the other. One DSLX
file, two implementations, one proof.

`rtl/core/tnca_sfu.v` is the SFU as the core sees it: **one four-function slice
shared by all VLEN lanes**, one lane per cycle. The ratio comes from the gate
counts above — the four units are about 60 k gates between them against 7 k for
one FMA, so a per-lane SFU would cost more than the rest of the vector datapath
for operations the architecture overview (§6.2) puts off the GEMM inner loop.
The slice is fully pipelined, so a stream of SFU instructions costs one pass
each rather than one pass per lane.

It retires through the vector FPU's existing multicycle completion port, the
one FP reductions already used, carrying its own lane enable so a predicated
`v.rcp.approx` leaves the inactive lanes alone. `CAPABILITY0.SFU` reads 1 as a
result; a build with `HAS_SFU = 0` neither instantiates the slice nor sets the
bit, which is what ISA §11 means by a capability bit.

What is *not* wired up yet: the five-format converter has no ISA encoding — §8.4
covers int↔FP32 and reserves the pack/unpack space — and `tnca_fpu_multi` is
not yet the matrix unit's front end. Both are ISA additions rather than fixes,
and the units are verified and waiting.

## Verification

Every generated module is measured against an exact reference, never against
another implementation.

| Bench | Coverage |
|---|---|
| `tb_cvt_xls` | 866,495 vectors. **Exhaustive**: every bit pattern of BF16, FP16, E5M2 and E4M3, to every one of the five formats. Binary32 sources are sampled from the boundaries. |
| `tb_sfu_xls` | 16,740 vectors against the §8.8 bound, evaluated in 60-digit decimal. 431 of them are checked *exactly* rather than inside a band, because §8.8 specifies them: the flushes, the saturations, and the special-value table — every infinity, every signed zero, every subnormal input, and the NaNs, which are compared by class because the ISA promises no payload. |
| `tb_sfu_core` | The same vectors through `rtl/core/tnca_sfu.v`: 1,042 batches of 16 lanes through one shared slice, checking the handshake and the lane-to-result mapping. |
| `tb_fpu_multi` | 5,012 vectors against `fp32.fma_ref`, with both FP8 formats exhaustive in one operand. |
| `t20_sfu` | The SFU from a program: 40 checks, the bands expressed as `vf.cmp` against bound vectors, the specified values as exact lanes, plus a masked write and back-to-back issue. |
| `t21_cvt` | `vf.cvt.<src>.<dst>` from a program: all twenty-five pairs, sixteen lanes each, compared bit for bit with an integer compare so a stray bit in the unused half of a narrow lane fails. Plus NaN by class and a masked convert. |
| `t11_matrix` | The same 16×16×16 product through all four `m.mma` fragment formats, against `mma_ref`. The FP8 cases are what a wrong 8-bit fragment stride fails: forcing the 16-bit stride turns them red and leaves FP16 green. |

Each pipelined bench measures its unit's latency instead of assuming it. The
neighbouring values are not near misses: at LATENCY 5 or 7, `tb_sfu_xls` puts
12,164 of 16,740 vectors outside the band, and `tb_sfu_core` fails on 12,319
lanes reading their neighbour's answer.

The references live in `tools/model/floatfmt.py` and
`tools/model/gen_sfu_tables.py`. The SFU's coefficients and the SFU's reference
come from the same script, so a table and its reference cannot drift apart.

Three bugs were found this way, all mine and all in the DSLX:

- every binary32 subnormal was flushed on the way to a narrow format, which is
  right for FP16 and both FP8s and **wrong for BF16** — BF16 shares binary32's
  exponent range, so its subnormals *are* binary32 subnormals;
- the FP16 subnormal widening had its exponent off by six binades;
- 128 interpolation segments gave `rcp` about 2⁻²⁰ relative error, over the
  2⁻²² budget, because `1/(1+x)` has much the largest third derivative of the
  four functions. 256 segments brings it to 2⁻²⁴.

The last one is worth noting as a method: the segment count was chosen from an
error estimate, the estimate was wrong for one of the four functions, and the
measurement said so.

## Regenerating

```
make xls          # regenerate rtl/gen from xls/
make xls-check    # fail if rtl/gen has drifted from the sources
make xls-report   # the scheduling table above, remeasured
```

`rtl/gen/*.v` **is checked in**, and nothing in `make test` needs XLS. That is
structural, not incidental: XLS publishes binaries for linux-x64 only, this
project is developed on macOS arm64, and a clone should be able to simulate and
synthesise the whole design without a toolchain it cannot install. XLS is a
build-time dependency of the RTL, not a runtime dependency of the project; you
need it only to change a `.x` file.

### Where the generator runs

`xlsgen.py` picks a runner, and which one it picks is an acceptance question,
not a convenience one.

**Local**, and preferred: XLS binaries on this machine, found at `$XLS_HOME`,
at `tools/xls/config.json`'s `local.xls_dir`, or at `.xls/cur`.
`tools/xls/fetch_xls.sh` puts them there — it downloads the release pinned in
`config.json`, verifies its sha256 **before** unpacking, and unpacks into
`.xls/<version>`. A directory only counts as an install if the three tools are
actually in it.

**Remote**, over ssh, for development only: XLS publishes linux-x64 binaries
and this is developed on macOS arm64, so `config.json`'s `remote` section
names a host to run them on.

`R1_STRICT=1` refuses the remote runner outright. The acceptance contract
(`doc/phase1_tasks_and_acceptance.md` R1-09) rules out depending on a personal
SSH host, a private key or a remote `cur` directory, and a generation step that
quietly fell back to one would make `make r1-check-generated` a check against
somebody's home directory. An acceptance environment is Linux x86_64 by
contract, so it runs the binaries itself.

Every generated file carries the XLS version and a hash of the DSLX it came
from in its banner.

Generation asks for plain Verilog (`--use_system_verilog=false`). XLS defaults
to SystemVerilog and emits unpacked-array literals for its lookup tables, which
iverilog aborts on with an internal assertion — and this project simulates with
iverilog and synthesises with yosys.
