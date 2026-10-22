# F09: the FMA's last stage into the accumulator file

**Status:** implemented on `r41s2-fma` (this commit, after F03), bit-exact, no cycle change, no latency
change. After the whole series the family's unit path projects to **6.04 ns** routed at 10 ns (target 9.0).
Not yet routed.

## The family

Attempt-3 routed census, residual population (`../F03/census_families.json`):

| Family (routed, 10 ns) | Endpoints | TNS (ns) | Worst slack | Its data path |
|---|---:|---:|---:|---|
| MTX `u_mac/u_fma…/p1_sh_mag` → `u_acc/g_rd[*].lane_mem` | **684** | **-223.9** | -0.871 | 10.394 ns, 27 levels |
| neighbour: VFPU `p1_sh_mag` → core `w_vdata` | 92 | -26.5 | -0.878 | 10.720 ns, 28 levels |
| neighbour: VFPU `p1_*` → reduction `acc` | 19 | -3.05 | -0.455 | 10.218 ns |

The FMA's result is combinational from its last stage into the consumer's register: the accumulator
file's LUTRAM data input (through `acc_w_data`'s `direct_w` select) and the vector writeback. The census path
is `p1_sh_mag` → the clamp to 100 → `sh_r - 1` → an 89-bit sticky mask, AND and OR reduction → the round
decision → a 24-bit increment (three CARRY8) → its carry-out (fanout 46) → `e_f = sel + carry` and
`e_biased = e_f + 127` → the `≥ 255` overflow compare → the result select → the accumulator write data.

## Design (`xls/tnca_fma.x`, `fma32_round`)

- **One shifter for the significand and the guard.** `mag` with a zero appended below it, shifted right by
  the normalising amount, puts `mag[sh-1]` at bit 0 (0 for `sh = 0`) and `mag >> sh` above it. The old code
  had two shifters, by `sh_r` and by `sh_r - 1`.
- **The sticky mask is taken from the shift amount itself**, on `mag << 1`: bits `[sh-1:1]` of it are
  `mag[sh-2:0]`, so no decrement stands in front of the mask, and amounts past the field saturate the mask
  exactly as the old clamp to 100 did.
- **Carry-select rounding.** `sig + 1` and `sig == all-ones` are formed beside the round decision, which
  then only selects among `sig`, `sig + 1` and `0x800000`; `carry_out = round_up && all-ones`.
- **The biased exponent and the overflow test likewise:** `eb0 + 1` and both compares
  (`eb0 ≥ 255`, `eb0 ≥ 254`) beside the carry, selected by it.
- `sig_f == 0` is `sig == 0 && !round_up`, formed beside the decision.

### Why the result is unchanged

- `(mag ++ 0) >> sh`: bit 0 is `mag[sh-1]` for `1 ≤ sh ≤ 89` and 0 for `sh = 0` (the old `guard`, which was
  0 for `sh_r = 0`); bits `[24:1]` are `(mag >> sh)[23:0]` (the old `sig`). For `sh ≥ 89` both are 0, as they
  were after the clamp.
- `((mag << 1) & mask(sh)) != 0` is `mag[sh-2:0] != 0` for `sh ≥ 2` and false for `sh ≤ 1` (the old `rest`);
  for `sh` past the field the mask is all ones, `rest = mag != 0`, as with the clamp.
- A left shift (`sh_left`) is unchanged: the low 24 bits of `mag << sh`, no guard, no rest.
- Rounding: `sig + round_up` is `sig` without a round-up, `sig + 1` with one, and `2^24` (renormalised to
  `0x800000`, exponent + 1) exactly when `sig` is all ones; the exponent and overflow candidates are
  `eb0 + carry` and `eb0 + carry ≥ 255`.

## Verification (GB10)

| Check | State | Result |
|---|---|---|
| DSLX unit tests | this commit, final | 2/2 pass |
| `tb_fp32_fma`, latency 2 and 3, exact Fraction oracle | this commit, final | 41,240 vectors each, 0 failures |
| `make fma-deep`, latency 2 and 3 | this commit, final | 516,240 vectors each, 0 failures |
| Verilator miter against r41s2's generated RTL (`../F03/miter/`) | this commit | 210 M vectors × 2 latencies, 0 mismatches |
| the same | final | 700 M vectors × 2 latencies, 0 mismatches |
| `matrix_oracle.py` (12 cases, every dtype and K, row ops, ABUF 0 and 1) | final vs r41s2 | identical `results.txt` (accumulator words and per-command busy cycles) |
| `tnca-numeric-probe matrix` on it | final and r41s2 | 36,864 accumulator words each, 0 mismatches |

Core, system, MMA and bitwise K-order regressions on the final state: `../F08/README.md`.

## Cycle impact

None: latency 2, the same writeback timing; the MTX `PIPE` (FMA_LAT + OPND_REG = 3) and its tag pipeline
are untouched; the matrix oracle's per-command busy cycles are identical to r41s2 (MMA K16 287/283 with ABUF,
279 without, K32 551/559, K64 1095; row ops 3 and 8; m.zero 18).

**SystemC model keys:** no change (`mtx_pipe_cycles`, `mtx_row_cycles`, `vector_fp_latency` unaffected).

## Unit OOC, 10 ns, xcu280-2L, image parameters

Method and scripts: `../F03/README.md`. Projection = after-synth × (census routed / baseline synth).

| Unit path (family) | Baseline synth | Census routed | Ratio | After F03+F09 | After F03+F09+F08 | Projected (final) |
|---|---:|---:|---:|---:|---:|---:|
| MTX last stage → `u_acc` lane_mem | 5.795 (27 levels) | 10.394 | 1.794 | queued | **3.366** (15 levels) | **6.04** |
| VFPU last stage → reduction `acc` | 6.084 | 10.218 | 1.680 | queued | ≤ 5.0 | ≤ 8.40 |

"≤ 5.0": the path is absent from the unit's list of every endpoint whose worst slack is under 5 ns, so its
delay is at most 5.0 ns; its exact value comes with the family re-report queued on the host. "queued": this
commit's tree (F03+F09) is queued on the shared host; the column is filled by a later evidence commit.

The VFPU → core `w_vdata` neighbour ends in the core's writeback mux, outside the VFPU, so it is not timed in
the VFPU OOC; its FMA part is the same last stage.

## Open risks

- Synthesis estimates scaled by the census ratio; the routed result decides. The accumulator file's LUTRAM
  write data still comes through the `direct_w` select in `tnca_matrix.v`, unchanged.
- The `w_vdata` neighbour also contains the core's writeback mux (core-owned, not changed here).

## Files

- `ooc_reports.tgz`: OOC reports of this commit's tree (no checkpoints).
