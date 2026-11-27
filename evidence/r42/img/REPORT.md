# R42 img: combined image preparation, final report (written by the lead from the worker's message; the harness refused the worker's own REPORT.md)

Branch `R42-img` (final commit `befa78cf`, built on `origin/R42` `e2fdfb2f`; **not merged into R42**, see the plan entry of 2026-11-27 for why), sparse worktree `/home/ubuntu/r42-wt/img`, evidence in `evidence/r42/img/` of that branch
(`rtl_3c_tables.txt`, `rtl_1c_tables.txt`, `model_prediction.txt`, `ooc/ooc_summary.txt`, `staging.txt`, `commands.txt`). **Domains: stand-in simulation, out-of-context Vivado, or model; nothing is a board number.**
No link, card, board or host load was requested.

## Built
| Step | Commit |
|---|---|
| merge R42-lineelig (clean) | `b5fb3158` |
| merge R42-mtx | `cea69b56` |
| merge R42-fp32 | `4c7f3f5b` |
| bench and stand-in wiring, arm images | `bf4da745` |
| lockstep bench, unit-test runner | `b8a081ba` |
| overlapped sequencer runs the binary32 command; profile `core-r42-img` | `a099878e` |
| profile `core-r42-img-lf` | `fffaccf1` |

Merge conflicts: shell parameter lists and instantiations resolved as a union and checked by script; both Makefile blocks and all three profiles kept; `doc/support_matrix.md` and the target jsons regenerated with the repo generators; in `frontend.py` the xfer
deferred-fragment path and the fp32 fragment path are kept (their tile kinds are disjoint); in `tnca_matrix.v` the lineelig 16-register A buffer is re-applied in the old sequencer that mtx had moved. MTX_PIPE now runs `m.mma.f32.async` (pending-command flag,
B-row-in-flight flag, A-buffer depth 16); the core's pendable-mma test includes the new opcode; `MTX_PIPE` and `MAC_F32` reach the stand-in tools through `board_regs`, `link_budget_rtl.py`, `image_geometry_tests.py`. Every item's default is cycle-identical to origin/R42.

## Gates (all passed)
`make unit`, core-g2c4, core-run2, core-mmadp, core-abuf16, r41-xpose8-rtl, r38-xpose-rtl, tensor-line32, core-r42, matrix-ovl, core-mmaf32, core-r42fp32 (before the OVL+F32 integration, exit 0); matrix lockstep against origin/R42's matrix engine, 300 cmds x 2 seeds,
32/32 cycle-identical (before) and 10/10 (final tree); core-g2c4, core-run2, core-mmadp, core-r42, core-r42fp32, core-mmaf32, matrix-ovl, core-abuf16, core-r42img on the final tree exit 0; overlapped sequencer vs old sequencer with binary32 commands in the random list
(busy cycles 50-64 % of the old sequencer's) 7/7; `make r1-check-generated` pass; `make r1-profile-tests` PASS (6 checks incl. `core-r42-img`, `core-r42-img-lf`). `tnca_tensor.v` is byte-identical to lineelig's, so its own miter/lock/mutant gates stand (not rerun).
Not run: `default_change_audit.py`. Default-off identity: rtl-1c 21 cheap tiny points plus gpt2_block (173,811) and llama_decode (177,017) cycle-identical; kernel synthesis of the merged RTL at defaults +322 LUT (+0.04 %), +49 FF, DSP/BRAM unchanged (noise);
rtl-3c matches only to about +-0.2-1 % across builds (matmul/small 41,583 vs 41,131 for the same default RTL), so it cannot prove identity.

## rtl-3c arms (tiny/small, current default software; arm / A cycles; A = run-3 image; L = LINE_HALF, M = MTX_PIPE, F = MAC_F32; bit-identical outputs, all records inside tolerance)
| Point | L | M | F | LF | LMF |
|---|---|---|---|---|---|
| matmul fp16 tiny | 0.818 | 0.981 | 1.001 | 0.817 | 0.779 |
| matmul fp16 small | 0.584 | 0.927 | - | - | 0.457 |
| linear tiny | 0.837 | 0.985 | 0.979 | 0.837 | 0.758 |
| gemv tiny | 0.719 | 0.976 | 0.964 | 0.716 | 0.650 |
| gemv small | 0.799 | - | - | - | 0.702 |
| bmm tiny | 0.875 | 0.972 | 0.999 | 0.876 | 0.852 |
| mlp tiny | 0.810 | - | - | - | 0.785 |
| matmul fp32 tiny | 0.997 | 0.997 | 0.743 | 0.742 | 0.705 |
| matmul fp32 small | - | - | 0.667 | - | 0.618 |
| conv2d tiny | - | - | 0.812 | - | 0.796 |
| attention_prefill tiny | 0.989 | 0.989 | 0.910 | - | 0.914 |
| fp8 Hub-kernel proxy `op.fp8_linear/xl/fp8-m16-k256-n128` | 0.984 | - | - | - | 0.947 |

Marginal value of M on top of L (name-matched): gemv -9.6 %, linear -9.8 %, matmul tiny -4.4 %, bmm -2.2 %; at small size LMF against L is -21.7 % on matmul and -12.1 % on gemv; M alone about -2 % on tiny points. F also moves attention_prefill by -9 % (one tiny point).
rtl-1c with the combined profile (tiny): resnet_block 0.741, conv2d 0.739, mlp 0.937, T17 geometric mean 0.971 against origin/R42; all 138 records pass. Points slower than 2 % (argmax +2.3 % rtl-1c; add +2.9 %, rmsnorm +2.1 %, softmax +9.0 % rtl-3c; others in `rtl_1c_tables.txt`)
are a profile-name layout effect: the same L image under A's profile name reproduces A exactly (add 591, softmax 1809 vs 1808), and core-r42, core-r42-fp32 and core-r42-img give identical cycles on these points in rtl-1c though their hardware differs; the reason the name changes the artifact was not found.

## Resources and timing (OOC, Vivado 2023.2, 10 ns, tile 0 in SLR0; measured)
| Run | CLB | LUT | FF | WNS (ns) |
|---|---|---|---|---|
| base (origin/R42 RTL, run-3 image) | 43,077 | 251,632 | 68,740 | +0.816 |
| idef (merged RTL, defaults) | 42,763 | 252,520 | 68,752 | +0.248 |
| l | 43,235 | 252,430 | 68,897 | +0.625 |
| f | 43,258 | 252,035 | 68,714 | +0.607 |
| **lf** | 43,242 | 252,706 | 68,882 | **+0.909** |
| m | 42,879 | 251,252 | 68,997 | +0.291 |
| lmf | 42,939 | 252,848 | 69,019 | +0.363 |

Per core (kernel synthesis): core base 213,352 LUT, lf +1,085 (+0.51 %), lmf +1,177; tensor engine LINE_HALF about +0.8k; matrix unit in isolation MAC_F32 +812, MTX_PIPE +743, both +615 (WNS stays >= +1.2 ns); the 32x512 A buffer maps to LUTRAM (RAM32M16 x37) so BRAM/FF unchanged;
kernel totals base 814,086, lf 815,876 (+0.22 %). Timing: the identical-function control (idef) scores +0.248 against base +0.816, so single WNS values cannot resolve effects below about 0.6 ns. Worst slack of the core `x_valu_imm` control cone: base 0.830, idef 0.964, l 1.000, f 1.896, but
m 0.291 and lmf 0.363: the only item-specific timing signal (the cone runs through the accumulator-hazard and operand-fetch enable that MTX_PIPE modifies). SLR packing (derived): run-3 SLR0 95.56 % CLB, ceiling 96.74 %; tile-0 delta at 5.84 LUT/CLB gives about 95.9 % for lf and lmf; SLR1/SLR2 about +0.3 points.

## Predicted board effect (model: stand-in ratios per (family, dtype) applied to the ab-int2 board table, relative to its 1.638x geometric mean)
`family/2` halves the log-ratio on medium, large and hbm tiers; `family` applies the full ratio; `models` also gives gpt2_block, llama_prefill, llama_decode the mlp ratio; pointwise families held at 1.0; resnet_block takes conv2d's ratio.
| Image | measured only | `family/2` | `models` | `family` | xl tier |
|---|---|---|---|---|---|
| LF | +2.5 % | **+5.9 %** | +7.9 % | +8.2 % | +2.8 to +2.9 % |
| LMF | +3.3 % | +7.5 % | +9.8 % | +10.5 % | +5.2 to +5.5 % |
Uncertainty roughly +-2-3 points (ratios from tiny/small only, medium/large/hbm extrapolated; stand-in is not an exact board model; the profile-name layout effect is +-3-9 % on tiny pointwise points; xl rests on one reduced-shape FP8 proxy). The baseline is ab-int2's table (1.638x), not the ab-int3 software.

## Worker's recommendation
One image, **LF**: LINE_HALF + MAC_F32, profile `core-r42-img-lf`, parameter file `evidence/r42/img/images/image_lf.tcl` = `evidence/r41/hbm_bw/g4/run2/image.tcl` (run-3 image 791b8af2) plus `TENSOR_LINE_HALF 1` and `MAC_F32 1`; about +0.5 % LUT per core, tile 0 near 95.9 % of SLR0,
best tile screen of seven (+0.909 ns), +5.9 to +8.2 % on the 113 geometric mean (model), +2.9 % on xl. Drop MTX_PIPE from this image: +1.6 to 2.3 points of geometric mean, but its core control cone sits at +0.3 ns where every other run is at >= 0.83 ns; a failed link costs 3 h 40 min and one of five runs;
it stays in the RTL default-off; follow-up: shorten the hazard cone (`x_haz`, `d_pf_va_en`), re-screen, take it into a later image. If the timing risk is accepted, `image_lmf.tcl` with profile `core-r42-img` passes every gate.
Staging: source tree `stage_88890078.tar.gz`, sha256 `67bc6d529393efca90fb5b133e7c73e70d220d0ba217aaa17aca4250cb4dabbb`, extracted on the x86 host at `~/r42img/stage/src_88890078`; XO packaging dry run passed for both images (exit 0, kernel.xml m_axi 512 bit x3, 94 files, 0 critical warnings); the link recipe is in `staging.txt`.

## Failed or not done
- One preserved failure: rtl-3c L-arm `model.mlp/tiny/fp16-t16-d64` on thor gave a runtime fault `[DEVICE_LOST] ... invalid or unknown value` in `_copy_kernel`; it passed on two reruns (GB10 35,935 cycles, thor 35,926; lineelig measured 35,882). Cause not established; the failed record is kept in the table.
- Not measured on the stand-in: mlp small and resnet_block on rtl-3c; the medium, large and hbm tiers; the full rtl-1c tiny tier for LMF (24 of 27 points; gpt2_block, llama_decode, llama_prefill missing). `default_change_audit.py` not run.
- Resource use: GB10/thor about 5.3 h of wall window with at most 3 processes per host, except short overlaps (GB10 briefly 4 during the rtl-1c sweeps; thor about a minute at 5 from a duplicate run the worker killed by PID). No other worker's process was touched.
- x86 host check requested by the coordinator: none of the worker's Vivado or chain jobs were killed or restarted (this also answers the area worker's pattern-kill question for this worker's jobs).
