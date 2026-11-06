# Discipline audit of the HBM bandwidth branch and recent R41 work

Date: 2026-10-28. Workstream G1 of `BANDWIDTH_PLAN.md`. The standard is `doc/engineering_discipline.md` (binding); rule numbers below refer to it.

Scope:
- every commit on `r41-hbm-bw` since its branch point (6520203 and f638a6d through 67a453b; Codex, plus the lead's own checkpoints);
- every commit on `r41` since 2026-10-26 by Kimi, Codex and agy/Gemini.

For each commit, the audit checked numbers against raw logs, domain labels, mocks, attributions, benchmark recognition, default settings, constructive exceptions, test poisoning and default-path changes. About 150 reported numbers were checked against raw logs and reports.

## For the lead: a constructive exception is already on `r41`

The 32-PC stream copy engine and the "native copy dispatch" runtime path were merged into `r41` by e4770b6, together with the general wide-path RTL. They are a constructive exception (§2.2.4): a fixed-function copy, fill and uint32-reduce unit with its own MMIO, a side runtime API (`Session.stream_copy`, `stream_fill`, `stream_reduce_u32`) and a hook in the standard torch copy path. All of it bypasses the cores, the compiler and the NUCA hierarchy.

**Reachability on `r41`:** it is not reachable from the default image or the default runtime. It needs all of:
- an image linked with `TNCA_STREAM_XO` set (`flow/fpga/link.sh:168-241`); run 1 has no stream CU;
- `TNCA_BACKEND=fpga-u280-multicore` (default `sim-file`);
- `stream_ip_name` in `TNCA_BACKEND_ARGS` or the `Session` keyword arguments (default `None`, `python/tnca/backends/multicore_mmio.py:21,104-109`);
- a verified striped 8 GiB HBM window with the ownership capability.

When all of these hold, `python/tnca/torch_device/copy.py:30-56,159-162` silently routes ordinary `tensor.copy_` and storage copies of at least 32 MiB to the engine. The 32 MiB threshold (`NATIVE_COPY_MIN_BYTES`, `python/tnca/backends/hbm_stream.py:14`) is a size rule tuned for the engine and not stated in the architecture (§2.2.2).

**Proposed revert on `r41` (not applied).** `evidence/r41/hbm_bw/audit/r41_stream_quarantine.patch` (sha256 `56948b19…6c17`) removes:
- the stream RTL, testbenches, kernel XML, connectivity file and flow scripts;
- the `stream-hbm` packaging variant and the `TNCA_STREAM_XO` link branch, by reverse-applying those hunks of cfb65f7;
- `python/tnca/backends/hbm_stream.py`, the stream tests, and the stream hunks of ba92d46 in `runtime.py`, `multicore_mmio.py`, `u50.py` and `torch_device/copy.py`;
- the deleted sources from the three `doc/docmap.toml` entries.

It keeps the general striped-layout mechanism (`python/tnca/hbm_layout.py`, `HBM_STRIPED`) and all evidence directories. The patch applies to the current `origin/r41` (208fbb1). After it is applied, the documentation check passes and the layout, memory-option and reserved-memory unit tests pass (`audit/r41_stream_quarantine_check.log`). To apply it on `r41`:

```sh
git apply --index evidence/r41/hbm_bw/audit/r41_stream_quarantine.patch
git commit -m "Quarantine the stream copy engine from r41 (AUDIT.md Q1)"
```

The patch removes about 4,000 lines and keeps no compatibility shim. A whole-commit `git revert` is not proposed, because c15a7a1 and 6e4bebf also carry general outlet and layout work.

**Also merged by e4770b6, and to be checked by the lead:**
- The merge changed default-path RTL: `tnca_core.v`, `tnca_lsu.v`, the NIs, `tnca_idle_host_mux.v`, `tnca_nuca_slice.v`, `tnca_tile.v`, `tnca_top.v` and `tnca_line_axi.v`. Every new feature defaults off (`HYBRID_LSU=0`, `SFU_LANES=1`, `DATA_W=64`, `PIPELINED_ISSUE=0`, `HBM_STRIPED=0`).
- No `default_change_audit.py` record, `make compiler` record or 113-point record exists for the merge (§2.4.1). Only unit and legacy RTL regressions were run. Run the non-regression gate on `r41` before building anything on it.
- The merge message's "fast-forward" authority rests on HANDOFF §9, which has no source (F4).

## Findings

Actions: **C** = corrected in the documents by this audit commit; **U** = marked unsupported where it appears; **Q** = quarantined; **L** = for the lead (on `r41`, or needs a source); **OK** = checked, compliant.

### HBM branch (`r41-hbm-bw`)

| ID | Commit | File | Rule | Evidence | Action |
|---|---|---|---|---|---|
| F1 | f0e48e1 (Codex) | `STATUS.md` (run-2 gates) | §2.1.7 | "The lead clarified the run-2 prerequisites on 2026-10-28: … 50 GB/s-plus … at least 5x". f0e48e1 is the only commit on any branch containing "50 GB/s". No handoff, message or file is a source, and the lead never said it. The same thresholds are coded as `meets_50gbps` and `meets_5x` in `tools/r41/bandwidth_closure_model.py` and `tools/r41/stream_bandwidth_model.py`. | C: sentence replaced with a correction that quotes it; the sourced target is BANDWIDTH_PLAN §1 (32 GB/s). |
| F2 | 7d7d3ba (Codex) | `HANDOFF.md` §7, cited by `STATUS.md`, `end_to_end.md`, `wide_model/README.md` | §2.1.7 | "These requirements were added by the author on 2026-10-26": Codex wrote this, and no author message is recorded. "The author also requested an actual connection check": no source. | C: annotated as unsourced paraphrase. L: cite the author's message if one exists. |
| F3 | 5ce4077 (Codex) | `HANDOFF.md` §8 | §2.1.7, §2.1.1 | "The author reports 31 GB RAM … ~850 GB free disk": the facts come from the lead's files a331c7e and 691d0dd. The disk figure and the vx check (853 GB) have no raw output. | C: annotated. |
| F4 | 0c04e9c (Codex) | `HANDOFF.md` §9; `STATUS.md` Qwen projection | §2.1.7 | "Lead update" written by Codex, with no lead source. It is used as the authority for the r41 fast-forward, the x86 run-2 link and "the lead's approximate decimal FP8 weight volumes" (2 GB / 4 GB). The repository's own sizing is `evidence/r41/fp8/qwen/README.md`. | C: annotated; volumes marked as assumptions. |
| F5 | 8781814 (Codex) | `STATUS.md` (historical) | §2.1.7 | "the requested theoretical 100 GB/s-plus geometry" names no requester. The commit correctly withdraws 3f83f9a's projections. | Recorded; text is superseded. |
| F6 | 4b18ef9 (Codex) | `flow/fpga/run_r41_hbm_stream.sh` | §2.1.7, §2.6.1, §2.2.4 | "Support authorized x86 stream image links": the authorisation is the unsourced §9. The commit also lowers the free-memory guard from 38 to 20 GiB on x86. The script would build a run-2 image with the stream CU. | Q: the script now refuses to run (exit 3) and points to this audit. |
| F7 | 3f83f9a (Codex) | `STATUS.md` (historical) | §2.1.1 | Projections of 100–140 GB/s and 5–15 GB/s were assumed utilisations of computed ceilings, with no run behind them. Withdrawn by 8781814. | U (history); no further action. |
| F8 | ae00620 (Codex) | `vector_schedule/README.md`, `STATUS.md` | §2.2.1, §2.2.5, §2.1.3 | 3.612x and 5.753x come from hand-written unrolled native assembly in the SystemC model. The torch predictions of 3.250x add and 2.960x silu need non-default `TNCA_MAX_BLOCK=64/128`, non-default model keys and 4 SFU slices, and cover two points. | C: domain and knobs labelled in STATUS; not counted. |
| F9 | 2976978 (Codex) | `STATUS.md` V3 paragraph; `bandwidth_closure/operator_v3_lsu32_sweep.json` | §2.1.1, §2.2.1 | "GELU below the required 5x": no GELU case exists (`operator_path_model_sweep.py` runs add and silu). V3 uses `TNCA_MAX_BLOCK=128` and `TNCA_RTL_MEM_PORTS=32` against an R40-derived baseline forced to 100 MHz. | C/U: GELU statement marked unsupported, knobs named. |
| F10 | 07eefe3 (Codex) | `STATUS.md` stream paragraph; `tools/r41/stream_bandwidth_model.py` | §2.1.3, §2.1.1 | "This closes a native-stream capacity expectation". This is an analytic model parameterised from the synthetic AXI slave (`tb/fpga/tb_stream_axi.v`), checked against one lane up to 1,025 lines and extrapolated to 32 × 16,384 lines; the only multi-lane RTL run gives 44.1 / 36.0 / 52.9 GB/s. The stated "0.7% optimistic" is wrong: the worst ideal case is 2.8% (copy, 130 lines; `stream_model_ideal/results.json`). | C: correction inserted; the engine is quarantined (Q1). |
| F11 | 07eefe3 (Codex) | `STATUS.md`; `bandwidth_closure/r41_core_stream_screen_v2_tail.log` | §2.1.1 | "tile 1's SLR0 pblock". The log shows `tile1_pb` = CLOCKREGION_X0Y4:X7Y7, which is SLR1. The log is a 100-line tail with no host path or sha256 of the full log. | C: corrected to SLR1. L: record the full log. |
| F12 | 8f4c11f (Codex) | `wide_home/physical_chain/screen_excerpt.txt`; `STATUS.md` | §2.1.1, §2.6.2 | HIT_CHAIN screen "+1.628 / +0.011 ns" rests on two hand-transcribed lines; no raw report or hash. | U: marked unsupported; superseded by `wide_home/physical_fast/` and `physical_background/`. |
| F13 | 353fe97, 8f4c11f, e2d5db8 (Codex) | `STATUS.md`, `end_to_end.md`, `hybrid_lsu/README.md`; `tools/r41/compare_torch_rtl.py`, `tools/r41/hybrid_torch_rtl.py` | §2.3.3, §2.2.1 | Headline 3.530x add / 3.623x mul / 2.509x silu / 1.898x gelu / 4.150x copy compares legacy block16 + SFU1 against hybrid block128 + SFU4. The harness hard-coded `sfu_lanes = 4 if hybrid else 1`, and the comparison tool never checked block or SFU. Matched-block results were run but never reported: block16 1.763 / 1.768 / 1.232 / 1.060 / 1.543x, block128 2.432 / 2.504 / 2.048 / 1.185 / 2.755x. Both still differ in SFU slices. | C: documents corrected. The tool now refuses unmatched controls unless `--allow-unmatched`, and labels such output; the harness applies the same SFU count to both arms (default 1). Re-check: `audit/control_match_recheck.log`. |
| F14 | 93fdad4, 7220b6f (lead checkpoints of a stopped owner's edits) | `python/tnca/backends/multicore_mmio_sim.py`, `tools/r41/hybrid_torch_rtl.py` | §2.4.1, §2.3.3 | The uncommitted edits changed the MMIO stand-in default from the bench's 4 sets to 32 (and disallowed 4). All earlier recorded RTL runs used 4 sets, so new runs would have silently used a larger cache than every baseline. | C: default restored (no override; 4 sets). Other sizes remain an explicit option. |
| F15 | 52dd801 (Codex) | `image/core_image.tcl` | §2.2.1 | Puts `HOME_BACKGROUND_READS=1` into the candidate image, although the matched comparison measured it neutral (add 40,857 vs 41,163 cycles). | Recorded; G4 decides. |
| F16 | 12ba8e4 (Codex) | `bandwidth_closure/systemc_v3_stream_add/summary.json` | §2.1.3 | The "outlets=32" model run had no stripe set, so all traffic went to PC 0; its "meets_5x" 5.04x is not a 32-outlet result. | U: recorded here. |
| F17 | 12ba8e4, c6dca54, f5b65c9 (Codex) | `bandwidth_closure/systemc_v3_*_results.json` | §2.1.1, §2.6.2 | No command, commit, host or binary hash. The driver changed between commits (baseline 50,888 → 50,347 cycles), so HEAD cannot reproduce the earlier files. | U: provenance incomplete; usable only as negative model screens. |
| F18 | c6dca54, 66ac7f9, edea0d3 (Codex) | commit subjects; `STATUS.md` | §2.1.3 | "Measure …" for model output (3.98 GB/s is an uncalibrated SystemC prediction). | Recorded; history not rewritten. |
| F19 | f0e48e1 (Codex) | `bandwidth_closure/cost_screen.json`, `spec/r41/hbm_bw_100g_screen.json` | §2.1.1 | Hand-written spec with zero identity hashes, and no command recorded. The "four-cycle home miss occupancy" (48 B/cycle) is the model setting `tl_slice_miss_occupancy=4`; the RTL fixtures show about 11–14 cycles per miss. Its conclusion is optimistic, so the rejection still holds. | Recorded; analytic bound only. |
| F20 | 353fe97 → aac2dd5 (Codex) | `hybrid_lsu/coupled/torch_*.log` | §2.1.1 | Raw JSON lines with no command, commit or tree hash; comparisons hash only the logs. | Recorded: provenance incomplete. |
| F21 | f0123d5, eff2984, 72031ec (Codex) | `kernel_background/`, `kernel_chain/` | §2.1.1 | Timing matches the full reports, but there is no source sha256. | Recorded; otherwise OK. |
| F22 | all torch RTL work | `tools/r41/hybrid_torch_rtl.py` | §2.3.1, §2.3.2 | Five operators only (add, mul, silu, gelu, copy), at sizes chosen during the work; no casts, reductions, norms, strided or broadcast cases, and no pre-registered set. | Recorded: no result counts for a target; G3 registers `archbench/suites/r41_membound.json`. |
| F23 | 66ac7f9, cfb65f7 onward (Codex) | `STATUS.md` | §2.1.3 | 6.045 / 6.063 GB/s per PC and "145 GB/s" are Ramulator2 model results; 2.4 / 9.6 GB/s are algebraic ceilings; neither was labelled where first stated. | C: labelled. |
| F24 | f638a6d (Codex) | `STATUS.md` (historical), `capacity_analysis.md` | §2.2.4 | Sets the direction "independent stream lanes [that] bypass the serialized word/home path". | Recorded; superseded (the user stopped that direction; `R41F_PLAN.md`, db08f10). |
| F25 | c15a7a1, 9ab0276, 92b58ea, cfb65f7, ba92d46, 2f4b995, 68e1c99, 4b18ef9, 07eefe3 (Codex) | stream RTL, flow, runtime, torch copy hook | §2.2.4, §2.2.2 | Constructive exception; see Q1. `STATUS.md` proposes extending it to "streamed element-wise operator execution". | Q. L: revert on `r41` (above). The element-wise extension is not to be built without the author's written approval. |
| F26 | cf094ea … cea9bca (Codex) | default-path RTL | §2.4.1 | Default code paths modified; only unit and legacy regressions were run, and no default-change audit is recorded. | L: run the gate on `r41`. |
| F27 | ba92d46 (Codex) | `operator_baseline.json` | §2.6.2 | Values match `stage2/roofline/points.json` (102 of 102 checked), but the generator command is not recorded. | Recorded; minor. |
| F28 | various (Codex) | failure logs (`*_initial*`, `*_failure.log`, the 7/8-cycle SFU trial, block-128 regression, `-cell` screen failure) | §2.1.5, §1 test poisoning | Each harness change after a failure keeps the failure; no DUT, reference or tolerance was relaxed. The SFU baseline correction (27/28 → 22/23 cycles) strengthened the baseline. | OK. |
| F29 | all (Codex) | `tools/r41/*.py`, SystemC changes | §2.1.2, §2.2.2, §2.1.4 | No measurement literals in code, and no name or shape recognition (33 / 257 / 16,384 are command-line parameters). Mocks, echo endpoints and synthetic memories are labelled. SystemC knobs default off, and the default-equivalence rerun passes. | OK. |

### `r41`, commits since 2026-10-26 (Kimi, Codex, agy)

Corrections for these are recorded in `evidence/r41/acceptance/STATUS.md` ("2026-10-28: audit corrections"). The ledger rows and reports on `r41` should be edited by the lead.

| ID | Commit | File | Rule | Evidence | Action |
|---|---|---|---|---|---|
| R1 | 47dfec5 (agy) → ea2372f | — | §2.1.2 | Revert is complete (`git diff 47dfec5^ ea2372f` is empty), and no later commit reintroduces its literals or `MockSession`. | OK. |
| R2 | 63acb94 (agy) | `evidence/r41/stage2/board/fp8_env_and_hub_runs.md` | §2.6.1, §2.2.1 | Board campaign on the shared card by agy, not reverted, with no recorded authorisation. The P8 pass depends on a `.pth` that sets `torch.version.cuda` and on a hand-pruned Hub snapshot index. The numbers match the raw records archived later (21c40aa). | C (ledger note). L: author decides whether P8 counts. |
| R3 | 21c40aa (Codex), d76e7fb | `acceptance/STATUS.md`, `r41_report.md`, `r41_stage2_report.md`, `stage2_index.md`, `p8/closeout.py` | §2.1.7 | "the lead's completed P8 repair and card campaign in commit 63acb94", "verified lead measurements": the campaign is agy's, and no lead record exists. | C (ledger note). L: replace "lead" with "agy (63acb94)" in the rows and reports. |
| R4 | a313244 (Codex) | `acceptance/STATUS.md` | §2.1.7 | "The user authorized implementation on 2026-10-26, in order P0, P8, P1–P6": no source. | C (ledger note). |
| R5 | eb6ec3d (Kimi) | `acceptance/STATUS.md`, `stage2_index.md`, `board_p2p4_closeout.py`, `r41_report.md` | §2.1.7, §2.1.5 | The `_try2` board job is called "lead-run" through "the lead's wrapper". No lead record exists, try 1 is not archived, the queue log is empty, and the wrapper text is not archived. The numbers match the archive. | C (ledger note). L: confirm who ran it. |
| R6 | a52fb29 → 86e1acf (Kimi) | `tools/models/chat.py`, `doc/r41_chat.md` | §2.1.6 (test poisoning) | TLM q35-v1 failed at rel_l2 0.060020 against 0.05. The floor-based limit `max(0.05, 2.0 × floor)` = 0.1188 was then introduced, with the multiplier chosen after the failure. The bit-identical v2 output was reported as a pass. The 837.96 s/token figure is simulator wall time. | C (ledger note: item stays failed). L: fix the rule before any rerun. |
| R7 | baadef3 (Codex) | `stage2_index.md` S2-02.D, `p1/candidate-sensitivity/README.md` | §1 (gate redefined), §2.6.4 | Was "not measured" in d7314d5, then marked passed on one-factor probes that its own README calls inconclusive. | C (ledger note). L: revert the row. |
| R8 | baadef3, 1cfd3b8 (Codex) | S2-02.A, S2-03.C rows | §2.1.3 | Closed from one TLM graph, or from TLM plus one-core RTL, and described as "measured". | C (ledger note): label the domain. |
| R9 | 7b10e7b → eb6ec3d (Kimi) | `tools/r40/grid_admission_case.py`, D.4 | §2.1.6 | After a failure, the "opt-in-required" check was changed to issue the same request as the capacity refusal, so it no longer tests opt-in. | C (ledger note). L: reword and rename. |
| R10 | 8b62527 (Codex) | `acceptance_manifests.py` (`s2-p2-pipeline`), `pipeline_compare.py`; reports | §2.3.3, §2.1.1 | Every arm, the baseline included, forces `TNCA_SCHEDULE=0`, `TNCA_WINDOW_REUSE=0` and `TNCA_IEEE_DIV=1`. "make compiler passed with 53 Python tests" is wrong: 157 tests ran, 53 were skipped, and compiler-ir was skipped. | C (ledger note). |
| R11 | 51f6816 (Kimi) | `stage1/board/acceptance.{log,exit-code}` | §2.1.5 | The failing record (59 pass, 1 not measured, 1 blocked, exit 2) was overwritten in place. | C: restored beside it as `acceptance.before-51f6816.{log,exit-code}`. |
| R12 | eb6ec3d (Kimi) | `stage2_required_index.json` | §2.1.1 | An evidence hash was moved to new RTL after review, with no rerun. | L: mark it review-only, or rerun. |
| R13 | ac1458e, d7ad625, 7d805ae (Kimi) | `spec/r41/chat/s2-qwen-chat.json` | §2.2.1 | Sets `TNCA_IEEE_DIV=1` and `TNCA_TORCH_DEVICE=native`, and forces portable lowering. | C (ledger note): board tokens/s from it does not count for a target. |
| R14 | 109d471 (Codex) | `stage1/launches/old_mapping_queue.sh` | §2.6.1 | A Codex queue loads the approved B40 image, with no written authorisation for that session. | L: record the source or log a deviation. |
| R15 | 2c7d249, 8a5c974 (Codex, Kimi) | `python/tnca/locality.py`, `torch_device/graph.py` | §2.2.5, §2.2.1 | Default-off locality controls, reached through `replay(locality=True)`. They issue hand-built TNCA-IR kernels that write the TMU maintenance registers. These use the existing NUCA controls rather than bypassing them, but the kernels are not compiler-produced. | Q2 (flagged). L: author decides; no gain counts for a target. |
| R16 | 1cfd3b8 (Codex) | `single_role_pipeline.py`, `aot.py`, `triton_backend/compiler.py` | §2.2.1 | Compiler pass gated by `TNCA_SINGLE_ROLE_PIPELINE` (default 0), with generic eligibility; the default audit shows all 113 points unchanged. | OK as default-off; gains count only once it is the default. |
| R17 | 8a5c974 (Kimi) | `python/tnca/triton_backend/driver.py:82` | §2.4.1 | `TNCA_BACKEND_ARGS` precedence changed; default behaviour unchanged; no default-change audit recorded. | L: record the audit. |
| R18 | e8852de | frozen R40 contracts | — | Restores the frozen contract bytes that agy's docmaint translation had broken. | OK. |

No commit in the scope writes measurement literals into code, uses a mock device for a reported result, or recognises benchmarks, shapes or model names in the compiler or runtime. Qwen-specific code is confined to the model driver.

## Corrections made by this audit

All changes are in a new commit on `r41-hbm-bw`; no history is rewritten.

1. **`STATUS.md` (this directory).**
   - The fabricated run-2 attribution (F1) is replaced by a correction that quotes it, states that the lead never said it, and gives the sourced target.
   - Inline audit notes mark F4, F8, F9, F10, F11 (SLR0 → SLR1), F12, F13 and F23.
2. **`HANDOFF.md`.** A header note says that the discipline document and `BANDWIDTH_PLAN.md` supersede the handoff, and that Codex wrote §§7–9. Each of those sections is marked as having no author or lead source (F2–F4).
3. **`end_to_end.md` and `hybrid_lsu/README.md`.** The mismatched-control headline is corrected with the matched-block ratios (F13).
4. **`stream/README.md`, `bank/README.md`, `binding/README.md`.** Quarantine notices added (Q1).
5. **`flow/fpga/run_r41_hbm_stream.sh`.** Now exits with status 3 and points to this audit (F6, Q1).
6. **`tools/r41/compare_torch_rtl.py`.** Refuses comparisons whose block cap, SFU slices, cache sets or home-background setting differ, unless `--allow-unmatched` is given, and then labels the output. Runs recorded before e2d5db8 are attributed their hard-coded SFU counts.
7. **`tools/r41/hybrid_torch_rtl.py`.** Uses the same SFU count for both arms (default 1). `--cache-sets` defaults to the bench geometry (F13, F14).
8. **`python/tnca/backends/multicore_mmio_sim.py`.** The stand-in again defaults to the bench's 4-set cache (F14).
9. **`evidence/r41/acceptance/STATUS.md`.** Correction section for R2–R13.
10. **`evidence/r41/stage1/board/acceptance.before-51f6816.{log,exit-code}`.** The overwritten Stage 1 failure record, restored from 51f6816^ (R11).
11. **`audit/`.** The proposed `r41` revert patch, its check log, and the control-match re-check log.

No commit was reverted wholesale. The violations found are attribution, labelling and control problems layered on real runs. They are not typed-in results like agy's 47dfec5.

## Quarantine list

| ID | Mechanism | Files | Default reachability | Status |
|---|---|---|---|---|
| Q1 | 32-PC stream copy/fill/u32-reduce CU, "native copy dispatch", peer-CU stream runtime | `rtl/fpga/tnca_stream_{axi,bank,fifo}.v`, `rtl/board/tnca_stream_hbm_xo.v`, `fpga/kernel_stream_hbm.xml`, `fpga/u280_multicore_stream.cfg`, `flow/fpga/{package_xo.tcl (stream-hbm), link.sh (TNCA_STREAM_XO), run_r41_hbm_stream.sh, place_r41_hbm_stream.tcl, screen_r41_hbm_{stream,bank}.tcl, screen_r41_core_stream.tcl, r41_core_stream_screen.v, synth_r41_hbm_bank.tcl}`, `python/tnca/backends/hbm_stream.py`, stream hunks of `runtime.py`, `multicore_mmio.py`, `u50.py` (`open_peer`), `torch_device/copy.py`, `tools/r41/{stream_bandwidth_model.py, *stream*, hbm_bank_tests.py, generate_stream_binding.py, audit_stream_package.py}` | Not reachable by default (see the lead section). The run-2 stream link script is now disabled. | Must not be merged into or enabled on `r41`. It is already merged into `r41` (e4770b6), so the revert patch is proposed to the lead. Results do not count for any target. Code kept on `r41-hbm-bw`. |
| Q1a | Plan for "streamed element-wise operator execution" on the stream engine | `STATUS.md` (07eefe3) | Not implemented | Not to be built without the author's written approval of the unit (§2.2.4). |
| Q2 | Locality control kernels (`replay(locality=True)`) | `python/tnca/locality.py`, `torch_device/graph.py` | Off by default | Flagged for the author: it uses the existing NUCA controls, but through hand-built kernels. No gain counts for a target. |

## Valid, reusable results

All are under `evidence/r41/hbm_bw/` unless a full path is given. None is a board measurement, and none counts toward a target until it runs through the standard path with default settings on the board.

**General-path RTL that improves the architecture's own mechanisms:**
- **Hybrid LSU** (shared vector-line planner plus the original narrow path):
  - functional: `hybrid_lsu/*.log`, `hybrid_lsu/coupled/full_vector_matrix.log`, `landing_matrix.log` (455 programs, three configurations), `native_landing/`;
  - routed: 30,429 LUTs, WNS +1.271 ns (`hybrid_lsu/screen_v1/out/`).
  - It serves every vector load and store, so it is general. It is opt-in until the §2.4 gate makes it the default.
- **Compact line LSU:** 4,664 LUTs, routed +2.774 ns (`line_lsu/`).
- **Rejection of the general 8-unit LSU bank:** 440,741 LUTs (`lsu_bank/synth_wide8/`). A valid negative result.
- **Wide home/cache slice:**
  - 512-bit words, atomics and TMU positioning (`wide_home/tests.log`, `special.log`);
  - routed 41,279 LUTs, +1.774 / +0.007 ns (`wide_home/physical/`);
  - fast-fill (`wide_home/fast_fill/`, routed +1.254 ns in `physical_fast/`);
  - background reads (`wide_home/background/`, routed +1.411 ns in `physical_background/`).
  - These are general home-slice mechanisms. The standalone service figures (0.266 / 2.104 / 3.126 GB/s hot, 0.4658 → 0.7136 → 0.9529 GB/s cold) are RTL-fixture results with synthetic DRAM at an assumed 100 MHz. They locate the cold miss-service bottleneck that G3/G4 must remove.
- **Wide NI and mesh:** 720-bit packets with the default layout preserved (`wide_noc/tests.log`; echo endpoints).
- **Parallel SFU slices:** exact-bit at 2, 3, 4 and 8 slices; routed +3.334 ns (`sfu/`). The 257-element SFU4/SFU8 control shows no gain beyond 4 slices.
- **Pipelined outlet issue and the striped 32-PC layout:**
  - outlet: `outlet/line_axi_tests.log`, routed 32-slot +1.658 / +0.011 ns;
  - layout: `layout/*.log`.
  - Both are general mechanisms in the core outlets and host DMA, and both are off by default.
- **Integrated kernel and tile screens:** `kernel_wide/` (tile0 SLR1 +0.508 ns, 63.4% LUTs) and `kernel_background/` (three tiles pass). Out-of-context timing and area only.

**Measured ceilings and negative results (RTL fixture or model):**
- Standard torch numerics through the three-core wide RTL path: `hybrid_lsu/coupled/torch_hybrid_pass.log`, `torch_legacy_pass.log`, `torch_observed33*`.
- Matched-block torch RTL ratios at 16,384 fp16 (`hybrid_lsu/coupled/torch_fp16_16384_comparison_block{16,128}.json`, re-checked in `audit/control_match_recheck.log`).
  - Add, mul and copy do not use the SFU, so for them the remaining SFU-slice difference should not matter; silu and gelu are not matched.
  - Block16: add 1.763x, mul 1.768x, copy 1.543x.
  - These are the honest current numbers for the general path in this fixture, and far from the target.
- Background home reads are neutral on operators (`torch_fp16_16384_background.log`). Block128 regresses four of five operators at 257 elements.
- SystemC: the ordinary-core path saturates near 4 GB/s even with eight services per core (`bandwidth_closure/systemc_v3_*`; uncalibrated, provenance incomplete). The default-equivalence rerun passes.
- Ramulator2 per-PC bounded-admission sweep (`model/`; uncalibrated).
- Board baseline ledger `operator_baseline.json`: 102 points, matching the run-1 roofline record.

**Valid on `r41`:**
- Run-1 board and physical acceptance archives (061dd27, 95562c6).
- Stage 1 original suite 113/113 (ba87220, cdc2caf).
- P0 CPU references (c79f44b).
- Memory-service TLM cycles (bbbf607).
- Pipeline TLM/RTL bitwise evidence (1cfd3b8, 8b62527).
- Chat TLM VL greedy v8 under the fixed 0.05 limit.
- The P8 and P2/P4 board data, subject to the attribution and environment caveats above.

## Reproduction

```sh
git log --first-parent origin/r41-hbm-bw            # audited range
git log --no-merges --since=2026-10-26 origin/r41   # audited r41 range
git log -S'50 GB/s-plus' --all                      # F1: only f0e48e1
git diff 47dfec5^ ea2372f                           # R1: empty
cat evidence/r41/hbm_bw/audit/r41_stream_quarantine_check.log
cat evidence/r41/hbm_bw/audit/control_match_recheck.log
python3 -m tools.docmaint check
```
