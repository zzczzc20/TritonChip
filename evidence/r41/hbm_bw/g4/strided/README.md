# Strided-point investigation

Domain: RTL simulation, not a board result. The binding rules are in `doc/engineering_discipline.md`. The source tree for the runs is `8f19b1131c8f8b4b454831b6317d772774f93383` (GB10 directory `~/r41-codex-t9b`). The board-top bench uses the synthetic AXI line model, not physical HBM; no board image or clock measurement is involved. The two point definitions are the registered fp16 small stride-2 add and clone. The compiler setting is the default, with `core-vrf-sync-v5-x8-g2` for all arms. The arm files in `images/` are diagnostic configurations, not delivered images or board performance claims.

`worker.sh` runs the existing `tools/r41/link_budget_rtl.py` for each arm and point, recording timestamped starts, exit statuses, logs and JSONL independently. Four workers were started on GB10 on 2026-10-29 UTC, with output under `~/r41-codex-t9b-results/`; their shell PIDs are 2710277–2710280. The first jobs reproduce G2 and s4, followed by the listed single changes and the s1/s2 combinations. No result is claimed until the corresponding log and record are retrieved. S4's HBM striping is not represented by this bench's synthetic small memory; it requires the board-sized address window, as documented in `G4_DESIGN.md`.

## Existing evidence (RTL simulation)

`python3 evidence/r41/hbm_bw/g4/strided/history.py` recomputes the comparison from `../rtl_s4/`. `history.tgz` preserves the relevant original JSONL and worker logs, plus its computed summary (SHA-256 `c402412c377de4b489ad8cf1b854f555a51b2199f175a40bcdf6dc25dcc03865`). The source and software hashes are in the records. The G2 and s1 arms have the same recorded software tree; s4 has a different software tree, so the latter comparison is not by itself a controlled attribution.

| RTL point | G2 cycles | s1 cycles | s1 change versus G2 | s4 cycles | s4 change versus G2 |
|---|---:|---:|---:|---:|---:|
| fp16 strided add | 168,017 | 202,715 | +20.65% | 186,522 | +11.01% |
| fp16 strided clone | 109,935 | 130,340 | +18.56% | 120,237 | +9.37% |

For both points, s1 combines hybrid LSU, background reads and no-fill, so the observed regression cannot yet be assigned to any one of them. Summed RTL LSU-stall counters and hazard counters increase in s1 while memory read/write counts remain almost the same; the original probe does not establish the cause. The new matrix tests the changes separately and repeats G2 and s4 on the same current tree.

For the new matrix, `python3 evidence/r41/hbm_bw/g4/strided/report.py 'PATH/*.jsonl'` summarizes each completed record, including failures, software-tree identity, per-point G2 comparison and summed probe counters. A missing result is not a pass.

## New runs (RTL simulation)

G2 strided clone passed its numerical check at 110,197 device cycles (one timed sample; `clock_hz` is unspecified by the RTL harness). The raw JSONL, worker output and register-bench trace are in `g2-clone.tgz`, SHA-256 `95cf2beb7bfc86ab3dcee813db08126f593598915577f218136dd634a62fe442`. It used software tree `files:cb25a432a4544b5cc1082550237f90a0d89d0d465e58f66a29aa063e4648d6dd`, source commit above and the G2 diagnostic image; no board image hash applies. The remaining arms were still running when this record was written.

S4 strided clone passed at 118,007 RTL device cycles, 7.09% more than the new G2 clone sample; both records carry the same software-tree hash and default compiler settings. The raw JSONL, worker output and register-bench trace are in `s4-clone.tgz`, SHA-256 `679b249400ffed32e00a336e6ed31163784c833e19704be2d310e9d84b23166d`. In these samples, the sums over the three RTL tiles show the same 3,087 memory line reads and 768 writes in each arm, while LSU-stall cycles sum to 131,035 on G2 and 169,089 on s4. These counters do not isolate the responsible change; the one-change runs remain necessary.

### Hybrid path observation

The board testbench additionally prints the existing hybrid LSU counters at the same stage-probe boundaries when `HYBRID_LSU` is enabled. The signals are read-only; the print is absent when stage probes are disabled. Elaborations on GB10 for both `HYBRID_LSU=0` and `HYBRID_LSU=1` exited 0 using `iverilog -g2012` with the board-register bench, one tile and one port; raw output is `probe-elaboration.tgz` (SHA-256 `b967a12a9c7aeba313e4a32947a982445f5122c2a529f4c4207f3078e152a285`). This checks elaboration only, not the counter values or functional correctness. A probe run uses the source at `9714e405965337d4cf66ac46c095ad5c4b599f79` plus this observation-only testbench change, in a separate GB10 checkout so that the original matrix remains unchanged.

G2 strided add passed its numerical check at 167,351 RTL device cycles. The raw JSONL, worker output and register-bench trace are in `g2-add.tgz`, SHA-256 `e37d8949390e67cfbead0a206486dd8ee3b10626248991ab97fe63f70e05b494`. It has the same software-tree hash as the new clone records. The corresponding s4 and the single-change comparisons were still running when recorded.

Probe elaboration command (GB10, after `source ~/tc_env/env.sh`):

```sh
for hybrid in 0 1; do
  iverilog -g2012 -I rtl/common -I rtl/noc -s tnca_multicore_board_regs \
    -Ptnca_multicore_board_regs.N=1 -Ptnca_multicore_board_regs.PORTS=1 \
    -Ptnca_multicore_board_regs.MEMORY_BYTES=65536 -Ptnca_multicore_board_regs.HYBRID_LSU=$hybrid \
    -y rtl/common -y rtl/core -y rtl/gen -y rtl/nuca -y rtl/noc -y rtl/tile -y rtl -y rtl/fpga -y rtl/board \
    -o /tmp/t9b-probe-$hybrid.vvp rtl/common/*.v tb/common/tnca_axi_line_model.v tb/fpga/tnca_multicore_board_regs.v
done
```

The read-only observation run was started on GB10 in `~/r41-codex-t9b-probe`, shell PID 2735285, using:

```sh
~/tc_env/torchenv/bin/python tools/r41/link_budget_rtl.py \
  --image evidence/r41/hbm_bw/g4/strided/images/hybrid.tcl \
  --profile core-vrf-sync-v5-x8-g2 \
  --point op.clone/membound/fp16-small-strided-r32-c1024 \
  --out ~/r41-codex-t9b-probe-results/hybrid-clone.jsonl
```

S4 strided add passed at 183,876 RTL device cycles, 9.87% more than the new G2 add sample; both records carry the same software-tree hash and default compiler settings. The raw JSONL, worker output and register-bench trace are in `s4-add.tgz`, SHA-256 `6f91db512635e1289db90529f270ae3510470f7c38e49d387c131489508ac348`. RTL tile sums: G2/s4 memory reads 5,142/5,143, writes 862/861, LSU-stall cycles 187,034/196,916, and hazard-stall cycles 143,940/224,262. These single samples reproduce the slowdown but do not isolate its cause.

With only `HYBRID_LSU=1` added to G2, strided clone passed at 131,466 RTL device cycles, 19.30% above the new G2 sample on the same software tree. The original 3,087 memory line reads and 768 line writes are unchanged, while summed tile LSU stalls rise from 131,035 to 195,039 cycles and hazard stalls from 45,346 to 58,559 cycles. This isolates hybrid LSU as a sufficient cause of a slowdown in this sample, but does not yet show the internal mechanism or quantify the other changes' effects. Raw JSONL, worker output and register-bench trace: `hybrid-clone.tgz`, SHA-256 `ed25c75028685ea24e836a8c3e16e674f9561d2a6ccf5be368c8c2e4ecfed64c`.

The core asserts the hybrid LSU's `req_valid` only when its pending command can be consumed. Consequently the hybrid module's own `perf_switch_stall` does not count cycles where the core holds a pending command waiting for readiness; a zero there does not prove that no switch cost exists. The testbench now also counts pending-but-not-consumed cycles and the subset for which the selected mode is unavailable. Both hybrid settings again elaborated successfully with the command above (output filenames `t9b-probe2-$hybrid.vvp`); `probe2-elaboration.tgz` SHA-256 `f2a7ecc297edf3c35632b86afc8b0b8c952a13744f393029872572359932551a`. The first probe run used the earlier counter-only testbench; these extra counters require a separate run.

The s1 combination (hybrid LSU, background reads and no-fill) passed strided clone at 130,422 RTL device cycles, 18.35% above G2 and 0.79% below hybrid-only in these single samples. Its raw record, worker output and register-bench trace are in `s1-clone.tgz`, SHA-256 `d3caa056d3fca1459b5431d7e4f12673ed02a30910667663fa6211446e82599f`. The sample comparison is not a noise estimate.

The first observation-only hybrid clone run passed at 131,383 RTL device cycles; its JSONL, log and complete register-bench trace are `probe1-clone.tgz` (SHA-256 `67b8308407f343242966d926d900fa2f083779f8b1097e718a991982223214a8`). As explained above, its module-level switch counter cannot measure pending-before-handshake waits. The revised observation run was started on GB10 in `~/r41-codex-t9b-probe2`, shell PID 2769964, with the same command and image as the first probe and output in `~/r41-codex-t9b-probe2-results/`; its source is commit `9714e405965337d4cf66ac46c095ad5c4b599f79` plus the testbench changes in commits `09ae45a0` and `6e877c47`.

With only `HYBRID_LSU=1` added to G2, strided add passed at 205,152 RTL device cycles, 22.59% above the new G2 sample on the same software tree. G2/hybrid tile sums: memory line reads 5,142/5,143, writes 862/862, LSU stalls 187,034/225,790 and hazard stalls 143,940/255,649 cycles. Raw JSONL, worker output and register-bench trace: `hybrid-add.tgz`, SHA-256 `df66470042cfeaa7eb70dd9d7f9faf18d74485437b02e2cc0c02d018461a7a43`. Thus hybrid LSU alone is sufficient to produce a slowdown for both points in these samples; internal cause and the impact of the other individual settings are still being examined.

Adding no-fill to hybrid LSU yielded 131,516 RTL device cycles for strided clone, versus 131,466 with hybrid alone (50 cycles, 0.038% higher); both passed and each recorded 3,087 line reads and 768 writes. These are single samples, not a measured noise range. Raw JSONL, worker output and register-bench trace: `nofill-clone.tgz`, SHA-256 `ef20e127ef39778c45e2f61203902cf61eef80ab488fd28f1b56c90c931e1505`.

Adding the pipelined home to hybrid LSU gave 117,946 RTL device cycles for strided clone (pass), versus 131,466 with hybrid alone and 118,007 with full s4. These single samples show the pipeline improving this point relative to hybrid-only; they do not establish a noise range. Raw JSONL, worker output and register-bench trace: `pipe-clone.tgz`, SHA-256 `8c1d5b24d109ecf40cf4172e9bf963b4c13e84a2dcab157ff3254a3aa350c43c`.

### Candidate under test

The hybrid path reserves tag bits for the outer client, fast/slow path and narrow-pair unit; the current tile hard-limits the narrow pair to four beats per unit even when wider tags have spare capacity. `candidate-beats.patch` (SHA-256 `90839ed707ff15f52ae81038fb63e041d86d8f5af8c2b2698e5bb6572d64c0d0`) proposes choosing up to the configured beat count subject to the actual tag width, keeping the original four-beat limit when the width only permits four. It is a candidate under test, not a validated fix. The default non-hybrid path is unchanged. Performance, timing and full regressions have not yet been established.

The revised observation run passed strided clone at 131,086 RTL device cycles. Across its three tiles in the timed probe window, 2,048 fast and 6,144 slow commands retired; the pending-command wait summed to 204,940 cycles, of which 30 were counted with the selected fast/slow mode unavailable. Thus the mode-drain condition measured here does not explain most of the pending wait; its remainder includes readiness/resource waits and needs the candidate comparison. The raw JSONL, stdout and complete stage-probe trace are in `probe2-clone.tgz`, SHA-256 `d9191a1059aa7066e1e634c43e51709d9d875b501435c1d5299618b00b788291`.

The candidate's s4 clone comparison was started on GB10 in `~/r41-codex-t9b-fix`, shell PID 2793044; output is in `~/r41-codex-t9b-fix-results/`. That tree uses the source at `9714e405965337d4cf66ac46c095ad5c4b599f79`, the observation-only testbench changes in `09ae45a0` and `6e877c47`, and `candidate-beats.patch`. Its exact run command is `tools/r41/link_budget_rtl.py --image evidence/r41/hbm_bw/g4/strided/images/s4.tcl --profile core-vrf-sync-v5-x8-g2 --point op.clone/membound/fp16-small-strided-r32-c1024 --out ~/r41-codex-t9b-fix-results/s4-clone.jsonl`, with `~/tc_env/torchenv/bin/python` after sourcing `~/tc_env/env.sh`. It subsequently completed; the result is recorded below.

The s1 combination passed strided add at 203,195 RTL device cycles, 21.42% above G2 and 0.95% below hybrid-only in these single samples. Raw JSONL, worker output and register-bench trace: `s1-add.tgz`, SHA-256 `4e6124402fd7c7710f5c5d06a4c1c1a7fcae40a01bb711f11ac4cbcfe5d78925`.

Candidate verification so far (RTL simulation, GB10): the existing `python3 tools/r41/lsu_hybrid_tests.py` matrix passed all 48 configurations and its reset test. That run used the candidate RTL and the initial directed-test variant; the 48 existing configurations retain their original four-beat candidate setting. Separately, the strengthened directed test and a mixed, backpressured reference comparison each passed at `BEAT_W=5,6,7` with `CAND_BEATS=8`. The directed test observed 16 simultaneously outstanding narrow tags, filling both units, and the miter checked completions and memory against the original pair. `candidate-tests.tgz` (SHA-256 `c16cb8422c10523aeafaf2d4300af0c77d3647971740d932513a33c3f19b211c`) preserves both sets of raw output, including the initial directed run. `candidate-tests.patch` (SHA-256 `8248739d91d68c547db28e34d14468e100f3452377d8a7cc3c140cd31ecca1e5`) contains the final test changes and the repeatable runner invocations. These tests do not establish board timing or performance.

A second s4 candidate run for strided add was started in the same GB10 candidate tree, shell PID 2795978, using the same command as the candidate clone with the point changed to `op.add/membound/fp16-small-strided-r32-c1024` and output `~/r41-codex-t9b-fix-results/s4-add.jsonl`. Both candidate point runs subsequently completed; their results are recorded below.

Enabling background reads alone on G2 passed strided clone at 108,639 RTL device cycles, 1.41% below the G2 sample; a single sample does not establish a noise range or causal improvement. Raw JSONL, worker output and register-bench trace: `background-clone.tgz`, SHA-256 `3a2ec5ac1f27f33738ecf113aba31cf5fbcb6fd02d1c5f45120e112283bd5bf7`.

With hybrid LSU and `LINE_SLOTS=32` on the original RTL, strided clone passed at 131,381 RTL device cycles, close to the hybrid-only sample of 131,466. The original tile still caps the narrow pair at four beats despite the wider tag; this run precedes the candidate. Raw JSONL, worker output and register-bench trace: `slots-clone.tgz`, SHA-256 `e6e9c398afcb335ed24dc93acc9c85a68346e093ae011ef2b4a402966047f31e`.

The candidate s4 strided clone passed at 90,189 RTL device cycles, 23.57% below the original s4 sample and 18.16% below the G2 sample. The three-tile LSU stall sum fell from 169,089 in original s4 to 82,187, while line reads/writes remained 3,087/768. This is a single simulation sample, not a board result or noise characterization. The candidate record has a different software-tree fingerprint (`files:5014abefbf3d1527c5011162b90678b9274dc6df91c09982844e4e6de50c7512`) because it includes the RTL and testbench changes; `git diff 8f19b113 9714e405 -- compiler/python python/tnca archbench tools/r41/link_budget_rtl.py rtl` is empty, and the candidate tree adds the recorded testbench and RTL patches to that source. Raw JSONL, worker output and register-bench trace: `candidate-s4-clone.tgz`, SHA-256 `b8feb72b579fc100fb81482e59c232369a5b01583e10dfdc7bde379090d68f0d`. The add comparison and full non-regression gates remain open.


## Completed matrix and candidate (RTL simulation)

All 22 diagnostic records and both candidate records passed their numerical checks. `all-runs.tgz` (SHA-256 `7c8d952c520ef1ff0c1dd2411b2b7654ccecb8161b0186945ef7dcd410853cad`) contains the worker logs, all JSONL records, each recorded register-bench trace, a computed summary, and compiler build/test logs. These are single samples on the synthetic-memory board bench, not board results or a noise characterization. The source and non-default diagnostic geometry are recorded in each JSONL; the synthetic bench has no board image hash and its cycle records leave the clock unspecified.

| RTL configuration | add cycles | clone cycles |
|---|---:|---:|
| G2 | 167,351 | 110,197 |
| G2 + hybrid LSU | 205,152 | 131,466 |
| G2 + background reads | 164,543 | 108,639 |
| hybrid + no-fill | 205,085 | 131,516 |
| hybrid + 32 line slots | 205,386 | 131,381 |
| hybrid + home pipeline | 184,273 | 117,946 |
| G2 + 32 MSHRs | 166,653 | 110,166 |
| G2 + 32 outlets and pipelined issue | 167,236 | 110,052 |
| s1 | 203,195 | 130,422 |
| s2 | 183,946 | 118,137 |
| s4 | 183,876 | 118,007 |
| s4 + candidate | 135,109 | 90,189 |

The outlet arm changes depth and pipelined issue together, so it does not isolate those two parameters individually. HBM striping is not represented by this small synthetic memory model. The candidate reduces s4 cycles by 26.52% for add and 23.57% for clone; compared with G2 it is 19.27% and 18.16% lower, respectively. The one-change runs show that hybrid LSU is sufficient to create the regression, while the home pipeline reduces it. The pending-wait probe rules out mode-drain wait as its principal source for clone in the measured window. The fixed four-beat narrow-unit limit, despite widened available tags, is the identified design limitation; the candidate's controlled correction and lower LSU stalls support this attribution.

On the candidate snapshot, `R1_STRICT=1` `make compiler` passed 157 unit tests and six IR RUN cases, after building `tnca-opt` and `sim/build/tb_core.vvp`; its logs are in `all-runs.tgz`. Those results predate the later upstream merge and do not validate unrelated later changes. The remaining gates and final integration are recorded below when completed.

After merging upstream through `7d3b2864`, strict `make compiler` stopped before the unit tests because the generated `doc/support_matrix.md` lacked the upstream `core-vrf-sync-v5-x8-g2c4` profile. The failed log is preserved in `compiler-stale.tgz` (SHA-256 `d8036de2a5a96f9896ff73bbb376e2f2c9ecaa51266c09ca434a236b352ed5f7`). The matrix was regenerated by `python3 tools/isa/gen_support_matrix.py`; the strict gate must be rerun on the regenerated tree.

The outlet depth and pipelined-issue changes were initially measured together. To isolate them, `outlet_depth.tcl` (32 outstanding, original issue) and `outlet_pipe.tcl` (original depth, pipelined issue) were prepared on the original `8f19b113` diagnostic tree, with `outlet_worker.sh` running both registered strided points under default compiler settings. The add worker started on GB10 as PID 2975767; the clone worker awaits an available resource slot. No result is yet claimed for either isolated arm.

After regenerating the support matrix, the merged candidate tree passed strict `make compiler`: 157 unit tests and six IR RUN cases, with `tnca-opt` and the core simulator present. `compiler-merged.tgz` has the raw GB10 output (SHA-256 `6a64a8d6eb2492a2e2bc17ea709f92b353f88829a3684efee431c080ae332fad`); the prior stale-file failure remains in `compiler-stale.tgz`. The source was upstream `7d3b2864` with the recorded candidate and testbench patches plus the regenerated support matrix from `7295f6a7`. The rerun used the torch interpreter, this tree's `PYTHONPATH`, `R1_STRICT=1`, and `TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20`; the exact command is in the earlier compiler log and the core-build and result are in the new archive.

The isolated outlet clone worker was subsequently started on GB10 as PID 2984917. Its depth and issue arms, like the add arms, remain RTL simulations using the diagnostic tree and synthetic memory.

On the merged candidate snapshot, GB10 RTL regression results were: hybrid LSU 48 existing configurations plus the added eight-beat directed and mixed cases, line LSU 48 configurations, wide NI/NoC 12 configurations, and home pipeline 189 cases, all exit 0. The raw logs and invocation script are in `merged-unit-regressions.tgz` (SHA-256 `2413a3263e4caa6ef089fb2d23421dfef0ba3915adbabba777e9ba9d9c740416`). `make compiler-rtl` and the separate no-fill home regression were started afterward; their outcomes are not implied by these unit results.

The merged-candidate `python3 tools/r41/g4_home_tests.py` regression passed all 50 RTL cases; raw GB10 output is `merged-home-nofill.tgz` (SHA-256 `081ff5240b42f81b5d00f7d6367407fe91cfd07f67980f37bf602bfff5698c07`).

The merged candidate also passed `R1_STRICT=1 make compiler-rtl` on GB10: 68 RTL tests, exit 0. Raw log: `merged-compiler-rtl.tgz`, SHA-256 `004a079704d2529d6beac4bb76627e3bbebcf68a03ecd4b221c0bf0f7bf448c8`.
