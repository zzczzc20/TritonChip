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
