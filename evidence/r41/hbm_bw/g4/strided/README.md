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
