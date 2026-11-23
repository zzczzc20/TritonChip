## 8. Files, commands, hashes

| path | what |
|---|---|
| `rtl/core/tnca_kfuse_lane.v` | the lane (plus `tnca_kfuse_prod`, `_maxtree`, `_addlvl`, `_addtree`, `_dly`); Verilog-2005; uses the generated `tnca_mac_b0_xls`, `tnca_fmad_norm_xls`, `tnca_fma_round_xls` unchanged |
| `tools/r42/kfuse_rtl_vectors.py` | vector generator (the model's `scalar_step` is the only source of expected values) |
| `tools/r42/kfuse_rtl/tb_kfuse_lane.v`, `run_sim.sh`, `sim_matrix.sh`, `mutation_check.py`, `result_classes.py` | bench, runners, bench sensitivity, result classes |
| `tools/r42/kfuse_rtl/ooc.tcl`, `ooc_tops.v`, `run_ooc.sh`, `run_queue.sh`, `netlist.tcl`, `run_netlist.sh`, `ooc_summary.py`, `stage_slack.py` | OOC flow and its tops, netlist export |
| `tools/r42/kfuse_rtl/collect.sh`, `make_tables.py`, `build_report.sh` | evidence collection, tables, this report |
| `evidence/r42/kfuse_rtl/ooc/<run>/` | final-RTL Vivado reports (`*.gz`: utilisation, hierarchical utilisation, timing summary, 30 worst paths, worst path into every register, log) |
| `evidence/r42/kfuse_rtl/ooc_v1/<run>/` | the same for the first RTL |
| `evidence/r42/kfuse_rtl/sim/` | per-run logs (`f_*.log`), vector generator logs and sha256 of every vector file, `list_f.txt`, `final_tool_sha256.txt`, mutation check, result classes |
| `evidence/r42/kfuse_rtl/sim_v1_partial/` | the interrupted first matrix (first RTL), kept, not counted |
| `evidence/r42/kfuse_rtl/netlist/` | netlist-simulation logs, utilisation of the synthesised netlists, the first run's GSR artifact |
| `evidence/r42/kfuse_rtl/tables.md`, `commands.txt` | generated tables; every command |

The vector files (up to 90 MB each) are not kept: they are a pure function of `(G, W, N, seed)` and the model, and their sha256 are in `sim/*.sha256`.
Source hashes at the time of the final matrix: `sim/final_tool_sha256.txt` (the bench used for the matrix is `c3ef2fe7...`; the bench in the tree adds only a `ifdef NETLIST` delay, used by the netlist runs).
Environment: iverilog 12.0 (GB10, aarch64), Vivado 2023.2 (x86 host, Ubuntu 24.04 WSL2, 16 cores).
