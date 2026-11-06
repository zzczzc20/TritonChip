# GB10 regression record for the final-image candidate

**Where and what.**
- Host: GB10 (`gx10-6646`), iverilog 12.0 and yosys 0.33 from `~/tc_env`, at
  most 4 concurrent lanes under `nice`.
- Source: a copy of branch `r40-final-image` at `~/tc_hw_final`. Its `rtl/`
  tree hash (`find rtl -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`)
  is `0398cd0fdf88196363048b686e42992d38f6d79c393474854c32cacf4aa94af0` on
  both GB10 and the worktree.
- RTL timing: every RTL edit was made before the first run; later syncs changed
  only `tb/` and `tools/`.
- Baselines: the attempt-3 RTL is `git archive d3c215d rtl` at `~/tc_hw_base`.
- Logs: `logs/gb10_regression_logs.tgz`, including the batch scripts.

| Lane | Command | Result |
|---|---|---|
| Lint | `make lint`; elaboration of `rtl/board/tnca_multicore_hbm_xo.v` | 68/68 files; board top elaborates |
| Core | `make -j3 core core-vrf core-slowmem` | 117/117 programs (39 each, incl. `t40_f2i`), EXIT 0 |
| Core OOO / v5 / unit | `make -j3 core-ooo core-r39sw5 unit` | EXIT 0; core-ooo 39/39; core-r39sw5 fast + ooo; unit: bitops, mul64, fp32_fma ×2, decoder 773, cvt_xls 866,495×2, **vector_fpu_f2i 181,632**, sfu_xls, sfu_core, fpu_multi, asm round-trip |
| FPU oracle vs attempt-3 | `tb_vector_fpu_f2i` built against `~/tc_hw_base` `tnca_vector_fpu.v` | 18,289 of 181,632 differ, which confirms the oracle is sensitive to the defect |
| Host mux | `tb_idle_host_mux` (directed + probe + 20,000 random cycles) | pass: 1,837 grants; the attempt-3 mux fails the `c_ready` probe |
| Coherent host | `python3 tools/r40/coherent_host_tests.py` | pass (with the `tnca_top` grant assertion live) |
| Memory controls | `python3 tools/r40/memory_control_tests.py` | 16/16 steps pass |
| Native four configurations | `memory_control_tests.py --native --python ~/tc_env/torchenv/bin/python` | pass: outlets/outstanding 3/8, 1/8, 1/1, 3/1, numerically correct |
| Image geometry | `python3 tools/r40/image_geometry_tests.py` (attempt-3 `image.tcl`) | pass, EXIT 0 |
| Cone cuts | `python3 tools/r40/yosys_cone_cuts.py --baseline-root ~/tc_hw_base` | 3/3 cut on the candidate; chain present on attempt-3 |
| NUCA / top / mesh | `make top nuca mesh nuca-dco`; `nuca-dco` with `DCO_REGISTERED_RANGES=1`; `nuca-dco DCO_POLICIES='2 74' DCO_SEEDS='1 7'` with registered ranges, victim owner, epoch churn; `tb_tmu_registered_ranges` PORTS 3, 8 | EXIT 0 |
| R40 RTL | `make r40-grid-dispatch r40-multicore-shell r40-grid-deadline r40-core-parameters` | pass (20 dispatch configurations; multicore shell N=1/3/8; core parameters 6/6) |
| R40 memory | `make r40-memory-ports r40-line-axi r40-cache-maintenance r40-multicore-dma` (torch env) | EXIT 0: memory ports 15 protocol + 7 core configurations and native ports; line AXI 35 configurations; maintenance 19/19; DMA ownership on the plain and board tops |
| MMIO lanes | `multicore_mmio_tests.py` 1/1, 3/3, 3/3 `--pipeline 2 --board-top` | pass |
| Native lanes | `native_memory_ports_tests.py`, `native_multicore_features.py`, `--pair` (fpga-multicore-sim, 3 cores, pipeline 2, board top) | EXIT 0 each |
| Grid / runtime | `make r40-physical-grid-transport r40-physical-grid-runtime r40-multicore-backend`; `make r40-multicore-native-smoke` (3 and 8 cores); `make r40-multicore-mmio-checks` | EXIT 0 each |

**Process notes.**
- The first runs of the torch-based lanes used the Triton-only interpreter
  (`$TRITON_PYTHON`, which has no torch) and stopped at `import torch`. They
  were rerun with `~/tc_env/torchenv/bin/python` and pass. Both logs are kept
  (`C_r40_rtl.log` and `F_r40_mem.log`).
- Not run here: the software-only CI lanes (compiler, tilebench, r37/r38, which
  are unaffected by these RTL files), `r40-physical-chunks`, and
  `r40-pipeline-checks`. GitHub CI on the exact frozen commit remains required
  before the link.
