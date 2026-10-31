# Gate of the proposed default change (`TNCA_TILED_POINTWISE` 0 → 2)

**Domains: compile-only audit, RTL simulation and unit tests on GB10.** Nothing ran on a card. The change is **not applied** on the branch. It is `../default_change_tiled_pointwise.patch`, and the decision to apply it belongs to the lead and the author.

Trees: base `2a134be`; head `2a134be` plus the patch (default 2 in `compiler/python/tnca_cc/inductor.py` and `compiler/python/tnca_cc/triton_backend/compiler.py`). Script: `gate_g4.sh` (discipline §2.4.1 lanes). `gate-logs.tgz` (sha256 `b8aefd0e…`) holds every lane log, both audit reports and the comparison.

## Results
| Check | Base | Head |
|---|---|---|
| Lanes `core`, `core-vrf`, `unit`, `lint`, `top`, `compiler-rtl`, `triton-e2e`, `r40-pipeline-checks`, `r40-multicore-shell`, `r40-grid-dispatch`, `r40-cache-maintenance`, `r40-coherent-host`, `r40-core-parameters`, `r41-xpose8-rtl`, `lsu-reqreg` | all exit 0 | all exit 0 |
| `compiler` (157 unit tests, IR files), `test_pointwise_tiles.py` | — | exit 0 |
| Lane logs, normalised for paths and times | — | same pass/fail lines in every lane. `core` and `r40-pipeline-checks` differ only in build messages: the head tree regenerated test programs and rebuilt a stale testbench. |

**Compile audit** (`default_change_audit.py compare`, the 113 frozen points, 497 kernels):
- Every point has a changed kernel, because every point contains a pointwise kernel.
- 128 pointwise kernels have new code, the tiled lowering.
- 65 pointwise kernels, all 152 reductions and all 152 templates have identical code. They differ only in the recorded compile options, which now carry the new default.
- No Triton source changed. The recorded launch sequence differs in 42 points.

**Performance of the changed code** (from §8.1 of G4_DESIGN.md, not re-measured here):
- **Model, 113-point non-regression set:** geometric mean 1.083, minimum 0.991, none slower by more than 2 %.
- **RTL, 20 registered points on the run-1 image:** geometric mean 1.489, minimum 1.028.

**Not yet evidence for a default change:**
- No board measurement exists.
- The model and RTL runs cover the run-1 image (rtl) and the run-1 configuration (model).
- On the candidate image, mode 2 has been measured only on the 20 points of §8.4.
