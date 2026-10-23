# R41 Stage 2 integration: functional-gap audit

Scope: the two checks omitted from the integration regression. All compilation
and RTL simulation ran on GB10. No Vivado, v++, FPGA host or card was used.
Only evidence is changed by this branch.

Sources:

- Integrated: `e843bc4cd18649c1f162fb74911ca2b25d6c4e57`, copied with the common
  rsync exclusions to `~/r41s2-int-audit-src`; `make r2-stage-data` run with
  `PYTHON`, `TRITON_PYTHON`, and `TORCH_PYTHON` set to
  `~/tc_env/torchenv/bin/python`.
- Base: existing `~/r41s2-int-base`, archived from `origin/r41s2`
  (`708a920aaabbc57a307b7cea2dd1a7d31dbdd4bc`), with data already staged.
- Results: `~/r41s2-int-audit-results`. Existing `~/r41s2-int-logs` was read
  for comparison and not written.

## Default-change audit

| Comparison | Base compiled | Integrated compiled | Identical text and launch structure | Changed | Not compared |
|---|---:|---:|---:|---:|---:|
| Base v5 vs integrated v5 | 113/113 PASS | 113/113 PASS | **113/113** | **0** | 0 |

There are no changed points to explain at the delivered default profile.
Raw reports, per-point compiler logs and assembly are in `default-logs.tgz`;
`default-compare.json` is the unchanged harness's comparison output.

| Comparison | Both runs compiled | Identical | Changed | Not compared |
|---|---:|---:|---:|---:|
| Integrated v5 vs integrated v5-x8 | 113/113 PASS each | **0/113** | **113/113** | 0 |

**The declared-profile capability-only criterion is NOT met.** This is a
compiler scheduling-model selection defect, not X8 byte-transpose lowering:
`spec/r39_sched_model.json:by_target` names `tnca-sim-vrf-v5` (`banked2`)
but omits `tnca-sim-vrf-v5-x8`. `Model.for_target()` in
`compiler/python/tnca_cc/schedule.py` consequently selects the default
`flipflop` model for X8. The two targets have the same banked register-file
architecture; changing that model is not driven by the new capability.

Every one of the 113 points changes code text, covering 479 launched kernels.
All 479 retain identical Inductor Triton source hashes and imported TNCA IR.
All emitted pre-scheduler optimized IR (170 kernels) and SPMD IR (433 kernels)
are also byte-identical; the other kernels omit those optional dumps on both
sides. **Scheduled IR differs in all 479 kernels.** These stage boundaries
attribute every changed kernel to scheduler selection, without changing
product files or running a counterfactual product tree.

The resulting allocation changes `rf_private_per_role` in 117 kernels across
47 points. All 113 launch sequences/grids and all other launch-structure
fields are unchanged. For example, `op.add/tiny/fp32-n256` still emits 200
bytes / 25 instructions, but reorders independent loads and the vector
constant, renames registers, and reduces private vector registers from 4 to
3 per role.

The frozen suite contains 91 FP16 and 22 FP32 points, with no FP8 points.
Neither profile emits `t.ld.async.b.t` or `v.ld.gather.b` in these launched
kernels. Thus there is no byte-transpose replacement to explain here.
The actual X8 capability remains gated in `_transposed_tile_load` and
`_xpose_geometry_ok`; this finding is the unrelated target-name fallback.

Per-point and per-kernel stage hashes and byte deltas are recorded in
`x8-attribution.json` and `x8-compare.json`; raw logs, all assembly and representative paired
IR dumps are in `x8-logs.tgz`. Full dumps remain on GB10; the archive keeps
IR samples small while retaining stage hashes for every changed kernel.
The following table exhausts all 113 changed points; the reason for every row is the model selection described above.

| Family | Points with text changes | Points with resource changes |
|---|---:|---:|
| `model.gpt2_block` | 4 | 4 |
| `model.llama_decode` | 5 | 5 |
| `model.llama_prefill` | 5 | 5 |
| `model.mlp` | 4 | 0 |
| `model.resnet_block` | 4 | 0 |
| `op.add` | 6 | 6 |
| `op.argmax` | 4 | 0 |
| `op.attention_decode` | 4 | 4 |
| `op.attention_prefill` | 4 | 4 |
| `op.bmm` | 4 | 3 |
| `op.conv2d` | 4 | 0 |
| `op.copy` | 1 | 1 |
| `op.cumsum` | 3 | 3 |
| `op.embedding` | 5 | 5 |
| `op.gelu_tanh` | 4 | 0 |
| `op.gemv` | 5 | 0 |
| `op.layernorm` | 4 | 0 |
| `op.linear` | 5 | 0 |
| `op.matmul` | 9 | 0 |
| `op.rmsnorm` | 5 | 0 |
| `op.rope` | 4 | 4 |
| `op.silu` | 5 | 0 |
| `op.softmax` | 6 | 0 |
| `op.swiglu` | 4 | 0 |
| `op.transpose` | 5 | 3 |


Compiler/profile follow-up is required before claiming capability-only
changes for the image's declared profile. No product fix is included on
this evidence-only branch.

Content identities (also checked after each run):

| Tree | Product content hash |
|---|---|
| Base | `files:1739604e16fb42525d4d590937fbac8b77852c4bc11eb12746097aba0d042c22` |
| Integrated | `files:958b197fbd80c6bfbb3844bf77bb20d997f47b059f3e8453f032880ac356367a` |

The original `tools/performance_harness/default_change_audit.py` is used
for both trees, SHA-256
`1afa7bab3f0d225e7e424e976391ddb0f1214c946fdb030a9a77a6530dcfe0e3`.
`run_audit.py` sets the imported preflight harness's profile before invoking
that unchanged script; children receive the selected profile in their environment.
The ordinary runs select `core-vrf-sync-v5`; the second integrated run selects
`core-vrf-sync-v5-x8`. All other harness settings and compile knobs remain
those of the Stage 1 audit. Each point has a fresh process and private caches;
all launches are skipped, device sessions are forbidden, and host fallback
is checked. This measures compiled text and launch structure, not numerics or
execution performance.

## Native Torch on board-top multicore RTL

Pending. `run_native.py` imports Torch and Triton, opens an outer launch log,
then runs each unmodified test script with its original assertions. It records
warm and steady launches, including cycles, kernel names, grids, and host times.
Each test gets private compiler caches. The three requested recipe rows run
on both trees with 3 cores and pipeline 2; the ports row uses 3 memory ports.
These scripts select `core-vrf-sync-v5`, as the Makefile recipes do.

`--board-top` elaborates the actual coherent FPGA wrapper through
`tnca_multicore_board_regs`, with AXI line models. Its default cache is
SETS=4/WAYS=2 and its convergence-option defaults are retained. This is the
requested functional recipe, not an elaboration of every `image.tcl` parameter.

The first wrapper attempt imported the TNCA driver before Triton and failed
in Triton's backend discovery, before any native test executed. The import
order was corrected; the startup failure logs are retained separately.

## Resource bounds and reproduction

`free -g` was checked before each run and required at least 40 GiB available.
All lanes inherit CPUs 0–5. Their systemd memory limits total at most 15 GiB
(first one combined lane with MemoryMax=15G; subsequently compiler lane 12G,
native lane 3G; subsequently compiler 9G and two native lanes 3G each); swap is disabled within the run cgroups. Thread counts are 1,
including Inductor compile threads; the audit uses `--jobs 2`.
After the compiler lane finished, the remaining pair/ports cases were run as
four independent 2G services, alongside the integrated features service
(3G). The original native shell schedulers were stopped by PID to prevent
duplicate scheduling; their active simulations were allowed to finish.

`run_gb10.sh` records the original launch sequence; `run_native_gb10.sh` is
the native-only rerun after the wrapper import correction.
