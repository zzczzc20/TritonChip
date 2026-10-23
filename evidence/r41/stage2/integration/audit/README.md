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
The integrated X8-profile rerun is still pending.

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
The two lanes inherit CPUs 0–5. Their systemd memory limits sum to 15 GiB
(first one combined lane with MemoryMax=15G; subsequently compiler lane 12G,
native lane 3G; subsequently compiler 9G and two native lanes 3G each); swap is disabled within the run cgroups. Thread counts are 1,
including Inductor compile threads; the audit uses `--jobs 2`.

`run_gb10.sh` records the original launch sequence; `run_native_gb10.sh` is
the native-only rerun after the wrapper import correction.
