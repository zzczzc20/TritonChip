# Default-on compiler changes since R40: compile-only code audit

## Current state: broadcast fix narrowed (`default_change_audit_narrowed.json`)

The `[R,1]` column-broadcast fix now copies lane 0 only when the column is
not proven row-replicated (`Tile.replicated`, set from how the importer
materialises a tile: affine index columns with no column stride, dense
constants, splats of scalar registers, `_vector_to_col` rows, and lane-wise
operations, comparisons and conversions whose operands are all replicated;
loads and everything else stay unknown and keep the copy). Rerun on the same
harness, knobs and points, tree `files:8c713902…` (bf6fd2b + the narrowing):

| | points |
|---|---:|
| code text and launch structure identical to R40 | **113** |
| changed | 0 |

All 56 points the fix had changed return to R40's exact code text and launch
structure; none is left (no original point broadcasts a lane-0-only
column). Every one of the 183 previously changed kernels now records
`column_broadcasts: {lane0: 0, replicated: n}` in its provenance.

The fix still fires where it is needed. The depthwise graphs of
`tools/performance_harness/test_depthwise.py` record 1-3 lane-0 copies per
convolution kernel (`kickoff/narrowed-broadcast-v1-evidence.tgz`,
`probe/column-broadcast-decisions.json`), and with the copies forcibly
skipped the FP16 stride-1 dilation-2 case fails on SystemC (24/27 elements;
`control/control.log`); with the narrowing all depthwise cases pass on SystemC
and on the RTL. That file's `x + bias` column test is flattened by Inductor
to a 1-D kernel and reaches no `[R,1]` broadcast at all; the compiler test
`compiler/test/test_column_broadcast.py` covers both kinds directly on the
RTL, including the control that removing the copy from a loaded column
breaks lanes 1-15.

The analysis below is the record of the tree before narrowing.

## Before narrowing (`default_change_audit.json`, bf6fd2b product code)

`default_change_audit.json` compares, for all 113 original points
(`spec/r40/suites/hbm.json`, the union of the frozen suites), the code the
delivered default knobs compile on two software trees:

- B40-OBS delivery tree: `git archive r40` (9235f5d),
  `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`;
- r41s1: HEAD a0c9e72 plus this change set (the execution resource plan
  bookkeeping and its consumers), `files:a40bc877…`. Concurrent uncommitted
  work by other tasks in the same worktree is not in this tree.

Method (`tools/performance_harness/default_change_audit.py`, on GB10): one
fresh process per point with private Triton/Inductor/temporary caches;
`torch.compile(fullgraph=True)` of the ArchBench point on `tnca` tensors over a
host-memory stand-in Session (`tools/r40/v3_compile_preflight.py`
HostSession); every launch skipped, every device session refused. Knobs are
the R40 B40-S compile knobs (`TNCA_MAX_BLOCK=16`, `TNCA_GEMM_TILE=16,16,16`,
`TNCA_REDUCTION_ROWS=1`, `TNCA_BANK_ALLOC=0`, `TNCA_WINDOW_REUSE=0`,
`TNCA_SDPA=decomposed`, `TNCA_PIPELINE=1`, `TNCA_TUNE=0`) with the R40
preflight harness environment (`core-vrf-sync-v5`, O2, `fpga-u280-multicore`
compile options). Compared per launched kernel: the sealed text segment
(sha256) and the launch structure (manifest execution, entry points, semantic
options, constexprs, manifest resources), plus each graph's steady-call
launch sequence (kernel, grid). Artifact hashes are not compared: manifests
gained R41 provenance. Not measured: numerics, execution, performance.

### Result

All 113 points compile on both trees, with no host fallback and the
wrapper's launch count.

| | points |
|---|---:|
| code text and launch structure identical to R40 | **57** |
| code text changed | **56** |

The 57 unchanged points execute the same instructions as B40-OBS on the same
image and cannot regress there: every `op.add`, `op.argmax`,
`op.attention_decode`, `op.copy`, `op.cumsum`, `op.embedding`, `op.gelu_tanh`,
`op.layernorm`, `op.rmsnorm`, `op.rope`, `op.silu`, `op.softmax`,
`op.swiglu` point, and `op.transpose` large/hbm.

The 56 changed points: every point of `op.matmul` (9), `op.linear` (5),
`op.gemv` (5), `op.bmm` (4), `op.conv2d` (4), `op.attention_prefill` (4),
`model.mlp` (4), `model.gpt2_block` (4), `model.llama_prefill` (5),
`model.llama_decode` (5), `model.resnet_block` (4), and `op.transpose`
tiny/small/medium (3). In each, the Triton source Inductor generates is
unchanged (modulo `backend_hash`); only the TNCA lowering differs. 183
kernels change: 152 Inductor templates (GEMM/conv/attention matmul) and 31
pointwise kernels. Launch sequences, grids, role maps, entry points and
semantic options are identical; the only manifest launch-structure difference
is the declared private vector registers per role (`rf_private_per_role`:
+1 to +4 in 122 kernels, -2 and -4 in two).

### Why

Attribution by ablation: the r41s1 HEAD tree with one change reverted, the
same harness and knobs.

| Change (default on) | Reverted tree | Effect on the 113 points |
|---|---|---|
| `[R,1]` column-broadcast fix (`compiler/python/tnca_cc/frontend.py`, `_op_tt_broadcast`) | `files:c3a86e48…` | reverting it restores R40's code text **and** launch structure on all 56 changed points |
| predicate spill (`predicate_spill.py`, called from `adapters/r1.py`) | `files:0c98e890…` | none: identical to HEAD on all 113 points; no kernel records `predicate_spills` |
| pointwise retile fallback candidates at the default cap (`python/tnca/inductor.py`) | `files:946d523b…` | none: identical to HEAD on all 113 points; no `_BLOCK_FALLBACKS` entry (the first configuration always compiles) |

Compile decisions (`rctx_spmd`, `schedule` and their recorded fallbacks) are
the same on both trees for every kernel. In the JSON, `restores_control` of
the spill and retile ablations lists only the 57 points that were unchanged
anyway. The predicate spill and the retile fallback only act when R40 would
have refused a compile, which no original point does.

The broadcast fix replaces R40's relabelling of a `[R,1]` tile as `[R,C]` with
an explicit lane-0 extract and splat per row (for `i1`: select, extract,
splat, compare). It is required for a *loaded* `[R,1]` tile, whose other lanes
are not the row's value. In these 56 points it fires on computed,
lane-uniform columns instead. In the kernels inspected (the GEMM epilogue's
`(rm[:, None] < M)` store mask, the convolution template's `mask_x`/`mask_w`
inside its loop, the transposes' `ymask`) the `[R,1]` operand is a comparison
of an index column, not a load: every lane already holds the row's value, so
R40's relabel was correct there (e.g. `v.cmp.lt p1, v3, v4` on two splats in
the R40 `op.matmul/tiny` epilogue). Static size per changed kernel: +224 to
+9216 code bytes. Counting static instructions inside and outside
backward-branch loops (`*_static_instructions` in the JSON):

- 171 of the 183 changed kernels add instructions only outside every loop
  (+28 to +1152 static instructions, executed once per program); their
  loops are unchanged. The relative cost is largest where loops are short or
  absent (small-K GEMMs, `op.bmm`, the transposes) and small for long-K GEMMs.
- 12 kernels, the FP32 convolution templates of `op.conv2d` (4 points) and
  `model.resnet_block` (4 points), add 144-768 instructions outside loops and
  grow their loop bodies by 256 static instructions (from 2561-3012), a
  per-iteration cost.

These 56 points are therefore the only original points where the §6.5
non-regression budget can be consumed on the 90 MHz image, and the broadcast
fix is the only cause. Whether to measure them as a semantic-fix cost or to
narrow the fix to tiles not known to be lane-uniform is a separate decision;
this audit changes no code.

### Bookkeeping change set

The r41s1 HEAD tree (`files:d35b4197…`) and the final tree (`files:a40bc877…`)
emit identical code text and launch structure for all 113 points
(`same_code_as` in the JSON): the execution resource plan and its consumers
change no compiled code.

Run reports, kernel dumps and wrappers remain on GB10 under
`~/r41s1-audit/run-{r40,head,final,abl-bcast,abl-spill,abl-retile}`.
