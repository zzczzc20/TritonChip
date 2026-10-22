# R41 address arithmetic: 8 GiB SystemC window

Worktree `/home/ubuntu/TritonChip-addr64`, branch `r41-addr64`. Control software is `21d09f3`; reviewed WIP is `219e7dd`. Review fixes were pushed in `b048116`, `c8639de`, `8d6d57b` and `ea3fee7` (software tree `files:753b9f53…8c3e`); final validation below is on that tree. No ISA or RTL implementation was changed. No FPGA card was accessed.

## Audit and implementation

[AUDIT.md](AUDIT.md) lists the address/offset/size fields and calculations, their actual widths, range restrictions, and source lines across Torch storage/views, the allocator, Triton argument binding, code generation, descriptors, DMA, transport, LSU, AXI, and SystemC. The expanded source-site inventory is in the raw logs.

The WIP was not a completed result. Review reproduced its compiler tests on control and candidate, refreshed the generated support matrix, found missing scalar i64 division in GEMM grid arithmetic, and found 32-bit absolute-offset multiplication in the small explicit-descriptor path. Fixes preserve 64-bit scalar origins, bound local gather indices, and use per-lane scalar addresses when an independent wide offset cannot use the signed-i32 gather encoding. Tensor/device-storage copies split their total span into bounded launches; each launch receives full-width buffer offsets. Device copies execute on tnca.

The explicit descriptor regression is an additional direct Triton/RTL test: the standard Torch lowering does not expose a small explicit `TensorDescriptor` box. It is not counted as a standard-Torch before/after demonstration.

## Execution conditions

GB10 uses its existing `~/tc_env/torchenv/bin/python`, after sourcing `~/tc_env/env.sh`; nothing was installed. Source was synchronized with the instructed rsync exclusions. Fresh heavy-run checks showed 82, 69 and 94 GiB available. Later runs were pinned to CPUs 0–5; an aggregate process-tree snapshot recorded 8,263 MiB RSS. Other agents' processes were left alone. Processes stopped during audit restarts were selected by exact PID and descendant tree.

The backend is `systemc-tlm-grid`, `TNCA_TLM_MEMORY_BYTES=0x200000000`, profile `core-vrf-sync-v5`, three active cores, `TNCA_RCTX_SPMD=1`, descriptors enabled, `TNCA_OPT_LEVEL=2`, `TNCA_MAX_BLOCK=16`, GEMM tile `16,16,16`, reduction rows 1. The full stream additionally sets the existing `TNCA_MAX_BLOCK=128`; its pointwise grid is 9,437,184 programs. The eight-row embedding test sets `TNCA_REDUCTION_ROWS=8`. These are test settings, not new defaults.

The SystemC binary is `~/r41s1-archive-25094197f6e6-out/model-build/tnca-chip`, SHA256 `8aa953fde5168a8dc47e31f04cc21364b4fddbe310808f388b9baedd86ec8c5d`. The model uses untimed functional execution (`TNCA_TLM_UNTIMED_COLD=1`). These are numerical/address measurements, not timing or performance claims. Raw case JSON records TNCA settings, assembly hashes, grids, addresses, comparisons and refusals.

## Tests and before/after evidence

Each normal test uses standard Torch tensors on tnca and an independently evaluated CPU reference. Transfers to CPU only retrieve results for comparison. Exact tests compare bytes; floating reductions/GEMMs use the recorded absolute/relative bounds. Nonfinite mismatches cannot pass a tolerance comparison.

| Test | Control / earlier failure | Reviewed result |
|---|---|---|
| Compiler address suite (`compiler/test/test_addr64.py`), including RTL execution | Base: 9 tests, 6 failures and 8 errors across subtests | 10/10 passed on the final tree |
| 2.25 GiB int8/uint8 storage, tail arithmetic/reduction/cat/transpose, int64 gather | Base: 1 row passed, 5 failures each (one is the harness abort); narrowing trap, unsupported i64 remainder, later operations poisoned | 10/10 rows each |
| 2.25 GiB bf16 storage, sliced GEMM/linear weights | Base: 10 rows passed, 2 failures; assembler rejected `sNone` in a hoisted descriptor | 12/12 rows |
| Entire 2.25 GiB eager clone | Base: `64-bit copy indexing is not implemented` | Passed; head/middle/tail checked |
| Entire 2.25 GiB bf16 stream | Initial block-16 run refused the 75,497,472-program pointwise grid | Block 128: upload, read-all sums, write-all pointwise and clone-all passed; every output chunk checked |
| Placement at `0x100100000`, `0xffffc000`, `0x1fff00000` | Already passed on control; not claimed as a newly fixed defect | 11/11 checks at each base (prior run also `0x7ffe8000`) |
| 4.25 GiB bf16 storage (bases past 4 GiB) | Base: 1 row passed, 5 failures (one is the harness abort); kernel fault on the tail pointwise, unsupported 64-bit `arith.remsi` in `cat`, later rows poisoned. WIP: scalar i64 division unsupported, then runtime divisor, then VRF exhaustion | 12/12 rows |
| Eight-row embedding reduction | Also passes on base (not a fixed defect; kept as coverage) | 1/1 |
| `resize_` preserving a 2.25 GiB storage | Base: `tnca resize copy requires a positive 32-bit byte count` | 1/1 |
| Small explicit descriptor box with large stride, padding and negative coordinate | Base emits `v.mul` on absolute offsets | Direct descriptor/RTL regression passed |

Large-storage tests initialize independent head/middle/tail bands and compare the operations on those bands; they are not described as checking uninitialized storage. The separate stream initializes and checks the whole 2.25 GiB storage. `view_input` explicitly uses `torch.as_strided` with a storage offset past 2 GiB. Placement checks include the argument-block address as well as tensor addresses.

## Negative cases

The final runs passed high-bit truncation refusal (`2**32 + 5` with valid low bits), kernel access past the model window, allocation past capacity, transfer beyond a buffer/window, allocator exclusion and gap reuse around `[0x7fff0000, 0x7fff1000)`, and kernel access into that reserved TMU page. ABI tests reject over-wide i32 arguments and pointers outside u64, and preserve a packed descriptor base above 4 GiB.

The alias test refuses overlapping DMA arguments with `ALIAS_UNDECLARED`. Disjoint views of one storage are also conservatively refused for DMA because the ABI supplies no per-pointer extent: the runtime's span extends to the end of the allocation. That refusal is recorded as a limitation, not a numerical pass. The LSU-only two-view calculation passes its CPU reference.

## Emitted address computation

[emitted_i8_tail.s](emitted_i8_tail.s) is the complete emitted kernel for `b[-16:].float() * 2 + 1` on a 2.25 GiB int8 tensor. Inductor selects i64 indexing. The source byte offset is `2,415,886,336 + 16*pid + lane`, exceeding signed i32 even though the output is small. In the actual assembly, `s3` is `16*pid`, `s4` holds the full-width constant `2415886336`, and `s2` becomes their scalar sum. The relevant emitted instructions (intervening pair materialization omitted) are:

```asm
    .li       s4, #2415886336
    s.add        s2, s3, s4
    s.shli       s4, s2, #0       // element offset -> bytes (x1)
    s.add        s5, s1, s4       // &in_ptr0[...]
    v.ld.sb      v0, [s5, #0]
```

`in_ptr0` itself is loaded with `s.ld.d` from the argument block. The lane term stays a contiguous vector access; the large invariant is never narrowed into a gather index. The final evidence collection verifies this assembly against the measured case's stored hash.

## Regression gate and remaining limits

`make compiler` passed locally (Triton unavailable: skips) and on GB10 on the final tree (`compiler-unit PASS 123 tests`, none skipped). `tnca-opt` is unavailable on both hosts, so the IR suite is explicitly SKIP, not a measured pass.

Default audit (`default_change_audit.py`, compile-only, control `audit-base` on `21d09f3`): **113 points, 113 unchanged, 0 changed**. The earlier candidate (`8d6d57b`) measured 105 unchanged / 8 changed: all conv2d and resnet_block points, kernel `triton_tem_fused_convolution_*`, +32 in-loop instructions. Cause: the review's [R,1] row-pointer fix lived in the frontend's `tt.broadcast` of a pointer tile and also rewrote Inductor's ordinary `x_base[:, None]` pointers in the conv template. `ea3fee7` moved the fix into `wide.py` (`_column_row`: one-column offset rows narrowed/rebased from 64-bit values are made lane-uniform there) and restored the original broadcast; the row-broadcast compiler test still passes.

No tested width repair requires an ISA or RTL change: scalar registers, pointers, descriptor bases/strides, DMA accumulation and external buses already carry 64 bits. Arbitrary independent wide lane offsets use scalarized accesses and can be slower. A native vector-64 gather index would require an ISA/RTL extension if that performance feature is desired; it was not implemented. Tiles whose *local* span exceeds signed i32 remain explicitly refused/trapped; general multi-row i64 data operations and runtime vector division retain their existing implementation restrictions. The scalar runtime divider is supported.

The core RTL compiler tests execute smaller physical addresses because their backing memory is finite. Additional existing RTL benches exercise high NUCA tags and AXI bases at `0x100000000`, randomized backpressure, refusal and reset handling. They do not constitute a full 8 GiB RTL machine or board measurement.

## Raw evidence

`raw/review-prior.tgz` preserves the predecessor's `b1`, `h1`, `h2` logs, JSON, wrappers and assembly, including its failures. Further archives, final source digests, comparisons and checksums are listed after collection. No helper scripts or credentials are included.
