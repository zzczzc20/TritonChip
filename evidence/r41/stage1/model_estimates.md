# Full-model compile, code, IR and memory estimates (S1-04)

These are **estimates from compile-only builds**, not numerical execution, device
measurement, performance evidence or a Stage 2 capacity PASS. The summary is in
[model_estimates.json](model_estimates.json). The raw runs, which include results,
logs, wrappers, FX/ATen graphs, sealed artifacts and the TNCA IR of the main runs,
are in [kickoff/model-estimates-v1-evidence.tgz](kickoff/model-estimates-v1-evidence.tgz).

## Method

The tool is `tools/performance_harness/model_compile_estimate.py`, at harness SHA256 `b35c5c10…`.
It uses the committed product tree `files:d35b4197…`, which is identical at
`a0c9e72`/`a3eda5e`, with torch 2.14.0, Triton 3.8.0 and the pinned Transformers
`4321355e`. The model is the official 24-layer language branch, built on meta from the
pinned `config.json`. Its 320 parameters are checked against the audited inventory,
and the tied embedding stays one storage. The model is then placed on the native
`tnca` device as uninitialised storage.

Each graph uses `torch.compile(backend="inductor", fullgraph=True)`. Every kernel is lowered
by the TNCA Triton backend and sealed, and none is launched (`TNCA_LAUNCH=skip`), as in the
`v3_compile_preflight.py --device native-stub` path. The other settings are profile
`core-vrf-sync-v5`, backend `fpga-u280-multicore`, `opt_level` 2 and tensor descriptors.
Storage goes to a host stand-in session. That session uses the runtime's own
first-fit `_Allocator` over the device map: heap at `0x20000`, with the reserved
range skipped. No checkpoint byte is read. Every graph runs in a fresh process
with private caches, pinned to 6 cores, under an RSS and host-memory watchdog.

- **prefill**: B=1, 128 tokens. The official `StaticCache` holds 144 slots (128+16) and is
  written in place, and `logits_to_keep=1`. `Cache.reset()` runs before each call.
- **decode**: one token at position 128 in the same fixed bucket. A second call at
  position 129 did **not** recompile and launched the identical sequence (`dk`/`shim` runs).
- **prefill-nocache**: the no-cache, all-logits graph that `language_meta_analysis.py`
  traced. It is used as a cross-check.

Both cached graphs use the public mask-mapping argument: the full-attention mask is
built in the graph from `position_ids`, and linear attention gets `None`.

A kernel that TNCA fails to lower is recorded and skipped, so the wrapper and the
allocation plan still complete. The one refused ATen op gets a shape-only stand-in
that allocates its outputs and computes nothing. To measure past the blockers,
some runs add named **harness-only shims**. These shims are not product fixes.

| Run | Shims |
|---|---|
| `std` | none |
| `dk` | `decompose-k-off` |
| `log1p` | `log1p-kahan` |
| `shim` | `decompose-k-off`, `log1p-kahan`, `scan-no-split`, `int32-positions` |
| `export` | `shim` + `delta-solve-export-branch` (the official forward-substitution branch) |

## Blockers on the unmodified standard path

The official graph does not compile completely on the current TNCA path. A failing kernel
reports only its first diagnostic, so some blockers show up only after an earlier one is
shimmed.

| # | Blocker | Where | Effect | Measured past by |
|---|---|---|---|---|
| B1 | Inductor `NoValidChoicesError` for skinny GEMMs. With K ≥ 64·max(M,N), Inductor drops its mm template for decompose‑K, which has no tnca configuration | decode `in_proj_b/in_proj_a` (1×2048×16) | **the whole decode graph aborts in lowering** | `decompose-k-off` |
| B2 | `libdevice.log1p` is deliberately absent from TNCA libdevice. Inductor emits it for `softplus` | linear-attention decay: prefill cumsum scan kernel and decode `in_proj_a` epilogue, 18 launches/call each | kernel fails | `log1p-kahan` |
| B3 | Split-scan `cumsum` gets `R0_BLOCK=256` (raised to `min_split_scan_rblock` after the device heuristic). TNCA rejects it: "register block width 256" | prefill decay cumsum, 18 launches | kernel fails | `scan-no-split` |
| B4 | `aten.linalg_solve_triangular` stays a host ATen call in the wrapper. The tnca dispatcher refuses it at run time (2 per linear layer, 36 per prefill) | prefill chunked gated-delta rule | prefill cannot execute | `delta-solve-export-branch` (stand-in otherwise) |
| B5 | RoPE `position_ids.float()` on int64 is a 64-bit tensor op (`arith.sitofp`) | 1 RoPE kernel per graph | kernel fails | `int32-positions` |
| B6 | RoPE `cos`/`sin`: "no TNCA lowering for TTIR operation `math.cos`" (the trig family is deliberately absent) | same RoPE kernel | kernel fails | **not shimmed**: its code bytes are missing from every working set |

Nothing else failed. With the four shims, prefill compiles 59/60 kernels and decode
44/45. In each case the one failure is the RoPE kernel (B6). The `export` variant
compiles 153/154 and runs no host fallback.

## Code, compile and IR (24 layers, measured)

| | prefill `std` | prefill `shim` | decode `dk` | decode `shim` | prefill `export` |
|---|---:|---:|---:|---:|---:|
| Distinct kernels launched (+ failed) | 58 (+2) | 59 (+1) | 43 (+2) | 44 (+1) | 153 (+1) |
| Launches per call | 788 | 806 | 488 | 506 | 2,516 |
| Unique code bytes | 442,976 | 438,968 | 152,744 | 164,712 | 653,208 |
| ÷ 128 KiB `CODE_STORE_BYTES` (64-B aligned) | 3.39× | 3.36× | 1.18× | 1.27× | 5.02× |
| Largest kernel (bytes) | 37,624 | 37,624 | 14,216 | 16,520 | 32,936 |
| LRU replay, steady call: loads / code upload bytes | 664 / 6.21 M | 682 / 6.21 M | 28 / 108 K | 28 / 104 K | 2,374 / 10.07 M |
| Frame compile wall, s | 104.0 | 116.9 | 64.6 | 46.5 | 255.9 |
| of which TNCA lowering, s | 36.6 | 34.6 | 8.6 | 10.0 | 27.1 |
| Compiler process peak RSS | 1.13 GB | 1.15 GB | 1.05 GB | 1.06 GB | 1.80 GB |
| Dynamo FX nodes / Python bytes | 4,338 / 460 K | 4,338 / 460 K | 3,348 / 378 K | 3,348 / 378 K | 14,616 / 1.17 M |
| Post-decomposition ATen nodes / bytes | 6,854 / 601 K | 6,854 / 601 K | 4,778 / 404 K | 4,778 / 404 K | 26,546 / 2.47 M |
| Inductor buffers / scheduler nodes | 1,265 / 867 | 1,265 / 867 | 833 / 549 | 833 / 549 | 2,903 / 2,541 |
| Wrapper / Triton source bytes | 1.01 M / 253 K | 1.01 M / 253 K | 734 K / 191 K | 734 K / 191 K | 2.49 M / 654 K |
| TNCA TTIR / scheduled TNIR / asm / hex bytes | 1.48 / 3.35 / 1.71 / 0.95 M | 1.51 / 3.32 / 1.70 / 0.94 M | 1.05 / 1.13 / 0.62 / 0.33 M | 1.10 / 1.23 / 0.68 / 0.36 M | 3.95 / 4.94 / 2.54 / 1.40 M |

Breakdown of the compile time for `shim` prefill / decode, in seconds:

| Stage | Prefill | Decode |
|---|---:|---:|
| Dynamo trace and guards | 10.8 | 5.7 |
| AOTAutograd | 15.0 | 6.6 |
| Inductor lowering and scheduling | 17.8 | 10.0 |
| Inductor codegen | 36.0 | 12.5 |
| Triton TTIR | 2.3 | 1.5 |
| TNCA lowering | 34.6 | 10.0 |
| **Process total** | **131.1** | **54.2** |

The `std` decode aborts after 25.1 s of frame compile.

Six compiles shared 6 cores on a shared host with `compile_threads=1`, so wall times
are contended upper-end values. The `export` run finished on a less loaded host.
A short-lived child process reached a further 0.74 GB RSS. The sampled whole-tree
peak stayed ≤ 1.77 GB.

**Session view** (`shim` prefill plus 16 decode steps):

- 103 distinct kernels and 603,680 B of unique code, **4.63×** the code store.
- 8,902 launches, 1,350 code loads and 8.63 MB of code upload.
- A steady decode step reloads 28 modules (~104 KB).
- With the `export` prefill instead: 197 kernels, 817,920 B (6.28×).
- No single kernel exceeds the 112 KiB wide limit. The code store overflows only as
  a working set, which costs reloads (driver LRU eviction), not a refusal.

## Device-memory peak (allocator plan of the steady call)

| Bytes | prefill `shim` | decode `shim` |
|---|---:|---:|
| Weights, distinct (tied counted once) / allocator reserved | 3,763,655,360 / 3,763,655,936 | same |
| States: recurrent FP32 18,874,368 + conv 884,736 + KV 1,769,472 + lengths 48 (reserved) | 21,529,984 | 21,529,984 |
| RoPE buffers + inputs | 1,664 | 256 |
| Graph-held constants | 0 | 0 |
| Steady transient peak (activations + Inductor buffers + output) | 30,557,376 | 2,122,688 |
| First-fit fragmentation/alignment at peak | 4,259,904 | 665,280 |
| Heap extent peak | 3,820,004,864 | 3,787,974,144 |
| Launch argument blocks + scratch (8 in flight × 3 cores; single-submit 59,136 / 6,912) | 473,088 | 55,296 |
| Code region (on-chip, below the heap) | 131,072 | 131,072 |
| **H_peak** | **3,820,609,024 (3.558 GiB)** | 3,788,160,512 |

The session peak is the prefill one. The `export` prefill gives 3,821,084,160.

Under the rule **C_usable ≥ 1.25·H_peak**, the requirement is 4,775,761,280 B (4.448 GiB).
C_usable here is the window size minus the 4 KiB reserved range.

| Window | C_usable | Fits without margin | Meets 1.25× | Headroom vs margin |
|---|---:|:-:|:-:|---:|
| B40 4 GiB | 4,294,963,200 | yes (474 MB spare) | **no** | −480,798,080 |
| Stage 2 6 GiB | 6,442,446,848 | yes | yes | +1,666,685,568 |
| Stage 2 8 GiB | 8,589,930,496 | yes | yes | +3,814,169,216 |

Under the margin, the 4 GiB window allows at most 3,435,970,560 B, which is less
than the weights alone. 6 GiB and 8 GiB meet the margin on these estimates.

Cross-check: the no-cache, all-logits prefill gives an allocator transient peak of
**64,094,208 B**. This is exactly the straight-line ATen liveness peak from
`language-meta-v1` (prefill 128, `v4` report).

## Measured, modelled and not covered

- **Measured, full 24 layers, no layer extrapolation:**
  - compile wall time per stage and compiler RSS;
  - FX, ATen, Inductor and TNCA IR sizes;
  - code bytes and resources of every sealed artifact;
  - the launch sequence;
  - the first-fit allocator trace of weights, states, inputs and every Inductor
    buffer of a steady call.
- **Modelled:**
  - the LRU code-region replay (the driver's `_load` policy over the measured sequence);
  - argument-block bytes (the `Session.submit` formula over the manifests);
  - C_usable.
- **Deviations from the standard path:** the named shims above. They change one
  Inductor config value, add one libdevice function, set the scan split, use int32
  position dtype, and select an official model branch.
- **Not covered:**
  - code bytes of the RoPE kernel (B6), and in the `std` and `dk` runs also of the
    B2/B3 kernels;
  - the device implementation and workspace of `linalg_solve_triangular` (only its
    outputs are counted; `export` gives the pure-TNCA alternative);
  - buffer lifetime extension under `TNCA_SUBMIT=batched`;
  - numerics, device time, host staging, reset epochs and multi-session effects.

## Reproduce (GB10)

```sh
python model_compile_estimate.py --config config.json --inventory inventory.json \
  --sources sources.json --out runs/final-shim --graphs prefill decode \
  --shims decompose-k-off,log1p-kahan,scan-no-split,int32-positions
python model_compile_estimate.py --summarize std-prefill=final-std/prefill/result.json ... \
  --session shim-prefill,shim-decode --session export-prefill,shim-decode \
  --aten-crosscheck std-prefill-nocache=inputs/language-meta-prefill128-v4-report.json \
  --estimates-out model_estimates.json
```

The archive also keeps:

- the 2-layer discovery runs, which show the order in which the blockers were found;
- one superseded `export` run, kept as a negative. Its steady call recompiled into a
  continuation-prefill graph because the completed cold call had set
  `has_previous_state`. The harness now resets the cache before each prefill call
  and refuses a recompiled steady call.
