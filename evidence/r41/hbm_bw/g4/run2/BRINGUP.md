# Run 2: after the link, and card bring-up (checklist, prepared 2026-10-29)

[`doc/engineering_discipline.md`](../../../../../doc/engineering_discipline.md) is binding. This is the plan; nothing here has been run. The image is not loaded until the idle rule and the user's approval allow it. The flow reuses run 1's machinery ([`../../../stage2/run1/README.md`](../../../stage2/run1/README.md), [`../../../stage2/board/README.md`](../../../stage2/board/README.md)). The differences for run 2 are listed below.

## 1. Post-route check (x86, after the link ends; no card)

Run it from a tree that has `tools/r41/run2_postlink/`, a fresh archive of the latest `r41-hbm-bw`, **not** the frozen tree `~/r41run2/src_1f0a5aff`, which stays untouched:
```sh
bash tools/r41/run2_postlink/post_run2.sh $HOME/r41run2/build_1f0a5aff "$PWD"
```
- **Waiting and output.** It waits for `build/attempt.exit`, then runs `analyze_run2.tcl` on the routed checkpoint: run 1's analysis, 2 threads, read-only. Its outputs are route status, min/max timing, utilisation per SLR and per hierarchy, clock interaction, CDC, bus skew, check_timing, and endpoint lists for failing and near-critical (< +0.5 ns) paths. It then runs `check_run2.py`.
- **Gates** (`check_run2.py`, verdict PASS/FAIL):
  - design and kernel-clock (`clk_kernel_00_unbuffered_net`) setup and hold slack ≥ 0, with no failing endpoint;
  - every routable net routed, with 0 routing errors;
  - Critical CDC findings equal to the 56 platform findings reviewed for attempt 3 and run 1. Any other count needs a new review.
- **Reported, not gated:**
  - SLR CLB packing against the G4 estimate (SLR0 95.5 ± 2.5 %; run 1 96.74 %);
  - SLL use;
  - kernel endpoints below +0.5 ns (run 1: 1,111);
  - the tile screens (tile 0 +0.600 ns / 43,345 CLBs, tile 1 +0.641 ns / 42,369 CLBs).
- **Reference.** On run 1's reports the checker reads: kernel WNS +0.042 and WHS +0.003, design WNS +0.016, 1,153,083 nets routed, 56 Critical CDC findings, SLR0 96.74 %, and 1,111 endpoints below +0.5 ns. Verdict PASS.
- **If the link failed at placement or routing:** see run2/README.md, "If placement fails".

## 2. What changes against run 1's board batch

| Item | Run 1 | Run 2 |
|---|---|---|
| Profile | `core-vrf-sync-v5-x8` | `core-vrf-sync-v5-x8-g2c4-dp`: 8 barriers, 4 contexts, `m.mma.dp.async`; the compiler fences TensorMap snapshots (`compiler.snapshot_fence`) |
| Image parameters | `stage2/integration/image.json` | `g4/run2/image.tcl` (the multicore backend needs `hybrid_lsu=True`, 3 cores, 3 memory ports, `memory_controls 1`) |
| Memory layout | linear | `HBM_STRIPED 1` (layout `pc32-page4k-v1`): the runtime must use the explicit 8 GiB allocation it already enforces (`tools/r41/hbm_layout_tests.py`, `python/tnca/hbm_layout.py`) |
| AXI | 256-bit, 8 IDs | 512-bit, 32 IDs; v++ inserts a 512→256 width converter in front of each HBM port. Check `xclbin_hbm_map.py` on the new xclbin: the three outlets must still connect to `MBG[0:31]` |
| Clock qualification | `r41s2-run1-100mhz` in `python/tnca/operating_clock.py` | a new reviewed entry is needed for the run-2 xclbin's sha256 and exactly 100 MHz, after `check_run2.py` passes and the CDC review is equal. `clock_qualify.py assemble` then writes `qualified_clock.json` |
| Image-source identity | `board/image_source.json` (c9f2578) | regenerate for `1f0a5aff`'s image sources. The board tree must hash equal on `rtl/`, `tb/`, `flow/`, `xls/`, `spec/r1_profiles.json` and `compiler/targets` |

## 3. Load and first board tests (the user starts them, after approval)
1. **xclbin and UUID.** The xclbin is `$HOME/r41run2/build_1f0a5aff/xo/tnca_u280.xclbin` (run 1's runner location; confirm in `link.stdout`).
   - Copy it to the FPGA host with its sha256.
   - `xclbinutil --info`: record the UUID and `DATA_CLK` = 100 MHz, and confirm that AUTO-FREQ-SCALING did not lower it.
2. **Load and clock.** Load through the existing `program.sh` path under the idle rule. `clock_qualify.py capture` must then show the UUID in sysfs, three clock-counter samples agreeing on 100 MHz, and the ICAP readback.
3. **Identity** (`s2-identity`): the source check, `clock_qualify verify`, and three small ArchBench points with `clock_hz` 100 MHz. If this fails, the queue stops.
4. **Memory map:** the hbm8 manifests (window, negatives, alias, crossing, DMA, clamp) with the striped layout, then the stream / big-offset cases.
5. **The original 113 points** (`hbm.json`): numeric references, no fallback.
6. **The r41_membound suite** (`archbench/suites/r41_membound.json`) with the default compiler and with `TNCA_TILED_POINTWISE=2`: the bandwidth measurement this image was built for.
7. **FP8:**
   - `fp8_cases.py` and `fp8_blockwise_cases.py`. They must find `m.mma.dp.async` / `t.ld.async.b.t` in the dumped kernels.
   - The Hub `finegrained-fp8` kernels.
   - The Qwen FP8 layers (`qwen_tnca_layers.py`).
8. **Endurance:** the 2-hour mixed run.

Stop on any UUID or clock mismatch, card change or identity failure, as run 1's queue does (exit codes 3–7).
