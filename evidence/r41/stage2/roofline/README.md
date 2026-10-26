# Roofline of the 113-point ArchBench suite: TNCA (100 MHz and 90 MHz) against NVIDIA GB10

`roofline.png` is the figure. `roofline.json` holds the peaks, the per-platform summaries and the per-family efficiencies. `points.json` has every plotted point. Produced by `tools/performance_harness/roofline_report.py`.

## Definitions
- **x axis:** arithmetic intensity, in FLOP per byte of required traffic. FLOP is 2 × MACs + FP ops + SFU ops, and bytes are the required reads plus writes, both from the ArchBench `demand` field.
- **y axis:** achieved FLOP/s, which is demand FLOP divided by device time.
- **Roofline:** ArchBench spec peaks. They are never measurements.
  - TNCA: 64 FLOP/cycle (the vector FPU lanes plus the matrix MACs, summed, as the ArchBench strict bound does) and 8 B/cycle through the core's single memory port. That is 6.4 GFLOP/s and 0.8 GB/s at 100 MHz.
  - GB10: 273 GB/s with 36.9 TFLOP/s fp32 SIMT, or 155 TFLOP/s fp16 tensor.
- **eta:** the ArchBench bound divided by measured time. 1.0 means the point sits on its bound.

## Results
| Platform | Points plotted | eta geomean | eta median | eta best |
|---|---:|---:|---:|---:|
| TNCA 100 MHz (Stage 2 image, board) | 102 | 0.103 | 0.127 | 0.498 |
| TNCA 90 MHz (R40 delivery image, board) | 104 | 0.102 | 0.121 | 0.505 |
| GB10 (CUDA graph) | 104 | 0.064 | 0.093 | 0.724 |

- On the 102 points that both platforms have, TNCA's eta is 1.68× GB10's (geometric mean). Each platform is measured against its own spec bound.
- Moving from 90 to 100 MHz leaves eta unchanged. The 1.10× clock gain shows up as speed, not as closeness to the roofline.
- TNCA sits about 9.7× below its roofline, whether the point is memory-side (10.0×) or compute-side (9.2×).
- GB10 is 13× below its roofline on memory-side points (93 of 104 points are memory-bound) and 1.4× on compute-side points. A few large GEMMs get within 1.4× of the tensor-core roof.
- Best TNCA families (eta geomean): `linear` 0.41, `mlp` 0.37, `add` 0.24, `matmul` 0.23, `cumsum` 0.20. Worst: `attention_prefill` 0.012, `conv2d` 0.023, `resnet_block` 0.025, `bmm` 0.033.
- Families where GB10 is further from its bound than TNCA: `matmul`, `linear`, `mlp`, `cumsum`, `add`, `gpt2_block`.
- Families where TNCA is further from its bound: `attention_prefill`, `attention_decode`, `bmm`, `llama_decode`, `gemv`.

## Which memory ceiling? (correction)
The 0.8 GB/s roof is not the HBM bandwidth. It is ArchBench's per-core spec bound: one core, 8 B/cycle through its single memory port (`tnca_mem_mux.v`, 64-bit), at 100 MHz. The figure now shows all the ceilings of the 3-core board, from tightest to loosest:

| Ceiling | Value at 100 MHz | Basis |
|---|---:|---|
| One core's port (ArchBench bound) | 0.8 GB/s | 8 B/cycle |
| Three cores' ports | 2.4 GB/s | 3 × 8 B/cycle |
| Three AXI outlets | 9.6 GB/s | 3 × 256-bit (32 B/cycle) |
| U280 HBM2, theoretical | 460 GB/s | 32 pseudo-channels × 14.4 GB/s |

The compute ceiling for three cores is 19.2 GFLOP/s (3 × 64 FLOP/cycle). Distance of the 102 TNCA 100 MHz points to the roofline of each ceiling (geometric mean; no point is above any roof):

| Memory ceiling | Gap | Best | Worst |
|---|---:|---:|---:|
| One core (the ArchBench-bound roofline) | 9.7× | | |
| Three cores' ports | 29.0× | 6.0× | 597× |
| Three AXI outlets | 66.2× | 6.0× | 652× |
| HBM2 theoretical | 128.4× | 6.0× | 1,982× |

So the 9.7× gap above is relative to a one-core roof. Against the 3-core hardware, the gap is 29× to the core ports. The bottleneck is the core-side path, not HBM. R40's model attribution (`evidence/r40/v3/model_attribution`) says the same: the NUCA home-slice engine and the per-core memory path limit streaming, not the outlets or HBM. The HBM line is a theoretical ceiling that this design cannot reach with 3 outlets of 256 bits at 100 MHz (9.6 GB/s).

## What this does not say
- **Absolute speed:** the GB10 is about 600× faster in time on the same points (geometric mean of device-time ratios, 1/0.0016). The two rooflines differ by about 6,000× in FLOP/s. Efficiency is comparable between the platforms; absolute performance is not.
- **TNCA roofline:** eta uses the single-core spec bound, although the board runs three cores; see the section above for the three-core ceilings.
- **Dropped points:** 11 of the 113 points have zero FLOP demand (pure data movement such as `embedding` and `transpose`), so they have no place on a FLOP roofline. The GPU file has 115 records for the 113 points (two points appear twice); the same 11 zero-FLOP records are dropped, which leaves 104. The TNCA 90 MHz set also plots 104 points.
- **Time basis:** TNCA time is the serial-grid device cycle count, the basis of eta. GB10 time is CUPTI device activity with a cold L2.
- **Data:** the 100 MHz points are `s2-original-suite` from the Stage 2 board run. The 90 MHz points and the GB10 points are the R40 v3 delivery records (`evidence/r40/v3/gpu_comparison`). The Stage 1 `board-original-suite` rerun on the 90 MHz image has not finished.
