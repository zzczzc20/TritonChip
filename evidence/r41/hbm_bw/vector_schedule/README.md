# Independent-vector scheduling and LSU capacity

The previous width-only result is insufficient. This continuation evaluates independent vectors, finite LSU command capacity and standard torch code. All results here are uncalibrated model predictions. No new FPGA image or board operator speedup has been measured.

## Native scheduling experiment

`tools/r41/vector_schedule_model_tests.py` processes the same three disjoint 4,096-element integer add partitions as the original width sweep. Unroll factors 1/2/4/8 load independent vectors first, then add/store them. It checks every output byte. The optional `tl_lsu_units` extends the model's two execution slots to 4/8/16, retaining a shared issue port: a younger command waits for all older commands to finish issuing beats, while their replies can remain pending. Every unit remains occupied until response/completion. The experiment keeps **16 total beat tags**, reducing slots per unit as replication grows; it does not assume unlimited transaction identity space.

| Core/NoC useful width | LSU units | Best unroll | Cycles | Ratio vs original 59,179 cycles |
| --- | ---: | ---: | ---: | ---: |
| 8 bytes | 2 | 2 | 50,459 | 1.173 |
| 64 bytes | 2 | 8 | 29,949 | 1.976 |
| 64 bytes | 4 | 4 | 16,382 | 3.612 |
| 64 bytes | 8 | 8 | 10,287 | 5.753 |
| 64 bytes | 16 | 8 | 9,991 | 5.923 |

The 20 scheduling cases pass. The 4/8/16-unit hypotheses also pass the existing native-chip integer, floating-point, narrow-memory, reduction, shuffle, SFU, grid and refusal checks, plus masked unaligned accesses. Increasing units past eight offers little benefit in this microbenchmark. These are unbuilt capacity hypotheses, not RTL throughput guarantees; shared response landing/writeback arbitration, tags and area still need implementation and physical validation. Default two-unit selection retains the original choice and shared-port timing rules.

## Standard torch experiment

`tools/r41/operator_path_model_sweep.py` runs actual ArchBench standard torch add and silu, through Inductor/Triton/native artifacts and the model backend. It uses one worker and two host threads. The baseline and both candidates use the same software source; the candidates deliberately choose larger block caps to expose independent vectors. All must pass the existing numerical comparisons with the original tolerances. Model configured clock is 100 MHz; this is not a measured clock or a calibrated board-time prediction.

The source audit found that the inherited service configuration has stale SFU occupancy/latency 27/28. Both current RTL and the historical Stage-2 image source c9f25787494a use SFU LAT=6. The corrected baseline is 22/23 cycles for 16 serial lanes. Four identical scalar pipelines need four passes plus the same six-cycle latency: occupancy/latency 10/11. The RTL equivalence test separately observes 23 versus 11 completion cycles. An initial trial with 7/8 cycles was too optimistic and is excluded; its log remains in the raw archive. The earlier exploratory runs also used the inherited nominal 90 MHz configuration; the reproducible final operator sweep explicitly configures 100 MHz.

| Standard torch configuration | Add cycles / predicted ratio | Silu cycles / predicted ratio |
| --- | ---: | ---: |
| Corrected model baseline, block cap 16 | 44,505 / 1.000 | 51,741 / 1.000 |
| Wide path, four LSU units, four SFU slices, block cap 64 | 15,742 / 2.827 | 21,887 / 2.364 |
| Wide path, eight LSU units, four SFU slices, block cap 128 | 13,695 / 3.250 | 17,481 / 2.960 |

All six standard torch runs pass their original numerical comparisons. Maximum relative errors are unchanged across variants: approximately 0.0003033 for add and 0.0002409 for silu. These predictions justify implementing the coupled candidate rather than merely adding AXI outlets. They do not establish the requested board speedups or a 100 GB/s path.

The original suite argument adds its points to explicit selections rather than filtering them. The first baseline launcher inadvertently included additional suite points; only its own explicitly identified parent and descendants were stopped, after the requested add/silu results completed. The log is preserved. The final driver supplies exactly two explicit points without a suite argument. No unrelated process was stopped.

`raw.tgz` preserves the native scheduling inputs, references, binaries' identities, counters and the earlier operator experiments. `operator100_raw.tgz` and `operator100.json` contain the final 100 MHz configured standard torch runs. Ratios there compare candidate predictions to the model baseline; they must never be presented as speedups against the measured 100 MHz image. The governing board baseline remains the Stage-2 results and points.json. Coverage of two points does not qualify mul/gelu/copy, other shapes or the complete original suite.

## Design implication

A viable candidate must widen/coalesce the true operator path, increase finite independent LSU command capacity, expose independent vectors in compiled code and increase SFU throughput for silu/gelu while retaining the generated functions' bit patterns and latency. The separate stream bank remains useful for the 100 GB/s memory design target and copy diagnostics, but is insufficient to establish this operator path. Next implement and screen the required datapath boundaries, including their receiving stages. No full-run allocation or card access was consumed by these experiments.
