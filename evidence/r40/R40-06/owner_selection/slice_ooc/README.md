# Paired NUCA Slice OOC Screen

This experiment tests whether the owner-selection change improves actual
Vivado mapping before another full FPGA implementation run. It uses tile
zero's geometry from the three-core maintenance image, Vivado 2023.2,
four threads, identical source dependencies and constraints, and a 10 ns
clock for internal timing reports. It does not certify placement, routing,
CDC, a loadable image, or graph performance.

| Metric | Reference `09ebe06` | Corrected candidate `cbb6402` |
|---|---:|---:|
| LUTs | 83,925 | 79,258 |
| Registers | 32,064 | 32,052 |
| Internal estimated WNS | -13.417 ns | -5.957 ns |
| Worst-path logic levels | 114 | 66 |
| Worst-path data delay | 23.174 ns | 15.714 ns |
| Logic / estimated routing delay | 7.006 / 16.168 ns | 4.711 / 11.003 ns |

LUT count decreases by 5.56%, and estimated setup slack improves by 7.460 ns.
This supports the local hypothesis that sharing base priorities reduces
mapped slice logic and query-dependent critical-path depth. **Setup remains
negative; 100 MHz operation is not qualified.** The new limiting path runs
from `cur_addr_reg[7]` to `ms_wbadr_reg_0_7_0_13/RAMA/WE`.

The internal clock constraint is attached after synthesis, following the
existing OOC flow. No external I/O delays or placed clock source are supplied.
Routing delays and clock skew are estimates. These results must not be
converted into a routed frequency prediction or a whole-system speedup.

## Reproduction and baseline reuse

Run the Verilog-2005/SystemVerilog-2012 frontend checks and equivalence proof
before invoking the vendor tool. With the relevant source snapshot:

```bash
vivado -mode batch -source flow/fpga/synth_r40_slice.tcl -tclargs \
  /path/to/tnca_nuca_slice.v /path/to/rtl/common /path/to/NEW_OUTPUT
```

`experiment.json` in the retained inputs records complete parameters, source
hashes, tool version, thread count, hypothesis, and success/failure branches.
Each variant is supervised by the existing process-group runner with an
explicit 1,800-second limit. This limit bounds build time and never ranks
candidates.

The completed baseline was reused only after checking every source/header/Tcl
dependency used by it, all parameters, part, clock target, toolchain, and thread
count against the corrected experiment. The runner also verifies the baseline
reports and checkpoint hashes before and after the candidate run. This avoids
repeating a completed synthesis without treating a changed build as identical.

Two setup failures are preserved: an unsupported `read_verilog -include_dirs`
option rejected before synthesis, and the superseded candidate's SystemVerilog
reserved identifier rejected by Vivado. The corrected fileset include setup
and renamed identifier complete successfully. Both failures have specific
causes; no implementation directive rotation was performed.

## Retained artifacts and next decision

`baseline_and_failures.tar.gz` contains complete inputs, baseline reports,
runner source, and failed-run logs. `candidate_complete.tar.gz` contains the
corrected inputs, successful run record, and candidate reports. Checkpoints
remain on the FPGA server; their hashes and paths are retained in the run
records and `final_identity.txt`. `running_observation.json` is a dated live
process observation, superseded by the completed candidate run record.
`summary.json` binds source identities, reports, comparison, and archive hashes.

Inspect the reported remaining arithmetic/selection cone before changing it.
Any endpoint precomputation or additional staging must preserve full unsigned
64-bit range semantics, including zero sizes and overflow, as well as metadata
visibility. Prove the change and repeat a bounded module experiment before
considering full integration against the main linked design's actual paths.
The original three-core implementation continues independently; this experiment
does not replace its required physical and vertical checkpoints.
