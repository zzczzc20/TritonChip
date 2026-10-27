# Independent stream-outlet service experiment

## Qualification and result

This experiment evaluates a hypothetical stream outlet using the existing Ramulator2 HBM2 bridge. It is not native instruction execution, a calibrated whole-chip prediction, or a board measurement. The existing core/NUCA path is not bypassed in a product image by this experiment.

The probe executed 54 cases: 1,024 and 16,384 lines; read, write and alternating independent read/write traffic; one-, two- and three-cycle issue intervals; eight-, 32- and 64-line credit windows. Each line is 64 bytes. The nominal kernel clock is 100 MHz and the DRAM command clock 900 MHz, so admission occurs every ninth DRAM tick. The historical fixed pipeline assumption is represented as 20 kernel cycles between controller completion and credit return. This constant is an assumption, not a fresh measurement. Final elapsed time includes the fixed return delay and rounds up to a kernel edge.

For the 16,384-line cases with one independent outlet per PC:

| Traffic | Issue interval | Window | Per-PC GB/s | Ideal 24-PC aggregate GB/s |
|---|---:|---:|---:|---:|
| Read | 1 | 8 | 2.102 | 50.45 |
| Read | 1 | 32 | 6.045 | 145.09 |
| Read | 1 | 64 | 6.180 | 148.32 |
| Write | 1 | 8 | 2.188 | 52.52 |
| Write | 1 | 32 | 6.063 | 145.51 |
| Write | 1 | 64 | 6.154 | 147.70 |
| Alternating independent read/write | 1 | 32 | 5.714 | 137.14 |
| Alternating independent read/write | 1 | 64 | 6.169 | 148.06 |
| Read | 2 | 32 | 3.196 | 76.69 |
| Write | 2 | 32 | 3.196 | 76.69 |

The aggregate column multiplies the single-PC result by 24. It assumes independent PCs, uniform work and no shared bottleneck, and is not a multi-PC chip simulation. The two-PC regression checks credit independence only. Alternating traffic has independent read/write addresses; it is not copy and has no data dependency between the two directions. Real copy throughput remains unproven.

The current `tnca_line_axi` issues a read line no more often than every two cycles: `req_ready` is false while `issue_valid` is set, and accepting AR clears it at a later edge. The 512-bit write has the same minimum issue gap; narrower writes occupy more cycles. The experiment therefore rejects simple replication of the unchanged adapter onto 24 512-bit ports. Eight-line windows also fail by a large margin. A pipelined issue path and at least 32 outstanding lines are necessary candidates; 64 gives more tolerance at additional area cost.

## Implementation and checks

`StreamAdmission` is an optional argument to the bridge's existing batch replay. With it absent, the historical execution and skip paths remain unchanged. With it present, the controller is ticked directly, admission reserves one credit per line, both burst completions determine credit release, and the fixed return delay holds the credit until a later kernel edge. Partial burst admission retains its credit. Separate PCs have separate credit pools. The supplied trace gives earliest arrival; the replay determines the actual admission under credit and issue constraints.

The probe checks single-credit dense traffic against an independently generated explicit-arrival replay, for both reads and writes. Sparse traffic must exactly match the original replay, and a two-PC case checks independent timing and burst counts. Every swept case checks accepted/completed burst equality, line totals and the credit ceiling. Zero-window admission must refuse.

The unchanged bridge regression passes all ten tests, including the 200,000-request every-tick versus event-skip comparison with zero differing completions. Its raw record is `r41-hbmbw-bridge-test.json`. The new optional admission path deliberately disables skipping; that equivalence test qualifies the historical replay path, while the explicit-arrival tests check bounded admission.

## Reproduction

On GB10, with the pristine pinned Ramulator2 checkout and SystemC 3.0.2 installed:

```sh
cmake -S tools/performance_harness/systemc -B BUILD \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/tc_env/systemc-3.0.2/install" \
  -DRAMULATOR2_DIR="$HOME/tc_env/ramulator2"
cmake --build BUILD --parallel 4 --target tnca-stream-probe tnca-dram-bridge-test tnca-chip
BUILD/tnca-stream-probe tools/performance_harness/systemc/ramulator/hbm2_u280_pc.yaml
BUILD/tnca-dram-bridge-test tools/performance_harness/systemc/ramulator/hbm2_u280_pc.yaml
```

The raw JSON records the YAML hash, timing assumptions, all cases and the probe test status. Build/configuration logs and source/binary hashes accompany it. No HBM-task full implementation run or card access was performed.

## Remaining integration risks

Small contiguous allocations occupy one or a few physical PCs under the current BO map; naming more masters does not distribute their bytes. A usable stream design must solve physical striping and host DMA interpretation while preserving ordinary core and cache accesses. The model does not include AXI width conversion, the HBM switch, finite platform interconnect buffers, lane command distribution, cache ownership, core generation of stream work, copy data dependencies or reduction arithmetic. These require implementation, regressions and measurements before choosing an image.
