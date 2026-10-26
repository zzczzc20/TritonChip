# P1 registered FIFO output screen

Required gate status: unchanged, **not-measured**. The candidate is isolated;
production RTL and the running immutable `0c39793` implementation are unchanged.
No full-system implementation budget was consumed.

## Physical hypothesis

The running candidate's placed report shows no Laguna TX/RX register usage,
and its elastic links use two-entry FIFOs with asynchronous `mem[rptr]`
outputs. Dedicated-register usage alone does not establish a missing pipeline.
The concrete hypothesis is narrower: moving payload selection before a head
register could provide a direct FF launch point for selected crossing senders.
It does not reduce crossing bit counts or establish a remedy for the router's
213% estimated SLR1–2 hotspot column.

## Functional and mapping evidence

The ARM jump host ran Yosys 0.33 temporal induction against the actual FIFO
source from `0c39793`, with only its module name changed. For an eight-bit
payload, arbitrary input data, valid, ready and subsequent resets, the proof
establishes identical input readiness, output validity, occupancy and valid
output data after the initial reset. Invalid output data is unobserved.
Capacity, accepted transfer cycles and backpressure behavior are preserved.
The proof is unbounded in time; its tested payload width is eight bits.

A mutation replacing the queued tail with arbitrary input during a full-queue
pop fails equivalence. The archive retains its bounded counterexample VCD.
Separate UltraScale Plus logic-mapping screens cover widths 64, 137, 220,
517 and 582, including the actual NoC and line-link payload widths.

For a 582-bit line-request queue:

| Mapping result | Current FIFO | Registered head candidate |
| --- | ---: | ---: |
| Direct payload-output driver | RAM32M16, 582 bits | FDRE, 582 bits |
| FDRE cells | 5 | 1,166 |
| Logic LUT cells | 7 | 587 |
| RAM32M16 root macros | 42 | 0 |

These are Yosys technology-mapping results, not Vivado mapping or physical
area. The released RAM macros consume LUT-memory resources and must be
accounted separately; macro and child counts must never be added together.
The increased FF/logic demand and released distributed RAM can affect
packing differently. Neither CLB occupancy nor timing benefit follows from
these counts alone. Routed timing, actual Laguna packing, congestion and
instrumented workload performance remain not-measured.

## Decision and reproduction

Do not replace all FIFOs on this evidence. Retain the candidate for a matched
physical screen of senders selected from the current implementation's actual
terminal paths and crossing attribution. Require a demonstrated path benefit
and acceptable packing/congestion before integrating it into another full
candidate. Preserve required queue boundaries and metadata ordering.

`report.json` binds source hashes, tool version, mapping inventories and the
SHA256 of `screen.tar.gz`. The archive contains both implementations, formal
harness, induction log, mutation log/VCD, mapping scripts, logs, exit codes
and top-module mapped netlists. Extract into a fresh directory and run:

```sh
yosys -Q -T -s prove.ys
yosys -Q -T -s mutation_witness.ys
yosys -Q -T -s ref582.ys
yosys -Q -T -s head582.ys
```

The mutation command intentionally prints `model found: FAIL!`; its normal
process exit does not mean equivalence passed. The induction proof must print
`Induction step proven: SUCCESS!`.
