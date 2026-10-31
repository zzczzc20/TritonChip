# Coupled wide-path model experiment

This is an exploratory SystemC/native-ISA experiment, not RTL or board evidence. The result changes the next action: a 512-bit core/coalesced-home path by itself is insufficient for the author's several-times operator-speed acceptance. Keep the full image unqualified.

## Method and scope

`tools/r41/wide_path_model_tests.py` runs the existing native-chip numerical validation in five timeline configurations. It adds a masked, unaligned vector copy crossing both 8-byte and 64-byte boundaries and a sustained three-program integer vector add. Each program processes 4,096 elements in its own 16 KiB slice, using two vector loads, integer add, store and a counted loop. The numerical reference checks every output byte. This microbenchmark uses native instructions; it is not standard torch and its speed ratios cannot satisfy the operator gate.

The optional `tl_core_beat_bytes` hypothesis groups the existing exact 64-bit functional word components into one aligned 8/16/32/64-byte transaction. A group reserves one core port cycle, finite LSU beat credit, request/response NoC capacity, one home/cache service and the original outlet/HBM path. `tl_noc_payload_bytes` independently controls useful payload serialization for the grouped transaction. Groups never cross the configured region or 64-byte cache line. Arithmetic, masks and numerical conversions retain the existing execution. Tensor DMA and atomics are not widened. The default eight-byte path calls the original memory transaction unchanged.

The wide protocol is hypothetical: its extra useful data serializes through narrow NoC links; fixed protocol headers retain the historical model's one-flit assumption. Real packet headers, physical link packing, backpressure queues and atomic/TMU handling still require RTL design and validation. This is not a throughput guarantee. The fast DRAM path and synthetic logical address map were used; controller service must subsequently be checked with the physical map and Ramulator path.

## Results

| Hypothesis | Stream-add cycles | Predicted ratio vs baseline | FP32 add check cycles | SFU check cycles |
| --- | ---: | ---: | ---: | ---: |
| Existing model | 59,179 | 1.000 | 674 | 430 |
| Explicit eight-byte defaults | 59,179 | 1.000 | 674 | 430 |
| 64-byte core/coalesced home, eight-byte NoC payload | 43,262 | 1.368 | 626 | 402 |
| 64-byte core/coalesced home and NoC payload | 39,934 | 1.482 | 584 | 378 |
| Above plus one-cycle hit/wake occupancy, two-cycle hit latency | 39,278 | 1.507 | 578 | 375 |

All five configurations pass the existing integer/FP32/FP16 add, reductions, shuffle, SFU, grid and refusal checks, plus the new unaligned/masked and sustained-add cases. All deterministic counters, visible cycles and checked-byte totals match between absent optional keys and explicit eight-byte defaults. The FP16 tiny check actually regresses from 352 to 355 cycles with a narrow NoC, illustrating why isolated width ratios cannot stand in for operator measurements.

For stream-add, each core's memory-port service falls from 6,144 to 768 cycles, exactly eightfold. Home engine service falls from approximately 40,000 to 8,200 cycles per home. Nevertheless, with the wide NoC each core still records about 19,300 dependency-wait cycles and 15,500–16,000 LSU-wait cycles. These overlapping attribution counters must not be summed into a critical path. They identify the next investigation: independent-vector scheduling and finite LSU command/response capacity, alongside cache-miss/outlet latency. Wider service alone has not removed dependent instruction waits. Reducing home occupancy further yields only another 1.7% in this workload.

## Reproduction and provenance

Build the SystemC target `tnca-chip`, then run:

```sh
python3 tools/r41/wide_path_model_tests.py --binary /absolute/build/tnca-chip --out /absolute/fresh/output
```

GB10 used at most two build cores and a 16 GiB process address-space limit. `raw.tgz` contains all five configurations, native code, data, references, command scripts, output logs, identity hashes, counters and the successful build log. `sweep.json` exposes the summaries without archive extraction.

The first attempt through the sealed R40 runner refused the current contract-document hash: the historical lock expects a different document. Its failure log is preserved in the archive. Neither the lock nor document was changed. The R41 test instead invokes the model directly, preserving command-arity refusal and recording explicit exploratory identities. It makes no frozen-R40 qualification claim and does not alter the sealed runner. The existing generated validation report's RF-coverage limitations remain in force.

No FPGA was opened, programmed or reset. No full-run allocation was consumed. Next evaluate actual compiled torch traces and changes that overlap independent vectors; do not select a wide-line RTL image on the strength of this 1.5x prediction.
