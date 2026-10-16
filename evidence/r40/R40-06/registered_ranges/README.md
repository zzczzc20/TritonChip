# Registered DCO ranges: functional proof and implementation screening

Required-gate status is unchanged. This package evaluates an optional internal
microarchitecture; it does not qualify a three-core image or establish graph
performance. `DCO_REGISTERED_RANGES=0` remains the default.

## Hypothesis and implementation

Precomputing a 65-bit interval endpoint when TMU metadata commits removes
query-dependent subtraction from owner lookup. The endpoint commits atomically
with base, size, validity and descriptor data. Its high bit represents unsigned
64-bit overflow or the existing unbounded, zero-size convention. The shared
owner function preserves highest-base selection and lower-index tie priority.
Both TMU and slice consume the same function; board parameters propagate the
option without changing the ISA or software artifact format.

Concurrency testing also exposed existing generic TMU defects: requester IDs
were truncated to two bits, arbitration selectors could not address ports 4–7,
and a three-bit accumulator lost eight simultaneous drops. The implementation
uses complete requester IDs, parameter-derived selector widths and a drop
accumulator wide enough for every port. Invalid requester IDs cannot modify
staging or global policy state.

## Evidence and limitations

`functional_and_proof.tar.gz` retains:

- The original failing registration test and its original RTL (`ranges18`).
- The authoritative source-bound proof (`ranges25_proof/report.json`): exact
  unsigned-64 membership, owner-order composition for table sizes 1–16, and
  counterexamples for overflow, zero-size and tie-priority mutations. This is
  a combinational compositional proof conditional on the metadata invariant;
  sequential correctness is checked separately on actual TMU RTL.
- Three-port and eight-port sequential tests (`ranges24`): atomic visibility,
  staged concurrent requesters 0/4/12, requester 15, invalid requesters 16/31,
  zero-size/overflow intervals, replacement, clear/reset, tile notices and drops.
- Four compiled RTL mutations rejected by simulation: requester aliasing,
  stale endpoints, port aliasing and drop-count overflow. Their expected
  simulation failures are evidence that these checks detect the defects.
- Integration evidence (`ranges23`): actual-core dispatch with registered ranges,
  both-policy slice runs, both board-top parameter elaborations, joint-module
  elaboration and Verilog-2005 compatibility. All 75 archived RTL files match
  the current source bytes. Paired policy trace hashes and counter rows match.

The earlier `ranges24_proof` is retained for history; `ranges25_proof` binds all
current proof dependencies, including the screen helper. Reproduction requires
Icarus, Yosys and Z3:

```sh
python3 tools/r40/dco_registered_ranges.py --out NEW_PROOF_DIR --z3 Z3_EXECUTABLE
for ports in 3 8; do
  iverilog -g2012 -I rtl/common -s tb_tmu_registered_ranges \
    -Ptb_tmu_registered_ranges.PORTS=$ports -o /tmp/tmu-ranges-$ports.vvp \
    tb/nuca/tb_tmu_registered_ranges.v rtl/nuca/tnca_tmu.v
  vvp /tmp/tmu-ranges-$ports.vvp
done
```

## Paired physical screening

`joint_ooc.tar.gz` includes immutable inputs, input hashes, experiment manifest,
commands, supervisor results and raw Vivado 2023.2 reports. Each variant uses
an actual three-port TMU and an actual 64-set, eight-way slice, including the
clocked endpoint producer. Both use the same U280 part, geometry and 10 ns
clock constraint. The source base was `8a3306f` with recorded uncommitted
changes; exact input hashes identify the experiment.

| Diagnostic | Baseline | Registered endpoints |
|---|---:|---:|
| LUTs | 83,353 | 78,543 |
| Flip-flops | 33,843 | 33,818 |
| Estimated WNS (ns) | -6.333 | -6.037 |
| Worst data-path delay (ns) | 16.090 | 15.794 |
| Logic levels | 69 | 65 |

LUT use falls 5.77%, but estimated timing improves only 0.296 ns. Both worst
paths still run from `slice/cur_addr_reg[7]` to writeback-address RAM write
enable. Logic delay is effectively unchanged (4.524 versus 4.523 ns).
The range arithmetic is therefore only part of the remaining timing problem.
No full-system implementation retry is justified by this result alone.

These are synthesis timing estimates with an ideal clock source and no I/O
constraints. They do not establish placement, routability, operating frequency,
CDC, graph speedup, or a loadable image. Device performance and region timing
error remain `not-measured` / `null`.

## Packaging

`package_audit.json` records the actual XO SHA and verifies that its shared
header and TMU/slice sources match the experiment. Packaging explicitly includes
headers without treating per-module function templates as global includes.
The parameter is set to one for this packaging check. An XO is an intermediate
artifact; it is not a loadable FPGA image.

## Next decision

The immediate full-system blocker is the failed original three-core route; see
[the convergence ledger](../../v2/convergence/README.md). Inspect its retained
error checkpoint for actual congestion and critical paths before selecting a
change. Success identifies a specific source or placement cause and permits a
bounded causal experiment. Failure preserves the reports and requires a
competing explanation; it does not justify directive polling. The slice-local
screen remains supporting evidence, not a diagnosis of the linked design.
