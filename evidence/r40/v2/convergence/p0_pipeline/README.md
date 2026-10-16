# P0 tag and owner pipeline screening

Functional validation passes for the synchronous tag banks and registered
victim-owner path, independently and together. Physical qualification remains
**not-measured**. The final full implementation attempt has not been started.

## Latest candidate: balanced victim ranking

The source-identical serial/balanced comparison in `balanced_ooc.tar.gz`
removes the downstream comparison ladder without another pipeline cycle:

| Ranking | Levels | Data ns | Logic ns | Estimated net ns | WNS ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 49 | 12.476 | 3.438 | 9.038 | -2.719 |
| Balanced | 34 | 8.579 | 2.522 | 6.057 | +1.178 |

The balanced worst path starts at `victim_owner_q_reg[7][2]` and ends at
writeback-address RAM write enable. Tag mapping remains 32 RAM32M16 macros
and 424 FDRE outputs. The balanced variant uses 36,096 LUTs versus the serial
control's 35,768: this correction improves the path, rather than reducing LUTs.
The earlier pipeline-only comparison below used a different source revision
and is retained separately; its 51-level path is not the paired serial control.

`balanced_functional.tar.gz` retains exact output/cycle comparisons across
replacement policies, geometry and metadata-epoch churn. Its proof extracts
the actual RTL leaf records and comparator tree and checks arbitrary keys,
claimed masks and dead payloads; encoded way-index inversion is rejected.
`balanced_validation.json` records scope, identities and missing measurements.
These results establish a setup screening checkpoint, **not** final three-core
100 MHz, hold timing, CDC, routability, image or workload acceptance.

`cut_diagnosis.tar.gz` additionally verifies mapped boundaries: 424 tag-output,
88 victim-owner and eight request-owner flops survive synthesis. The tag-output
to victim-owner input cone is 11 levels/2.502 ns in the pipeline-only snapshot.
Unneeded constant owner bits are trimmed; a conceptual 12-bit owner does not
imply twelve physical flops per way.

The request selected in E_SEL addresses each way's tag bank. Registered tag
outputs feed E_LOOK. Maintenance uses E_MTAG to fetch each scan set before
E_MSCAN; pending maintenance must still allow queued requests to read tags.
Fills write the selected bank, and the single engine prevents simultaneous
fill/lookup or fill/maintenance accesses. Tags are unreset; valid bits control
whether they may be consumed.

E_LOOK captures per-way owners and the request owner. E_MISS consumes them
only if the TMU tensor-table epoch still matches; otherwise it returns to
E_LOOK before any MSHR, request-queue or TLW effect. The TMU toggles the epoch
on descriptor commit or table clear, together with metadata updates. Only one
such write can commit per clock, and the checked interval is one edge, so a
one-bit epoch cannot wrap unnoticed. Dead hints and policy remain live at
E_MISS. The tag outputs remain stable during retries because this engine
cannot fill or scan concurrently.

`functional.tar.gz` preserves exact RTL, benches, runners, full differential
outputs, ownership regressions and both rejected mutations. The archive also
retains the initial unsupported one-port invocation of a bench that exercises
multiple ports; the corrected three/eight-port runs pass. `validation.json`
binds source hashes, target clock, immutable contract and missing image identity.
These checks do not substitute for the final image's native 77, HBM 14 or frozen
model inventories. The decoder repair is included in the integrated source.

Four source-identical, geometry-identical actual TMU+slice OOC variants completed
under `~/tnca_build/r40_p0ab_screen` on the FPGA host. The async
control is the new source's explicit A/B branch, rather than a byte-identical
copy of the earlier failed hardware RTL. Earlier prototype screen results
must not be substituted for the corrected candidate. OOC net delays are
estimates and cannot qualify 100 MHz, CDC or routability.

## Measured structure and remaining failure

`matched_ooc.tar.gz` retains source-identical variant reports and scripts.
These are unplaced synthesis estimates, not routed timing:

| Sync tags / owner registers | Levels | Data ns | Logic ns | Estimated net ns | WNS ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | 64 | 15.542 | 4.333 | 11.209 | -5.646 |
| 1 / 0 | 59 | 14.536 | 4.323 | 10.213 | -4.779 |
| 0 / 1 | 48 | 12.277 | 3.875 | 8.402 | -2.381 |
| 1 / 1 | 51 | 12.907 | 3.561 | 9.346 | -3.150 |

Both cuts are present in the combined variant: the worst path now starts at
`victim_owner_q_reg[0][4]` and ends at writeback-address RAM write enable.
The tag banks map to 32 RAM32M16 macros and 424 FDRE output registers.
The RAMD32/RAMS32 rows describe children of those macros and must not be
added as independent storage. This is LUTRAM, not BRAM and not a tag FF array.

The register cuts therefore exist, but the downstream serial victim ranking
still has seven dependent comparison/carry segments. This is insufficient
physical convergence and does not justify the final full implementation.
A balanced argmin correction is being screened separately against its serial
control, preserving policy, claimed-way exclusion and low-way tie-break.
Success requires reviewing the new cone, tag mapping, physical packing and
actual SLL/pipeline placements together. Failure retains the remaining cone
and does not trigger the last attempt.

`registered_range_proof.tar.gz` preserves the successful source-bound unsigned
membership and selection-composition proof, rejected mutations, and the
initial missing-solver tooling failure. The rerun used an isolated Z3 4.13.0.0
installation; it did not change the frozen software environment.
