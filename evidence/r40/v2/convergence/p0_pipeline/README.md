# P0 tag and owner pipeline screening

Functional validation passes for the synchronous tag banks and registered
victim-owner path, independently and together. Physical qualification remains
**not-measured**. The final full implementation attempt has not been started.

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

Four source-identical, geometry-identical actual TMU+slice OOC variants are
running under `~/tnca_build/r40_p0ab_screen` on the FPGA host. The async
control is the new source's explicit A/B branch, rather than a byte-identical
copy of the earlier failed hardware RTL. Earlier prototype screen results
must not be substituted for the corrected candidate. OOC net delays are
estimates and cannot qualify 100 MHz, CDC or routability.

The remaining physical acceptance answers are the combined candidate's new
worst path, logic levels, logic/estimated-net delay, actual tag primitives and
proof that the mapped owner registers separate the original cone. Next command:
`tail -60 ~/tnca_build/r40_p0ab_screen/case11.log`. Successful screening triggers
review of those reports and the failed-design SLL/placement evidence; it does
not automatically authorize the final implementation. A retained long cone or
mapping failure requires diagnosis before that attempt is considered.
