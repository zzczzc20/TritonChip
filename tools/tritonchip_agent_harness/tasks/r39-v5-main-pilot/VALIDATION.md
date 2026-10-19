# R39 v5 main-line pilot: provenance and validation

This is a **pilot task**, not the earlier `replay-mzero` framework regression.
Its pinned base `054569984293a03ceb55a46e1b63317f9b34af63` is a
descendant of `main@ac28ca182b249e293731161656a4e1e638b9bd8e`.
That main descends from the signed-off R39 v5 software and includes its
profile. The 77-point board result in `doc/r39_report.md` belongs to
`be87779` with `u280_r39h`; it has not been repeated on `main@ac28ca1`.
The PR9 descendant adds the profile-header, observer and named-image support needed
by the harness. It does not remove the `m.zero` lowering. The profile is
`core-vrf-sync-v5`, the U280 `u280_r39h` design at the task's fixed 100 MHz.

The pilot retains the five matrix-related tiny points used to validate the
observer, so changes in the starting code can be compared directly. The
formal benchmark's task definition, hidden-set breadth and P2.2 result
provenance remain separate acceptance work.

## Checks on 2026-10-19

- A fresh single-commit export from the pinned base contained 570 files.
  `archbench/` and both harnesses were absent, a no-change gate passed, and
  no exact point ID or point-shape suffix from this task was found in the
  exported text files.
- On the Docker test host, `run-task ... --agent noop` completed with a
  platform observer for every point, `validity=valid`, gate `PASS`, and
  validation **3/3** plus final **2/2** passing. Every speedup was exactly
  **1.000**. The observed validation counts were 1497, 8285 and 2190;
  final counts were 4923 and 9727 cycles.
- Run ID: `r39-v5-main-pilot-noop-20261019T044648Z-300e`.
  `run.json` SHA-256:
  `62e78b2068b63825fd56155ba649b657d89d30b3450d3a76b33f083963dd2dbe`.
  Toolchain image ID:
  `sha256:88bc6930ffef777487dc568b3349ddcdc487b2905f6812fad02a7be5f93966e4`.

This check used simulation only. It did not run an agent, link a candidate,
or program a card. The earlier PR9 record certifies the v5 **base hardware**;
it is not a certification of a hardware-changing candidate from this task.
