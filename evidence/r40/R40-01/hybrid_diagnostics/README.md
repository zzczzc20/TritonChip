# Hybrid diagnostic records — device-time not accepted

All board runs use the audited single-core hybrid image in
`evidence/r40/R40-04/hybrid_image/`, at 100 MHz, descriptor generation enabled,
RCTX_ROLES=0, five warmups and thirty timed calls. Software is `9fdbc84`, with
the explicit diagnostic overlays listed in `source_overlay.json`.

Initial native transpose/add/silu all pass numerically with zero steady staging.
Their medians are 152418/101481/366761 cycles. Frozen B39 is
91440/93817/376974 cycles: this initial run does not meet the performance floor.
CPU migration on the same image gives 90176/101481/366761 cycles. The initial
native run explicitly set TNCA_HBM_BYTES=1073741824; CPU control did not. These
configuration records are preserved and are not a formal matched A/B experiment.

Allocation probes produce 90176 cycles for transpose, both with default settings
and with the explicit HBM setting, and both four-call and thirty-five-call
protocols. Instrumented host time is not performance evidence. Only input/output
storage is live during the final probe launches, at 0x20000 and 0x48000; both fit
the local region. A standard uninstrumented repeat with post-measurement memory
metadata also gives thirty identical 90176-cycle samples. Its live/peak tensor
bytes are 163840/327680 and session peak is 327936 (including launch space).
The actual reported HBM window is 256 MiB. A further standard run with fresh
Inductor/Triton cache directories again gives 90176 cycles and passes numerics.

The initial slower result remains an unresolved discrepancy; none of these
repeats silently replaces it. Add still has an over-5% regression against B39.
There is no complete 77-point/three-block performance acceptance in this folder.
The raw point records, samples, stdout and allocation histories are compressed
without loss. No R40-01 device-time pass or full R40 sign-off is claimed.
