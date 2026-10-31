# R41 chat speed work

The active-event runtime change and this candidate were developed outside the
fixed board archive. All numbers here are uncalibrated DP SystemC/TLM; no
board speedup is claimed.

## Equal-row lowering

On a DP target the compiler now proves equal row addresses and masks before
sharing a load or row-wise computation; it keeps the shared row in scratch
to control register lifetime. `TNCA_EQUAL_ROWS=0` is the control. The
[matched official fused FP8 results](equalrow_default_20261031.tgz) used
the default division setting, fresh processes and independent exact DP
references. These steady calls each used one launch and had zero code loads
and evictions:

| M×N×K | Instructions/MAC before → after | Device cycles before → after | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.70867919921875 → 0.63916015625 | 161804 → 118469 | 0/256 each |
| 1×250×384 | 0.7098645833333334 → 0.6386770833333333 | 256585 → 195907 | 0/250 each |
| 1×1024×256 | 0.70867919921875 → 0.63916015625 | 536417 → 408542 | 0/1024 each |
| 16×256×256 | 0.047428131103515625 → 0.047428131103515625 | 186575 → 186575 | 0/4096 each |

The all-zero activation also matched bitwise. The FP8 and compiled-graph
suite passed 16/16 and the isolated candidate's 113-point default audit
compared 113 unchanged ([gate records](equalrow_gate_partial_20261031.tgz)).
Four [mask and wide-reduction cases](equalrow_mask_20261031.tgz) with equal
or differing addresses and masks matched bitwise with the optimisation off
and on. An [actual integration-tree TLM run](equalrow_integrated_20261031.tgz)
reproduced the 118469 cycles and 0/256 mismatches, with assembly SHA-256
`f59e9b8010bbffb3a1768119a318643610532deaa85875b69aec093e26ce0f7c`
identical to the isolated final candidate.

The [first integration attempt](equalrow_integrated_failed_20261031.tgz)
failed because `os` was not imported in that source tree; the successful
retry imports it locally. Two [integration audit attempts](equalrow_integration_audit_failed_20261031.tgz)
had 0/113 passing because the copied tree lacked the ISA file and then the
`tools/r38` files, respectively. These are not passing audits. The corrected
[actual integration-tree audit](equalrow_integration_audit_20261031.tgz)
compared 113/113 unchanged, with zero changed or uncompared. Board comparison
is pending.
