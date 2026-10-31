# Actual BO/physical-PC translation

`kickoff/physical-dram-map-v1-evidence.tgz` contains ten completed remote
experiment steps and the recorded B40 allocation source identity. Its
source image is `a88ebdf5...`, BO base zero, local limit 131,072 bytes,
exposed size 4 GiB and physical tags PC0-PC15. The exported map SHA256 is
`a1019970fe9548fda5e9b7e44442a14f1a886decd397f472c498dd15f5970bc2`.

The model previously divided logical addresses directly. It now optionally
consumes this explicit mapping on both fast and slow paths:

`physical = BO base + logical - local limit`

The physical PC and offset come from actual topology-tagged bank ranges,
including partial allocations. They are not inferred from logical address
bits. `TNCA_DRAM_MAP` selects the map; identities bind its hash and report
the domain as `axi-bo-physical-pc`. Without it, the legacy domain is explicitly
`synthetic-logical`. SRAM and ranges outside the BO refuse, as do ambiguous
coverage, repeated PCs and unsupported geometry. A new Session's BO cannot
silently inherit this recorded allocation.

The exact same native artifact executes on both paths and independently
checks 256 output bytes at these boundaries:

| Logical address | Physical address | Actual PC |
|---:|---:|---:|
| 131,072 | 0 | 0 |
| 268,566,464 | 268,435,392 | 0 |
| 268,566,528 | 268,435,456 | 1 |
| 4,295,098,304 | 4,294,967,232 | 15 |

Both paths report exactly 640 bytes on PC0, 64 on PC1 and 64 on PC15,
including write allocation/writeback. The last logical line no longer
becomes PC16. This tiny case takes 483 fast-model cycles and 488 slow-model
cycles; those are uncalibrated predictions, not measured device timing.
Python and native tests additionally cover an offset BO, partial first/last
banks, deliberately non-arithmetic PC IDs, gaps, overlaps, tags and bounds.
Mapped standard Torch mm/addmm graphs and a depthwise representative graph
preserve their references; inconclusive timing remains distinct from PASS.

**Update 2026-10-21:** the bank/row mapping is now taken from the image's own HBM IP configuration (`hbm_controller/`: `ROW_BANK_COLUMN`, BG interleave, `USER_MC0_ADDR_BIT`) and the bridge implements it; the paragraph below records the earlier state. Bank/row mapping remained **unverified**. AMD describes a default Row Bank
Column layout with BG0 interleave and permits custom layouts. The bridge's
synthetic layout is therefore not certified by citing the default.
[AMD PG276 address-map discussion](https://docs.amd.com/r/en-US/pg276-axi-hbm/HBM-Address-Map-and-Protocol-Considerations).
AMD also describes a 12-entry controller queue and optional 64-entry AXI
queue, whereas this generic Ramulator controller uses separate 32-entry
read/write queues. These are distinct abstractions, with no demonstrated
equivalence. [AMD PG276 reordering discussion](https://docs.amd.com/r/en-US/pg276-axi-hbm/HBM-Reordering-Options).
The installed platform XSA exposes wrappers but has not supplied the actual
HBM bank/row customization. No physical-controller timing qualification or
board calibration is claimed. The final comment/doc corrections change no
mapping or timing code exercised by the archived campaign.
