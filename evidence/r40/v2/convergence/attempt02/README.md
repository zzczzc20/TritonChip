# Final bounded implementation candidate

The failed image reconstructed cache-private lookup/replacement logic under
TMU and placed it across SLR boundaries. Explicit LOC-to-site queries confirmed
this detour even though named core/slice groups and pipeline state occupied
the intended regions. This candidate preserves cache instance hierarchy so
that tile pblocks own the complete private cone, while retaining required
code-queue protection and internal optimization.

The corrected screen connects all maintenance inputs and uses three real
slices sharing one real TMU under rebuilt synthesis and `opt_design`:

| Cache boundary | LUT | FF | Foreign combinational consumers by cache | Worst data ns | Levels | Estimated WNS ns |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Unpreserved | 99,976 | 15,509 | 4,144 / 5,208 / 4,153 | 8.383 | 33 | +1.374 |
| Preserved | 100,039 | 15,524 | 0 / 0 / 0 | 8.344 | 33 | +1.413 |

The selected-private-register fanout audit covers tag outputs, victim owners,
ages and valid bits across all module namespaces, stopping at sequential
endpoints. Counts are per-cache dependencies, not unique whole-design cells.
Both versions retain 96 RAM32M16 tag macros and 1,272 tag-output flops across
three caches. Macro children are not additional independent storage.

The preserved worst path is victim-owner flop to writeback-address RAM enable:
2.465 ns logic and 5.879 ns estimated net delay. Tag reads register at
E_SEL/E_LOOK; owners register at E_LOOK/E_MISS with epoch retry; victim ranking
uses a balanced tree without another request cycle. Functional/mutation and
source-regression evidence is linked from the P0 package. The decoder repair
is included in this candidate, unlike the failed cf330dc image.

These are module-level, unplaced estimates. Cores, NoC and the HBM platform
are absent; these reports cannot qualify full packing, routed setup/hold, CDC,
a loadable image or frozen workloads. `preflight.json` binds scope, exact
sources, configuration, clock, immutable contract and raw archives. Missing
performance remains not-measured.

The final full attempt retains three cores/outlets, HBM[0:31], 100 MHz,
32 sets/eight ways, two-stage links and SSI_SpreadSLLs. It runs once in a fresh
build directory and never retries or loads a board. Success must be followed
by actual placement, route, clock/setup/hold, CDC and image qualification,
then the unscaled >=64 MiB standard Torch checkpoint and complete frozen
inventories. Failure retains logs/DCP and does not trigger another attempt.
