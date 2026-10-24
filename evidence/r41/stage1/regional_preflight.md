# Regional resource preflight of the changed RTL

`regional_preflight.json` is produced by `tools/performance_harness/regional_preflight.py`. It adds the synthesized
LUT/register delta of every unit the next implementation changes to A3's routed per-SLR counts. Each delta is
weighted by that unit's instance count in each SLR of the three-core image.

The changed units:
- M1–M4, from paired unit OOC: old A3 source vs merged source;
- F07 (C1), from its paired slice screen in `f07_ooc/`.

| SLR | ΔLUT | ΔFF | LUT util A3 → estimate | CLB packing A3 → scaled estimate |
|---|---:|---:|---|---|
| SLR0 | +139 | +724 | 73.45 % → 73.48 % | 95.24 % → ~95.28 % |
| SLR1 | +857 | +314 | 75.24 % → 75.44 % | 88.39 % → ~88.62 % |
| SLR2 | +795 | +309 | 65.21 % → 65.39 % | 85.41 % → ~85.65 % |

Where the deltas come from:
- M4 (float-to-int rewrite): +221 LUT per core.
- C1: +574 LUT and +305 FF per slice.
- M1–M3: SLR0 is nearly neutral in LUTs because the three outlet adapters shrink by 218 LUTs each, but it gains
  registers.

The deltas are small. The risk remains SLR0's existing 95 % CLB packing, which these changes do not relieve.

Boundaries:
- Unit synthesis deltas are not placed, packed or routed results.
- CLB packing is scaled linearly with LUTs, which understates packing nonlinearity.
- F01–F06 and F08–F10 have no implementation yet. Their cost is unknown and is not estimated.
- This does not replace the Stage 2 routed census.
