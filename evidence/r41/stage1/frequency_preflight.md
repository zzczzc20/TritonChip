# Frequency preparation and regional risk

The source audit and `kickoff/frequency-preflight-v2-evidence.tgz` retain the
complete raw residual population: 12,799 endpoints, 753 families, 94 unit
pairs and -4,110.801 ns TNS. This population comes from the historical routed
A3 DCP after the documented M1-M3 cone exclusions. It is not new routed
evidence for the merged RTL.

Rebuilding the frontier with the original classifier reproduces 95 MHz's
170 families / 8,959 endpoints and 100 MHz's entire population. At 93 MHz,
the exact cutoff produces 89 families / 5,874 endpoints; the old JSON says
86 / 4,338. The old README already says 89 / 5,874. The cutoff rounding and
three-decimal source slacks affect this boundary. Both original evidence
and the discrepancy are preserved; no historical frontier was rewritten.

The report reads F01-F10 verbatim from the author's execution document and
records responsible module hashes and verification requirements. This fixes
the proposed next implementation cohort without claiming the ten changes
have been implemented or that they cut every residual family.

Current RTL is byte-identical to the d2cc850 OOC source tree. The separate
source-equivalence audit permits reuse of those unit/kernel synthesis
diagnostics, including kernel unplaced WNS +1.740 ns and TNS zero. C1 F07
has functional evidence but has not been physically screened.

The raw historical A3 routed SLR report shows CLB packing of 95.24%, 88.39%
and 85.41%. SLR0 packing remains a material risk despite available registers.
The merged kernel synthesis report is unplaced and cannot predict regional
packing, SLL demand or routed timing. Four preflight tests cover the valid
population and changed coverage, slack and regional reports.

Stage 2 remains unready: candidate implementations and functional screens,
timing/area/regional deltas, a regenerated census, then full routed 100 MHz
and board verification remain outstanding. No new Stage 1 bitstream is
required or launched by this preflight.
