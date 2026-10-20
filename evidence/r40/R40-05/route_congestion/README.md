# Multicore route failure, retained for diagnosis

The pre-ownership candidate at
`~/tnca_build/u280_r40multicore_final` failed route on the FPGA host.
`runme.log.gz` records Vivado exiting on 2026-10-15 at 05:06:28 host time
with `Route 35-3`: global congestion level 7. The intermediate routing
summary reports 1,101,457 failed nets and 10,322 node overlaps; this is
not a routed image or timing acceptance result.

Localized SLL demand reaches 204% of one column between SLR1 and SLR2,
while aggregate demand is 73%. A new run must address physical placement
and routing pressure rather than interpreting the aggregate as sufficient.
`image.json` preserves this candidate's configuration. It predates the
ownership/code-queue-preservation candidate, which is being audited
separately. No performance gate is signed off by this evidence.
