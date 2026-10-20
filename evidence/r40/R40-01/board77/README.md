# R40-01 full native U280 inventory — device capability checkpoint

The complete frozen tiny/small/medium inventory passes 77/77 on U280
ac580b0 at 100 MHz, with 5 warmup and 30 timed calls per point. Point IDs and
definition hashes exactly match spec/r40/suites/existing.json. Native device
and descriptor settings are recorded; numerical references match, fallback
and tensor staging/upload/download counts are zero. The initial 77/77
numerical run whose cold compiler copies contaminated the old count audit
remains on the host; these are separate correctly scoped steady records.

This run uses TNCA_RCTX_SPMD=0 because the loaded image predates the 48-bit
barrier correction, and an 8 GiB HBM BO. Its source archive/overlay is
9630b9b; software file hash is retained in summary.json and each raw record.
Full JSONL and stdout are compressed without dropping fields or samples.

The existing verifier records R40-01.device capability-pass. The strict
R40-01.device-time comparison refuses the undeclared HBM_BYTES/SPMD knob
differences from B39. The diagnostic comparison retains that refusal too;
no performance improvement or package acceptance is claimed. The verifier
now handles diagnostic refusal while preserving the independent inventory
verdict. The replacement 48-bit/grid/common-clock image is still building.
