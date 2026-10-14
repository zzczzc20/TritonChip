# Same-image CPU migration control

All 77 frozen points were attempted with 5 warmups and 30 timed calls on the
same common-clock HBM image as the native inventory. 75 points pass; two
llama_prefill points fail while combining graph timestamps across Sessions
replaced by the old code-capacity handler. The patched driver preserves the
Session and is being verified on those two points separately.

For the 75 successful pairs, CPU/native device-cycle S_geo is 0.998998. This
supports external-memory implementation as the main source of the measured
regression versus frozen on-chip B39; it does not sign off a speedup or replace
B39. Native measured tensor transfer is zero; CPU migration still transfers its
storages. Raw failed records are preserved alongside successful ones.
