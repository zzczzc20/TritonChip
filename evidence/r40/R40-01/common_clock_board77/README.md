# Native common-clock U280: full inventory, performance gate still fails

2026-10-14. All 77 frozen tiny/small/medium points pass on the loaded common-clock
image. Point IDs and definition hashes exactly match the frozen existing suite.
Every point has 5 warmups and 30 timed calls; all measured host uploads/downloads
and fallback lists are zero. Descriptor generation is enabled. The only explicit
knobs are native device identity and RCTX_ROLES=0; SPMD uses its frozen default.
Software source is a6dd1e2; image source/UUID/hash and post-route audits are linked
by the summary and R40-04/grid_clock_image evidence. The xclbin has one core.

The unchanged device_time verifier records R40-01.device capability-pass, but
R40-01.device-time fails: S_geo=0.5872, worst medium FP32 transpose=0.128.
This is a measured regression, not a configuration refusal. No performance
acceptance or full R40 completion is claimed. A CPU migration run on this same
common-clock HBM image is running to separate native lowering from external-memory
cost; it is a diagnostic control and does not replace frozen B39.

Raw records and stdout are losslessly compressed. `gates.json` and `verdict.log`
retain the actual failing performance verdict alongside the capability result.
