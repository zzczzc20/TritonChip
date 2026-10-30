# Run-2 full link: generated image fails qualification

`doc/engineering_discipline.md` applies. This is x86 Vivado/Vitis evidence, not a board run. Full implementation budget: run 2 of 5. Source commit `1f0a5aff4b5d2df6ef9f3805513d37638bf5bc0d`; image Tcl `evidence/r41/hbm_bw/g4/run2/image.tcl`. The x86 build root is `/home/vivado/r41run2/build_1f0a5aff`. Link started at `2026-10-29T15:29:01Z`; Vitis reported 4 h 40 m 27 s elapsed. The log, exact image parameters, source tarball identity and checkpoints remain at that host path.

Vitis completed bitstream generation and wrote `xo/tnca_u280.xclbin`, SHA-256 `1bf769fdbfbeb93586be86561664d43e785af3a01a739d19a3882647132d2d13`, UUID `01490bbd-d831-23bf-d783-56ec51bd71ce`. **This is not a qualified image and must not be loaded.** Although its DATA_CLK is 100 MHz, Vitis automatically scaled platform `hbm_aclk` from 450 MHz to 441.1 MHz. The runner correctly refused it with exit 3 (`link.stdout.gz`, `link.log.gz`, `tnca_u280.xclbin.info.gz`).

The post-link checker exited 1 (`check.json`): the original routed timing constraints show design setup WNS -0.045 ns, TNS -0.046 ns and two failing endpoints. Both are `hbm_aclk` paths in the platform `hmss_0` path 18, crossing SLR1 to SLR0; the worst is a zero-logic-level register-to-register path with 1.756 ns routed delay. The second endpoint is -0.002 ns. See `hbm_failing_paths.rpt.gz`, excerpted from the full x86 report `/home/vivado/r41run2/build_1f0a5aff/analysis_g4/timing.rpt`, SHA-256 `cd4c6254c960f5f368ea38f4a9e86b606747fba5d467fc4ac07c00f17971f577`.

The same routed report gives kernel-clock setup WNS +0.038 ns and no failing kernel setup endpoint; design hold WNS +0.001 ns with no failing hold endpoint. `route_status.rpt.gz` records 1,275,538 of 1,275,538 routable nets fully routed and zero routing errors. `check.json` counts 56 critical CDC findings in the same categories as the reviewed platform baseline; matching counts alone are not a new individual CDC review. SLR0 CLB occupancy is 95.56%. These positive observations do not override the clock-scaling and full-design timing failures. Full CDC report `/home/vivado/r41run2/build_1f0a5aff/analysis_g4/cdc.rpt`, SHA-256 `35617dbb86a0ddb86718ea622492634bf91ff3c51effe52e19b7a299841785c5`.

## Post-route diagnostic

The original routed checkpoint is preserved at `xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp`, SHA-256 `fc3f84bbaebd81c59a4289389c1edf3fdf69319479e8abc93914b26b01d95454`. After the full link ended, an independent diagnostic was launched on x86 at `/home/vivado/r41run2/recovery_postroute_20261029`, PID 553718. `recover.tcl` opens that checkpoint read-only, reports the original timing, runs post-route physical optimisation and routing in memory, then writes reports and a new checkpoint to the separate diagnostic directory. It does not run synthesis, generate an xclbin, alter the original checkpoint, or access the board; it is not a new full synthesis plus place-and-route run. Its outcome is pending and cannot be counted as a passing image. Any candidate must still pass full timing, routing, CDC, clock metadata, source identity and normal packaging checks before a board-load request.

### Recovery diagnostic: preliminary reports, 2026-10-30 UTC

The independent reroute finished and its timing summary reports setup WNS
+0.016 ns, TNS 0.000 ns, zero failing endpoints; hold WHS +0.001 ns,
THS 0.000 ns, zero failing endpoints. Pulse-width failing endpoints are zero.
The route report records 1,275,538/1,275,538 routable nets fully routed and
zero routing errors. The router log attributes improvement of a platform HBM
path to relocation of a Laguna register. These are diagnostic Vivado results at
the original constraints, not an xclbin or board evidence.

The CDC report lists 56 critical findings (CDC-7: 13, CDC-10: 7, CDC-11: 32,
CDC-13: 4). Counts match the previous report, but a count match is not an
individual review and is not CDC sign-off. Vivado also requests a separate
`report_bus_skew`; that check remains outstanding. The diagnostic process was
still writing its recovered checkpoint when last checked. Reports on x86 at
`~/r41run2/recovery_postroute_20261029`: `after_timing.rpt` SHA-256
`fbdc20371adc1fd8a387e13e8187f76ed2b495929a5b68fb9648bf4b842e187d`,
`after_route.rpt` `a9f8adc3cd14885894ad79341aa1ba3673432cd5fac136d778c8e2fd141c4980`,
`after_cdc.rpt` `28a1cd250f8912685edd36b641f3c99a53b26ee55b9c5c3acb741a97400fcf3d`.
The original scaled xclbin remains unqualified; normal Vitis packaging and
clock-metadata verification are required for any new image.

A further comparison of the 56 detailed critical CDC rows against the original
run-2 `analysis_g4/cdc.rpt`, ignoring only each displayed row number, found
zero added or removed rows. This establishes equality to that prior run-2 CDC
report, not an independent disposition of the 56 findings.

The recovery diagnostic completed with exit 0 and produced
`recovered_routed.dcp` (620 MiB; SHA-256
`256788a457c9165f656e20184afd5289a152aea5f9e1d95ae819053613c12a6a`).
After its Vivado process exited, a separate read-only bus-skew and check-timing
audit was launched with `audit_skew.tcl` on that checkpoint (x86 PID 561140).
Results are pending; this audit does not create or load an image.
