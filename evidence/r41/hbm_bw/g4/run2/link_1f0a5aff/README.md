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

### Original image CDC identity comparison — 2026-10-30

A source/destination/rule comparison of the original run-2 `analysis_g4/cdc.rpt`
against the 56 individually listed findings in
`evidence/r40/v2/convergence/attempt03/qualified_clock90_after_load.json`
found **one new and one absent** finding, despite both reports having 56 total.
The new finding is CDC-10, depth 2, false-path exception, from
`level0_i/blp/blp_i/PLP/ss_fic/inst/user_fanout_firewall_and_regslices/axi_firewall_ctrl_user/inst/gen_mi.mi_r_error_i_reg/C`
to
`level0_i/blp/blp_i/PLP/ss_dma/inst/axi_intc_aggregated_mgmt/U0/INTC_CORE_I/INTR_DETECT_GEN[0].ASYNC_GEN.intr_ff_reg[0]/D`.
The absent finding has the same destination and rule, but source
`level0_i/blp/blp_i/PLP/ss_fic/inst/data_firewalls_and_regslices/axi_firewall_data_c2h/inst/gen_si.si_r_error_i_reg/C`.
Both reported endpoints are under the platform static logic. Their exact
identity is different: the previous count-only comparison cannot establish
that the prior individual review covers the new path. Circuit and constraint
review remains outstanding; the user's HBM clock exception does not by itself
constitute a CDC disposition.

### Original routed placement and pre-load checks — 2026-10-30

The read-only placement audit on the original `level0_wrapper_routed.dcp`
finished with exit 0. Cores 0, 1 and 2, and their corresponding local TMUs,
are entirely placed in SLR0, SLR1 and SLR2 respectively; it reports no
unplaced non-constant primitives and `failing_cores {}`. The raw x86 report
`~/r41run2/original_placement_20261030/actual_cores.rpt` has SHA-256
`eb5a884a26b7d2cab5fed2b26121aa044a257bcdb3551aaa94e5f29e02fd657f`.

On the board host, checks of the frozen source archive and extracted image
source, the original report-bound HBM clock exception including pulse width,
and the xclbin HBM mapping passed. Their artifacts are
`~/r41run2-auto/source_review_fullflow.json` (SHA-256 `b49c4ad9ad40666585d6519a6aceb49aac93d1b74c54762f6d280a52c3b7cfc4`),
`~/r41run2-auto/clock_review_pulse.json` (`c7249676f02803ed36c849ec0de54f1c2443d18827f90f33d6397099e2359892`),
and `~/r41run2-auto/hbm_review.json` (`94ef907d957074744673e09f6a6624c81b56c66faaaddbd5a9f95f1f90e20b16`).
These pre-load results are neither board measurements nor a CDC disposition.

### New CDC-10 trace — 2026-10-30

A read-only Vivado audit of the original routed checkpoint exited 0; its raw
[`original_cdc_trace.txt`](original_cdc_trace.txt) SHA-256 is
`ed2ec1ced819e81a3df0248ec3d4ba80bd462ef06e142fe3d4d4433ad7a65859`.
The reported source and destination each resolve to one fixed primitive in the
static `pblock_blp`. The source is an AXI firewall read-error register clocked
by `dma_ip_axi_aclk_1`; the destination is the first management interrupt
synchronizer register, marked `ASYNC_REG`. The destination's traced fan-in
includes 14 static-platform firewall error register clocks, including both the
newly reported source and the source absent from the new CDC report.

This trace establishes the structure and reconciles why both named sources
can feed the same endpoint, but it does not establish the pulse or glitch
behavior of all asynchronous inputs, or dispose the CDC-10 finding. The
original CDC review gate remains open pending that analysis.

The installed Vivado 2023.2 vendor RTL is a useful lead, not proof that it is
byte-identical to the source used to build the static platform. In its
`axi_firewall_v1_2_vl_rfs.sv` (SHA-256
`043f70b671e3523ddb75fd00207863dec9ce5c82086b88f03b1f954d2338c9cf`),
`mi_r_error_i` is a clocked state bit set by `r_error_set` and cleared by
reset or `r_error_reset`. Its `axi_intc_v4_1_vh_rfs.vhd` (SHA-256
`2108f76aa615004a7ccaf4afd3139b2565bed31d1ae03fdeb94fc2173ad149a6`)
contains an `ASYNC_REG` synchronizer chain for an asynchronous interrupt.
Neither inspection establishes a minimum pulse duration or excludes glitches
in the implemented multi-source aggregation.
