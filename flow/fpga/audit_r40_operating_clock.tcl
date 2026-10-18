# Post-route analysis only: no synth/opt/place/route/bitstream/checkpoint writes.
# Usage: vivado ... -tclargs ROUTED_DCP FRESH_OUTPUT_DIR INTEGER_MHZ
# Runtime MMCM programming and measured board clocks require separate evidence.
if {[llength $argv] != 3} {error "expected ROUTED_DCP FRESH_OUTPUT_DIR INTEGER_MHZ"}
lassign $argv checkpoint out mhz
if {![string is integer -strict $mhz] || $mhz < 1 || $mhz > 100} {
    error "operating frequency must be an integer in 1..100 MHz"
}
if {![file isfile $checkpoint] || [file exists $out]} {error "existing checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
set name clk_kernel_00_unbuffered_net
set old [get_clocks -quiet $name]
if {[llength $old] != 1} {error "expected one final-candidate kernel generated clock"}
set before [get_property PERIOD $old]
if {abs($before-10.0)>0.00001} {error "expected original 100 MHz routed checkpoint, got period $before"}
set mmcm level0_i/ulp/ulp_ucs/inst/aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00/inst/CLK_CORE_DRP_I/clk_inst/mmcme4_adv_inst
set source [get_pins -quiet $mmcm/CLKIN1]
set target [get_pins -quiet $mmcm/CLKOUT0]
if {[llength $source] != 1 || [llength $target] != 1} {error "kernel MMCM endpoint ambiguity"}
report_clocks -file [file join $out clocks_original.rpt]
# Preserve the master relationship used by the generated Vitis constraint.
# The original is multiply/divide 12/12. Re-analysis uses mhz/100 only.
create_generated_clock -name $name -divide_by 100 -multiply_by $mhz -source $source $target
set after [get_property PERIOD [get_clocks $name]]
if {abs($after-1000.0/$mhz)>0.0001} {error "requested operating period was not applied"}
report_clocks -file [file join $out clocks_operating.rpt]
report_route_status -file [file join $out route_status.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing.rpt]
report_timing -delay_type max -max_paths 25 -path_type full_clock_expanded -file [file join $out setup_paths.rpt]
report_timing -delay_type min -max_paths 25 -path_type full_clock_expanded -file [file join $out hold_paths.rpt]
report_clock_interaction -file [file join $out clock_interaction.rpt]
report_cdc -details -file [file join $out cdc.rpt]
check_timing -verbose -file [file join $out checks.rpt]
report_drc -file [file join $out drc.rpt]
set f [open [file join $out analysis.tsv] {WRONLY CREAT EXCL}]
puts $f "checkpoint\t$checkpoint"
puts $f "requested_mhz\t$mhz"
puts $f "original_period_ns\t$before"
puts $f "analysis_period_ns\t$after"
puts $f "scope\tunchanged routed netlist; hypothetical operating clock; no board qualification"
close $f
puts "R40_OPERATING_CLOCK_ANALYSIS_DONE"
exit
