# Usage: vivado -mode batch -source flow/fpga/audit_r40_image.tcl -tclargs BUILD_DIR
# Report generation only. Platform CDC paths and external I/O constraints
# still require explicit review; generating reports does not accept an image.
set build [lindex $argv 0]
if {$build eq ""} { error "BUILD_DIR is required" }
set audit [file join $build r40_postroute_audit]
if {[file exists $audit]} {error "preserve existing audit: $audit"}
file mkdir $audit
set_param general.maxThreads 4
open_checkpoint [file join $build tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp]
report_route_status -file [file join $audit route_status.rpt]
report_cdc -details -file [file join $audit cdc.rpt]
report_clock_interaction -file [file join $audit clock_interaction.rpt]
report_utilization -slr -file [file join $audit utilization_slr.rpt]
report_timing_summary -delay_type min_max -file [file join $audit timing.rpt]
report_bus_skew -file [file join $audit bus_skew.rpt]
check_timing -verbose -file [file join $audit checks.rpt]
puts "R40_POSTROUTE_AUDIT_DONE"
exit
