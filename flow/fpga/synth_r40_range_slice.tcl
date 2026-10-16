# Actual TMU + slice boundaries, so endpoint arithmetic cannot move off-clock.
# vivado -mode batch -source THIS_FILE -tclargs SOURCE_ROOT NEW_OUTPUT MODE
if {[llength $argv] != 3} {error "expected SOURCE_ROOT NEW_OUTPUT MODE"}
set root [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set mode [lindex $argv 2]
if {$mode ni {0 1}} {error "MODE must be 0 or 1"}
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
read_verilog -sv [list $root/rtl/nuca/tnca_tmu.v $root/rtl/nuca/tnca_nuca_slice.v $root/flow/fpga/tnca_range_slice_ooc.v]
set_property include_dirs [list $root/rtl/common] [current_fileset]
synth_design -top tnca_range_slice_ooc -part xcu280-fsvh2892-2L-e \
    -mode out_of_context -flatten_hierarchy none \
    -generic [list MAINTENANCE=1 FULL_LINE_TAG=1 TAG_W=5 SLICE_ID=0 DCO_REGISTERED_RANGES=$mode]
create_clock -name clk -period 10 [get_ports clk]
report_utilization -hierarchical -file $out/utilization.rpt
report_utilization -file $out/total_utilization.rpt
report_timing_summary -delay_type min_max -file $out/timing.rpt
report_timing -max_paths 20 -nworst 1 -file $out/critical_paths.rpt
check_timing -verbose -file $out/check_timing.rpt
write_checkpoint $out/synth.dcp
puts "R40_RANGE_SLICE_SCREEN_DONE=$out/synth.dcp"
exit
