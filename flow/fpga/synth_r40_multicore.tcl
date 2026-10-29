# Diagnostic OOC gate: not a substitute for linked placement/route acceptance.
# vivado -mode batch -source ... -tclargs <source-root> <config.tcl> <output-dir>
set root [file normalize [lindex $argv 0]]
source [file normalize [lindex $argv 1]]
set out [file normalize [lindex $argv 2]]
file mkdir $out
set_param general.maxThreads 4
set sources [list]
foreach dir {common core gen nuca noc tile fpga} {
    foreach source [lsort [glob $root/rtl/$dir/*.v]] {lappend sources $source}
}
foreach source [lsort [glob $root/rtl/*.v]] {lappend sources $source}
lappend sources $root/rtl/board/tnca_multicore_hbm_xo.v
read_verilog -sv $sources
set_property include_dirs [list $root/rtl/common $root/rtl/noc] [current_fileset]
set generics [list]
dict for {name value} $TNCA_IMAGE_PARAMS {lappend generics "$name=$value"}
synth_design -top tnca_multicore_hbm_xo -part xcu280-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy none -generic $generics
create_clock -name ap_clk -period 10 [get_ports ap_clk]
report_utilization -hierarchical -file $out/utilization.rpt
report_timing_summary -file $out/timing.rpt
check_timing -verbose -file $out/check_timing.rpt
write_checkpoint -force $out/synth.dcp
puts "R40_MULTICORE_SYNTH_DONE=$out/synth.dcp"
