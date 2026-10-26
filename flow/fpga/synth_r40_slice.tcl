# Isolated, paired NUCA-slice screening; not a linked implementation gate.
# vivado -mode batch -source THIS_FILE -tclargs SOURCE INCLUDE_DIR NEW_OUTPUT
# The slice geometry matches tile zero of the three-core maintenance image.
if {[llength $argv] != 3} {error "expected SOURCE INCLUDE_DIR NEW_OUTPUT"}
set source [file normalize [lindex $argv 0]]
set include [file normalize [lindex $argv 1]]
set out [file normalize [lindex $argv 2]]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
read_verilog -sv $source
set_property include_dirs [list $include] [current_fileset]
synth_design -top tnca_nuca_slice -part xcu280-fsvh2892-2L-e \
    -mode out_of_context -flatten_hierarchy none \
    -generic {MAINTENANCE=1 FULL_LINE_TAG=1 TAG_W=5 SLICE_ID=0}
create_clock -name clk -period 10 [get_ports clk]
report_utilization -file $out/utilization.rpt
report_timing_summary -delay_type min_max -file $out/timing.rpt
report_timing -max_paths 20 -nworst 1 -file $out/critical_paths.rpt
check_timing -verbose -file $out/check_timing.rpt
write_checkpoint $out/synth.dcp
puts "R40_SLICE_SCREEN_DONE=$out/synth.dcp"
exit
