# Actual TMU + slice boundaries, so endpoint arithmetic cannot move off-clock.
# vivado -mode batch -source THIS_FILE -tclargs SOURCE_ROOT NEW_OUTPUT MODE
if {[llength $argv] ni {3 5 6 7 8}} {error "expected SOURCE_ROOT NEW_OUTPUT MODE ?SETS SET_W? ?SYNC_TAG_BANK? ?REGISTER_VICTIM_OWNER? ?BALANCED_VICTIM?"}
set root [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set mode [lindex $argv 2]
set geometry [list]
if {[llength $argv] >= 5} {
    set sets [lindex $argv 3]
    set set_w [lindex $argv 4]
    if {$sets < 2 || $sets != (1 << $set_w)} {error "inconsistent cache geometry"}
    set geometry [list SETS=$sets SET_W=$set_w]
}
if {[llength $argv] >= 6} {lappend geometry SYNC_TAG_BANK=[lindex $argv 5]}
if {[llength $argv] >= 7} {lappend geometry REGISTER_VICTIM_OWNER=[lindex $argv 6]}
if {[llength $argv] == 8} {lappend geometry BALANCED_VICTIM=[lindex $argv 7]}
if {$mode ni {0 1}} {error "MODE must be 0 or 1"}
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
read_verilog -sv [list $root/rtl/nuca/tnca_tmu.v $root/rtl/nuca/tnca_nuca_slice.v $root/flow/fpga/tnca_range_slice_ooc.v]
set_property include_dirs [list $root/rtl/common] [current_fileset]
synth_design -top tnca_range_slice_ooc -part xcu280-fsvh2892-2L-e \
    -mode out_of_context -flatten_hierarchy none \
    -generic [concat [list MAINTENANCE=1 FULL_LINE_TAG=1 TAG_W=5 SLICE_ID=0 DCO_REGISTERED_RANGES=$mode] $geometry]
create_clock -name clk -period 10 [get_ports clk]
report_utilization -hierarchical -file $out/utilization.rpt
report_utilization -file $out/total_utilization.rpt
report_timing_summary -delay_type min_max -file $out/timing.rpt
report_timing -max_paths 20 -nworst 1 -file $out/critical_paths.rpt
# Report primitive mapping beneath the actual generated tag banks. A LUT
# count alone cannot establish that the tag array inferred memory.
set mapping [open $out/tag_mapping.tsv {WRONLY CREAT EXCL}]
puts $mapping "cell\tprimitive\tlocation"
foreach cell [get_cells -hierarchical -quiet -filter {NAME =~ "*g_sync_tags*" || NAME =~ "*g_async_tags*"}] {
    if {[get_property IS_PRIMITIVE $cell]} {
        puts $mapping "[get_property NAME $cell]\t[get_property REF_NAME $cell]\t[get_property LOC $cell]"
    }
}
close $mapping
check_timing -verbose -file $out/check_timing.rpt
write_checkpoint $out/synth.dcp
puts "R40_RANGE_SLICE_SCREEN_DONE=$out/synth.dcp"
exit
