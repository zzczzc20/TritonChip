# Three actual slices + one actual TMU under the full flow's rebuilt hierarchy.
# This is module-level screening, not full-system implementation/qualification.
# Usage: SOURCE_ROOT FRESH_OUTPUT KEEP_CACHE_BOUNDARIES(0|1)
if {[llength $argv] != 3} {error "expected SOURCE_ROOT FRESH_OUTPUT KEEP_CACHE_BOUNDARIES"}
set root [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set preserve [lindex $argv 2]
if {$preserve ni {0 1} || [file exists $out]} {error "invalid mode or reused output"}
file mkdir $out
set_param general.maxThreads 4
set f [open $root/flow/fpga/tnca_range_grid_ooc.v r]
set wrapper [read $f]
close $f
set marker {// P1_CACHE_BOUNDARY_ATTRIBUTE: experiment varies only this instance attribute.}
if {[string first $marker $wrapper] < 0} {error "wrapper marker missing"}
set attribute ""
if {$preserve} {set attribute {(* KEEP_HIERARCHY = "yes" *)}}
set wrapper [string map [list $marker $attribute] $wrapper]
set f [open $out/wrapper.v {WRONLY CREAT EXCL}]
puts $f $wrapper
close $f
read_verilog -sv [list $root/rtl/nuca/tnca_tmu.v $root/rtl/nuca/tnca_nuca_slice.v $out/wrapper.v]
set_property include_dirs [list $root/rtl/common] [current_fileset]
synth_design -top tnca_range_grid_ooc -part xcu280-fsvh2892-2L-e \
    -mode out_of_context -flatten_hierarchy rebuilt \
    -generic {N=3 MAINTENANCE=1 FULL_LINE_TAG=1 TAG_W=5 DCO_REGISTERED_RANGES=1 SYNC_TAG_BANK=1 REGISTER_VICTIM_OWNER=1 BALANCED_VICTIM=1 SETS=32 SET_W=5}
create_clock -name clk -period 10 [get_ports clk]
if {[llength [get_clocks -quiet clk]]!=1} {error "10 ns clock not established"}
report_utilization -hierarchical -file $out/utilization.rpt
report_utilization -file $out/total_utilization.rpt
report_timing_summary -delay_type min_max -file $out/timing.rpt
report_timing -max_paths 30 -nworst 1 -file $out/critical_paths.rpt
check_timing -verbose -file $out/check_timing.rpt
set f [open $out/cache_boundary_attributes.tsv {WRONLY CREAT EXCL}]
puts $f "cell\tkeep_hierarchy"
set boundaries [get_cells -hierarchical -quiet -filter {NAME =~ "g_tile*.u_slice"}]
if {[llength $boundaries]!=3} {error "expected all three cache boundaries"}
foreach cell $boundaries {
    set value [get_property KEEP_HIERARCHY $cell]
    if {$preserve && [string tolower $value] ni {true yes 1}} {error "cache boundary attribute not retained"}
    puts $f "[get_property NAME $cell]\t$value"
}
close $f
# A tag/age/owner register is not an architectural input to the TMU. If it
# reaches combinational cells reconstructed under TMU, cache logic escaped
# its physical owner. Keep raw startpoints/endpoints for review, not a gate.
set comb [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE && NAME =~ "tmu/*" && (REF_NAME =~ "LUT*" || REF_NAME =~ "MUXF*" || REF_NAME =~ "CARRY*")}]
if {![llength $comb]} {error "TMU combinational scope missing; cannot infer zero escapes"}
set pins [get_pins -quiet -of_objects $comb -filter {DIRECTION == OUT}]
set starts [get_cells -quiet -of_objects [all_fanin -flat -startpoints_only -to $pins]]
set ends [get_cells -quiet -of_objects [all_fanout -flat -endpoints_only -from $pins]]
foreach {name cells} [list tmu_comb_startpoints $starts tmu_comb_endpoints $ends] {
    set f [open $out/$name.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell $cells {puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]"}
    close $f
}
set f [open $out/scope.txt {WRONLY CREAT EXCL}]
puts $f "three actual slices, one actual TMU; no cores/NoC/HBM platform; no full-system gate claim"
puts $f "clock_target_ns 10; hierarchy rebuilt; preserve_cache_boundaries $preserve"
puts $f "all maintenance bases connected; external I/O delays unspecified; no routed timing or image"
puts $f "tmu_comb_cells [llength $comb]; sequential_startpoints [llength $starts]; sequential_endpoints [llength $ends]"
close $f
write_checkpoint $out/synth.dcp
puts "R40_RANGE_GRID_SCREEN_DONE"
exit
