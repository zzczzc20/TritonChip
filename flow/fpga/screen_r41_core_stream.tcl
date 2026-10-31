# Combined free OOC physical screen for the core kernel and 32-lane stream bank.
# Arguments: RTL_ROOT IMAGE_CONFIG OUT_DIR
if {[llength $argv] != 3} {error "expected RTL_ROOT IMAGE_CONFIG OUT_DIR"}
set root [file normalize [lindex $argv 0]]
source [file normalize [lindex $argv 1]]
set out [file normalize [lindex $argv 2]]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
set sources [list]
foreach dir {common core gen nuca noc tile fpga} {
    foreach source [lsort [glob $root/rtl/$dir/*.v]] {lappend sources $source}
}
foreach source [lsort [glob $root/rtl/*.v]] {lappend sources $source}
lappend sources $root/rtl/board/tnca_multicore_hbm_xo.v $root/flow/fpga/r41_core_stream_screen.v
read_verilog -sv $sources
set_property include_dirs [list $root/rtl/common $root/rtl/noc] [current_fileset]
set generics [list]
dict for {name value} $TNCA_IMAGE_PARAMS {lappend generics "$name=$value"}
synth_design -top r41_core_stream_screen -part xcu280-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy none -generic $generics
create_clock -name ap_clk -period 10 [get_ports ap_clk]
set pb [create_pblock stream_bank_pb]
resize_pblock $pb -add SLR0
add_cells_to_pblock $pb [get_cells stream_bank]
set_property CONTAIN_ROUTING 1 $pb
foreach {tile slr} {0 1 1 1 2 2} {
    set cell [get_cells -hierarchical -filter "NAME =~ *g_tile\[$tile\].u_tile && IS_PRIMITIVE == 0"]
    if {[llength $cell] != 1} {error "expected one tile $tile, found [llength $cell]"}
    set pb [create_pblock tile${tile}_pb]
    resize_pblock $pb -add SLR$slr
    add_cells_to_pblock $pb $cell
    set_property CONTAIN_ROUTING 1 $pb
}
report_utilization -file $out/utilization_synth.rpt
opt_design
place_design -directive SSI_SpreadSLLs
phys_opt_design
write_checkpoint -force $out/placed.dcp
report_timing_summary -file $out/timing_placed.rpt
route_design
phys_opt_design
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/utilization_routed.rpt
report_timing_summary -max_paths 100 -file $out/timing_routed.rpt
report_design_analysis -timing -setup -max_paths 2000 -file $out/design_analysis_routed.rpt
set paths [get_timing_paths -max_paths 100000 -nworst 1 -unique_pins -slack_lesser_than 0 -setup]
puts "R41_CORE_STREAM_SCREEN_DONE=$out failing=[llength $paths]"
