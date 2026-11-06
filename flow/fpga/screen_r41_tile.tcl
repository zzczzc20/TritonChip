# screen_r41_tile.tcl -- out-of-context place-and-route screen of one TNCA tile
# cut from a kernel synthesis checkpoint (flow/fpga/synth_r40_multicore.tcl).
#
#   vivado -mode batch -source flow/fpga/screen_r41_tile.tcl \
#       -tclargs KERNEL_SYNTH_DCP TILE_INDEX SLR OUT_DIR
#
# The tile keeps the exact image parameters of the kernel synthesis. It is
# placed and routed alone at 10 ns inside one SLR pblock (CONTAIN_ROUTING),
# with its ports left to Vivado's out-of-context partition pins.
#
# This is a screen for intra-tile structures. Almost all of the attempt-3
# residual census is inside the tiles. It is not a substitute for the linked
# image's routed timing: there is no shell, no other tile, no SLR crossing and
# none of SLR0's static-region density, so it is optimistic on placement
# pressure. The baseline run calibrates it against the A3 census for the same
# families.
if {[llength $argv] != 4} {error "expected KERNEL_SYNTH_DCP TILE_INDEX SLR OUT_DIR"}
set dcp [file normalize [lindex $argv 0]]
set tile [lindex $argv 1]
set slr [lindex $argv 2]
set out [file normalize [lindex $argv 3]]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4

open_checkpoint $dcp
set cell [get_cells -hierarchical -filter "NAME =~ *g_tile\[$tile\].u_tile && IS_PRIMITIVE == 0"]
if {[llength $cell] != 1} {error "expected one g_tile\[$tile\].u_tile, found [llength $cell]"}
set top [get_property REF_NAME $cell]
write_checkpoint -force -cell $cell $out/tile_synth.dcp
close_design

create_project -in_memory -part xcu280-fsvh2892-2L-e
add_files $out/tile_synth.dcp
link_design -mode out_of_context -part xcu280-fsvh2892-2L-e -top $top
create_clock -name clk -period 10 [get_ports clk]
set pb [create_pblock tile_pb]
resize_pblock $pb -add SLR$slr
add_cells_to_pblock $pb -top
set_property CONTAIN_ROUTING 1 $pb
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
report_timing_summary -max_paths 50 -file $out/timing_routed.rpt
report_timing -max_paths 2000 -nworst 1 -unique_pins -file $out/top_paths_routed.rpt
report_design_analysis -timing -setup -max_paths 2000 -file $out/design_analysis_routed.rpt
set paths [get_timing_paths -max_paths 100000 -nworst 1 -unique_pins -slack_lesser_than 0 -setup]
set f [open $out/failing_endpoints.tsv w]
puts $f "slack_ns\tstartpoint\tendpoint\tlogic_levels\tdatapath_ns"
foreach p $paths {
    puts $f "[get_property SLACK $p]\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]\t[get_property LOGIC_LEVELS $p]\t[get_property DATAPATH_DELAY $p]"
}
close $f
puts "R41_TILE_SCREEN_DONE=$out failing=[llength $paths]"
