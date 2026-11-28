#==============================================================================
# tile_screen2.tcl (R42-clk copy of tile_screen.tcl; flow directives selectable by environment variables TS_OPT, TS_PLACE, TS_PHYS1, TS_ROUTE, TS_PHYS2,
# TS_PHYS3 (an extra post-route phys_opt pass); unset = the shipped tile_screen.tcl flow) -- out-of-context place and route of one TNCA tile at a chosen period (R42-area, clock study).
# Same flow as flow/fpga/screen_r41_tile.tcl from the cut tile checkpoint (tile_synth.dcp), with the clock period a
# parameter. Optimistic like that screen: no shell, no other tile, no SLR crossing.
#   vivado -mode batch -nojournal -source tile_screen.tcl -tclargs TILE_SYNTH_DCP SLR PERIOD_NS OUT_DIR THREADS
# Writes (no checkpoint other than placed/routed in OUT_DIR): utilization_routed.rpt, timing_routed.rpt, endpoints.csv (every
# endpoint with slack below 1.5 ns: slack, start, end, levels, datapath delay, logic delay, net delay), top_paths_routed.rpt.
#==============================================================================
set dcp    [file normalize [lindex $::argv 0]]
set slr    [lindex $::argv 1]
set period [lindex $::argv 2]
set out    [file normalize [lindex $::argv 3]]
set thr    [lindex $::argv 4]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads $thr
create_project -in_memory -part xcu280-fsvh2892-2L-e
add_files $dcp
link_design -mode out_of_context -part xcu280-fsvh2892-2L-e -top tnca_tile
create_clock -name clk -period $period [get_ports clk]
set pb [create_pblock tile_pb]
resize_pblock $pb -add SLR$slr
add_cells_to_pblock $pb -top
set_property CONTAIN_ROUTING 1 $pb
report_utilization -file $out/utilization_synth.rpt
proc ts_env {n d} { if {[info exists ::env($n)] && $::env($n) ne ""} {return $::env($n)} else {return $d} }
set ts_opt   [ts_env TS_OPT ""]
set ts_place [ts_env TS_PLACE SSI_SpreadSLLs]
set ts_phys1 [ts_env TS_PHYS1 ""]
set ts_route [ts_env TS_ROUTE ""]
set ts_phys2 [ts_env TS_PHYS2 ""]
set ts_phys3 [ts_env TS_PHYS3 ""]
puts "FLOW opt=$ts_opt place=$ts_place phys1=$ts_phys1 route=$ts_route phys2=$ts_phys2 phys3=$ts_phys3"
if {$ts_opt ne ""} {opt_design -directive $ts_opt} else {opt_design}
place_design -directive $ts_place
if {$ts_phys1 ne ""} {phys_opt_design -directive $ts_phys1} else {phys_opt_design}
write_checkpoint -force $out/placed.dcp
report_timing_summary -no_detailed_paths -file $out/timing_placed.rpt
if {$ts_route ne ""} {route_design -directive $ts_route} else {route_design}
if {$ts_phys2 ne ""} {phys_opt_design -directive $ts_phys2} else {phys_opt_design}
if {$ts_phys3 ne ""} {phys_opt_design -directive $ts_phys3}
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/utilization_routed.rpt
report_timing_summary -max_paths 20 -file $out/timing_routed.rpt
report_timing -max_paths 500 -nworst 1 -unique_pins -file $out/top_paths_routed.rpt
set fh [open $out/endpoints.csv w]
puts $fh "slack,start,end,levels,delay,logic,net"
foreach p [get_timing_paths -max_paths 200000 -nworst 1 -unique_pins -slack_lesser_than 1.5 -setup] {
    puts $fh [join [list [get_property SLACK $p] [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] \
        [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p] [get_property DATAPATH_LOGIC_DELAY $p] \
        [get_property DATAPATH_NET_DELAY $p]] ","]
}
close $fh
puts DONE
