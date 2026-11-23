#==============================================================================
# ooc.tcl -- out-of-context synthesis, placement and routing of a K-fused lane screen top (R42-kfuse-rtl).
# Never touches a card. The U280 part, a 10 ns clock on an ideal source, as evidence/r41/pmac/lane/ooc/lane_ooc.tcl.
#
#   OOC_GENERICS="G=16 W=0 CUTS=8'b00111011 ..." vivado -mode batch -nojournal -nolog \
#       -source ooc.tcl -tclargs <top> <period_ns> <part> <out> <maxthreads> <files...>
#   env SYNTH_ONLY=1 stops after the post-synthesis reports.
# Reports: synth_* and route_*: _util.rpt, _util_hier.rpt, _summary.rpt, _paths.rpt (30 worst paths),
# _endpoints.csv (slack, start, end of the worst path into every register of the design).
#==============================================================================
set top    [lindex $::argv 0]
set period [lindex $::argv 1]
set part   [lindex $::argv 2]
set out    [lindex $::argv 3]
set thr    [lindex $::argv 4]
set files  [lrange $::argv 5 end]
file mkdir $out
foreach f $files { read_verilog -sv $f }
set_param general.maxThreads $thr
set gen [list]
if {[info exists ::env(OOC_GENERICS)]} {
    foreach g [split $::env(OOC_GENERICS) " "] { if {$g ne ""} { lappend gen -generic $g } }
}
set xdc [file join $out clk.xdc]
set fh [open $xdc w]; puts $fh "create_clock -name clk -period $period \[get_ports clk\]"; close $fh
read_xdc -mode out_of_context $xdc

set flat rebuilt
if {[info exists ::env(FLATTEN)] && $::env(FLATTEN) ne ""} { set flat $::env(FLATTEN) }
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy $flat {*}$gen

proc endpoints {tag out} {
    set fh [open [file join $out ${tag}_endpoints.csv] w]
    puts $fh "slack,start,end,levels,delay"
    foreach p [get_timing_paths -max_paths 20000 -nworst 1 -unique_pins -setup] {
        puts $fh [join [list [get_property SLACK $p] [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] \
                    [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p]] ","]
    }
    close $fh
}
proc reports {tag out} {
    report_utilization -file [file join $out ${tag}_util.rpt]
    report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out ${tag}_util_hier.rpt]
    report_timing_summary -no_header -no_detailed_paths -file [file join $out ${tag}_summary.rpt]
    report_timing -max_paths 30 -nworst 1 -unique_pins -file [file join $out ${tag}_paths.rpt]
    endpoints $tag $out
}
reports synth $out
write_checkpoint -force [file join $out synth.dcp]
if {[info exists ::env(SYNTH_ONLY)] && $::env(SYNTH_ONLY)} { puts "DONE (synth only)"; exit 0 }
opt_design
place_design
phys_opt_design
route_design
reports route $out
write_checkpoint -force [file join $out route.dcp]
puts "DONE"
