#==============================================================================
# ooc.tcl -- out-of-context synthesis, placement and routing of an area-study screen top (R42-area).
# Never touches a card. Part xcu280-fsvh2892-2L-e, clock on port clk (ideal source), as tools/r42/kfuse_rtl/ooc.tcl.
#
#   OOC_GENERICS="IMPL=2 LAT=2" vivado -mode batch -nojournal -nolog -source ooc.tcl \
#       -tclargs <top> <period_ns> <part> <out> <maxthreads> <files...>
#   env SYNTH_ONLY=1 stops after the synthesis reports; env FLATTEN (default rebuilt); env PBLOCK="SLICE_X..:SLICE_X.." optional.
# Reports: synth_* and route_*: _util.rpt, _util_hier.rpt (depth 5), _summary.rpt, _paths.rpt (30 worst), _endpoints.csv.
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
set inc [list]
if {[info exists ::env(INCDIRS)] && $::env(INCDIRS) ne ""} { set inc [list -include_dirs $::env(INCDIRS)] }
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy $flat {*}$gen {*}$inc

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
    report_utilization -hierarchical -hierarchical_depth 5 -file [file join $out ${tag}_util_hier.rpt]
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
