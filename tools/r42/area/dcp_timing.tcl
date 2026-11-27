#==============================================================================
# dcp_timing.tcl -- read-only timing census of a routed checkpoint (R42-area, clock study).
#   vivado -mode batch -nojournal -source dcp_timing.tcl -tclargs <routed.dcp> <outdir> <slack_limit_ns> <maxthreads>
# Writes (never a checkpoint): endpoints.csv  one row per endpoint, the worst setup path into it, for every endpoint whose
#   slack is below the limit: slack, startpoint pin, endpoint pin, startpoint clock, endpoint clock, logic levels, datapath
#   delay, logic delay, net delay, clock skew, requirement.
#   clocks.rpt, summary.rpt (report_timing_summary), top100.rpt (the 100 worst setup paths, full detail).
#==============================================================================
set dcp   [lindex $::argv 0]
set out   [lindex $::argv 1]
set lim   [lindex $::argv 2]
set thr   [lindex $::argv 3]
file mkdir $out
set_param general.maxThreads $thr
open_checkpoint $dcp
report_clocks -file [file join $out clocks.rpt]
report_timing_summary -no_header -no_detailed_paths -file [file join $out summary.rpt]
set fh [open [file join $out endpoints.csv] w]
puts $fh "slack,start,end,startclk,endclk,levels,delay,logic,net,skew,req"
set n 0
foreach p [get_timing_paths -max_paths 400000 -nworst 1 -unique_pins -setup -slack_lesser_than $lim] {
    puts $fh [join [list [get_property SLACK $p] [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] \
        [get_property STARTPOINT_CLOCK $p] [get_property ENDPOINT_CLOCK $p] [get_property LOGIC_LEVELS $p] \
        [get_property DATAPATH_DELAY $p] [get_property DATAPATH_LOGIC_DELAY $p] [get_property DATAPATH_NET_DELAY $p] \
        [get_property SKEW $p] [get_property REQUIREMENT $p]] ","]
    incr n
}
close $fh
puts "endpoints written: $n"
report_timing -max_paths 100 -nworst 1 -unique_pins -setup -file [file join $out top100.rpt]
puts DONE
