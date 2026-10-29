# R41 run 2 (G4 frozen candidate, copied from run 1 analyze_run2.tcl): read-only post-route analysis of the routed DCP.
# One open_checkpoint, 2 threads. Never writes a checkpoint, never touches a board.
#   vivado -mode batch -nojournal -log OUT.log -source analyze_run2.tcl -tclargs ROUTED_DCP FRESH_OUT
# Reports = audit_r40_image.tcl's set (route status, CDC, clock interaction, SLR utilization,
# min/max timing summary, bus skew, check_timing) + screen-style top paths + census TSVs
# (tools/r40/census_failing_endpoints.tcl's columns) for failing and near-critical (<+0.5 ns) endpoints.
lassign $argv checkpoint out
if {![file isfile $checkpoint] || [file exists $out]} {error "existing checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
proc stamp {m} {puts "R41RUN2 [clock format [clock seconds] -gmt 1 -format %H:%M:%S] $m"}
stamp opened
report_route_status -file $out/route_status.rpt
report_timing_summary -delay_type min_max -max_paths 20 -file $out/timing.rpt
stamp timing
report_utilization -slr -file $out/utilization_slr.rpt
report_utilization -hierarchical -hierarchical_depth 7 -file $out/utilization_hier.rpt
report_clock_interaction -delay_type min_max -file $out/clock_interaction.rpt
report_cdc -details -file $out/cdc.rpt
report_bus_skew -file $out/bus_skew.rpt
check_timing -verbose -file $out/checks.rpt
stamp audit
set clock_name clk_kernel_00_unbuffered_net
proc slr_of_pin {pin} {
    set cell [get_cells -quiet -of_objects [get_pins -quiet $pin]]
    if {[llength $cell] != 1} {return "?"}
    set slr [get_slrs -quiet -of_objects $cell]
    if {[llength $slr] != 1} {return "?"}
    return [get_property NAME $slr]
}
proc dump {paths file} {
    set f [open $file {WRONLY CREAT EXCL}]
    puts $f "slack_ns\tstartpoint\tendpoint\tlogic_levels\tdatapath_ns\tlogic_ns\tnet_ns\tskew_ns\tstart_slr\tend_slr\tgroup"
    set tns 0.0
    foreach p $paths {
        set s [get_property SLACK $p]
        if {$s < 0} {set tns [expr {$tns + $s}]}
        set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
        puts $f [join [list $s $sp $ep [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p] \
            [get_property DATAPATH_LOGIC_DELAY $p] [get_property DATAPATH_NET_DELAY $p] \
            [get_property SKEW $p] [slr_of_pin $sp] [slr_of_pin $ep] [get_property GROUP $p]] \t]
    }
    close $f
    return $tns
}
# Kernel clock: every endpoint below +0.5 ns (failing subset = slack < 0), census columns.
set near [get_timing_paths -setup -group $clock_name -max_paths 60000 -nworst 1 -unique_pins -slack_lesser_than 0.5]
set tns [dump $near $out/near_endpoints_0p5.tsv]
stamp "kernel near=[llength $near] tns=$tns"
# All path groups, failing setup and hold (shell, platform, inter-clock).
set fs [get_timing_paths -setup -max_paths 20000 -nworst 1 -unique_pins -slack_lesser_than 0]
dump $fs $out/failing_setup_all_groups.tsv
set fh [get_timing_paths -hold -max_paths 20000 -nworst 1 -unique_pins -slack_lesser_than 0]
dump $fh $out/failing_hold_all_groups.tsv
stamp "all-groups setup=[llength $fs] hold=[llength $fh]"
report_timing -max_paths 2000 -nworst 1 -unique_pins -file $out/top_paths_routed.rpt
report_timing -of_objects [lrange $near 0 199] -path_type full_clock_expanded -file $out/top_paths_full.rpt
report_timing -hold -max_paths 200 -nworst 1 -unique_pins -file $out/top_hold_paths.rpt
if {[catch {report_design_analysis -timing -setup -max_paths 5000 -file $out/design_analysis.rpt} m]} {puts "WARN da $m"}
if {[catch {report_design_analysis -logic_level_distribution -logic_level_dist_paths 5000 -file $out/logic_levels.rpt} m]} {puts "WARN ll $m"}
if {[catch {report_design_analysis -congestion -file $out/congestion.rpt} m]} {puts "WARN cg $m"}
stamp done
puts "R41RUN2_ANALYSIS_DONE"
exit
