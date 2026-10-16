# Diagnostic reports from an existing checkpoint, including failed routing.
# This script never modifies the source checkpoint or runs implementation.
# vivado -mode batch -source THIS_FILE -tclargs CHECKPOINT NEW_OUTPUT
if {[llength $argv] != 2} {error "expected CHECKPOINT NEW_OUTPUT"}
set checkpoint [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file isfile $checkpoint]} {error "checkpoint is absent: $checkpoint"}
if {[file exists $out]} {error "preserve existing reports: $out"}
file mkdir $out
set_param general.maxThreads 4
open_checkpoint $checkpoint
set status [open $out/report_status.tsv {WRONLY CREAT EXCL}]
puts $status "scope\tdiagnostic_only_no_image_qualification"
puts $status "checkpoint\t$checkpoint"
set failed 0
foreach {name command} [list \
    route_status [list report_route_status -file $out/route_status.rpt] \
    critical_paths [list report_timing -max_paths 30 -nworst 1 -path_type full_clock_expanded -file $out/critical_paths.rpt] \
    timing [list report_timing_summary -delay_type min_max -file $out/timing.rpt] \
    utilization_slr [list report_utilization -slr -file $out/utilization_slr.rpt] \
    utilization_hierarchy [list report_utilization -hierarchical -hierarchical_depth 6 -file $out/utilization_hierarchy.rpt] \
    congestion [list report_design_analysis -congestion -file $out/congestion.rpt] \
    clock_interaction [list report_clock_interaction -file $out/clock_interaction.rpt] \
    checks [list check_timing -verbose -file $out/checks.rpt]] {
    if {[catch {eval $command} message]} {
        puts $status "$name\tERROR\t$message"
        incr failed
    } else {
        puts $status "$name\tREPORT_WRITTEN"
    }
    flush $status
}
puts $status "report_errors\t$failed"
close $status
if {$failed} {error "$failed diagnostic reports failed; preserve partial results"}
puts "R40_CHECKPOINT_DIAGNOSIS_DONE"
exit
