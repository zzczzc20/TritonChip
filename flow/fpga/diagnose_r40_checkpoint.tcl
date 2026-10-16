# Diagnostic reports from an existing checkpoint, including failed routing.
# This script never modifies the source checkpoint or runs implementation.
# vivado -mode batch -source THIS_FILE -tclargs CHECKPOINT NEW_OUTPUT
# Rank logical timing cones within the existing report's endpoint sample.
# No timing query is repeated, and this is not an exhaustive path census.
proc r40_timing_family {pin} {
    set parts [split $pin /]
    set parts [lrange $parts 0 end-1]
    if {[regexp {^RAM[A-Z0-9_]+$} [lindex $parts end]]} {
        set parts [lrange $parts 0 end-1]
    }
    set leaf [lindex $parts end]
    regsub -all {\[[0-9]+\]} $leaf {[*]} leaf
    regsub {_reg_[0-9_]+$} $leaf {_reg} leaf
    regsub {_replica_[0-9]+$} $leaf {_replica} leaf
    set parts [lreplace $parts end end $leaf]
    return [join $parts /]
}
proc r40_rank_timing_report {input output} {
    set f [open $input r]
    set groups [dict create]
    set row [dict create]
    # Sentinel flushes the last endpoint without reading or querying a DCP.
    set lines [split [read $f] \n]
    close $f
    lappend lines {Slack (END)}
    foreach line $lines {
        if {[regexp {^Slack } $line]} {
            if {[dict exists $row levels]} {
                set key [list [r40_timing_family [dict get $row source]] [r40_timing_family [dict get $row destination]]]
                if {![dict exists $groups $key]} {
                    dict set groups $key [dict create count 0 worst $row]
                }
                set group [dict get $groups $key]
                dict incr group count
                if {[dict get $row slack] < [dict get $group worst slack]} {dict set group worst $row}
                dict set groups $key $group
            }
            set row [dict create]
            if {[regexp {^Slack .*?:\s*(-?[0-9.]+)ns} $line whole value]} {dict set row slack $value}
        } elseif {[regexp {^\s*Source:\s+(\S+)} $line whole value]} {
            dict set row source $value
        } elseif {[regexp {^\s*Destination:\s+(\S+)} $line whole value]} {
            dict set row destination $value
        } elseif {[regexp {^\s*Data Path Delay:\s+([0-9.]+)ns\s+\(logic\s+([0-9.]+)ns.*route\s+([0-9.]+)ns} $line whole total logic net]} {
            dict set row delay $total
            dict set row logic $logic
            dict set row net $net
        } elseif {[regexp {^\s*Logic Levels:\s+([0-9]+)} $line whole value]} {
            dict set row levels $value
        }
    }
    set ranked [list]
    dict for {key group} $groups {lappend ranked [list [dict get $group worst slack] $key $group]}
    set f [open $output {WRONLY CREAT EXCL}]
    puts $f "source_family\tdestination_family\tsampled_endpoints\tworst_slack_ns\tdata_delay_ns\tlogic_delay_ns\tnet_delay_ns\tlogic_levels"
    foreach entry [lsort -real -index 0 $ranked] {
        lassign $entry slack key group
        set row [dict get $group worst]
        puts $f [join [concat $key [list [dict get $group count] $slack [dict get $row delay] [dict get $row logic] [dict get $row net] [dict get $row levels]]] \t]
    }
    close $f
}

if {[llength $argv] ni {2 3}} {error "expected CHECKPOINT NEW_OUTPUT ?range-slice-cuts?"}
set mode [lindex $argv 2]
if {$mode ni {{} range-slice-cuts}} {error "unknown diagnostic mode: $mode"}
set checkpoint [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file isfile $checkpoint]} {error "checkpoint is absent: $checkpoint"}
if {[file exists $out]} {error "preserve existing reports: $out"}
file mkdir $out
set_param general.maxThreads 4
open_checkpoint $checkpoint
if {$mode eq "range-slice-cuts"} {
    set tag_regs [get_cells -hierarchical -quiet -filter {REF_NAME =~ "FD*" && NAME =~ "*read_tag_reg*"}]
    set owner_regs [get_cells -hierarchical -quiet -filter {REF_NAME =~ "FD*" && NAME =~ "*victim_owner_q_reg*"}]
    set request_regs [get_cells -hierarchical -quiet -filter {REF_NAME =~ "FD*" && NAME =~ "*request_owner_q_reg*"}]
    if {![llength $tag_regs] || ![llength $owner_regs] || ![llength $request_regs]} {
        error "expected mapped tag, victim-owner and request-owner register boundaries"
    }
    set f [open $out/register_boundaries.tsv {WRONLY CREAT EXCL}]
    puts $f "boundary\tcell\tprimitive"
    foreach {name cells} [list tag $tag_regs victim_owner $owner_regs request_owner $request_regs] {
        foreach cell $cells {puts $f "$name\t[get_property NAME $cell]\t[get_property REF_NAME $cell]"}
    }
    close $f
    report_timing -to $owner_regs -max_paths 5 -file $out/owner_input.rpt
    report_timing -from $tag_regs -to $owner_regs -max_paths 5 -file $out/tag_to_owner.rpt
    report_timing -from $owner_regs -max_paths 5 -file $out/owner_output.rpt
    report_timing -from $request_regs -max_paths 5 -file $out/request_owner_output.rpt
    puts "R40_RANGE_SLICE_CUT_DIAGNOSIS_DONE_diagnostic_only"
    exit
}
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
if {[file exists $out/critical_paths.rpt]} {
    if {[catch {r40_rank_timing_report $out/critical_paths.rpt $out/critical_cones_ranked.tsv} message]} {
        puts $status "critical_cones_ranked\tERROR\t$message"
        incr failed
    } else {puts $status "critical_cones_ranked\tREPORT_WRITTEN_sampled_endpoints_only"}
}
puts $status "report_errors\t$failed"
close $status
if {$failed} {error "$failed diagnostic reports failed; preserve partial results"}
puts "R40_CHECKPOINT_DIAGNOSIS_DONE"
exit
