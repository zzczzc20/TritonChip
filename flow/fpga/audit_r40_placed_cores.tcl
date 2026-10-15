# Read actual locations from a placed/routed checkpoint, not pblock intentions.
# Usage: vivado -mode batch -source ... -tclargs ROUTED_DCP NEW_REPORT
set checkpoint [file normalize [lindex $argv 0]]
set report [file normalize [lindex $argv 1]]
if {[llength $argv] != 2} {error "ROUTED_DCP and NEW_REPORT are required"}
if {[file exists $report]} {error "preserve the prior placement report: $report"}
open_checkpoint $checkpoint

set cells [get_cells -hierarchical -filter {IS_PRIMITIVE}]
set names [get_property NAME $cells]
set groups [dict create 0 [list] 1 [list] 2 [list]]
foreach cell $cells name $names {
    if {[regexp {(^|/)u_system/g_tile\[([0-2])\]\.u_tile/} $name whole prefix core]} {
        dict lappend groups $core $cell
    }
}
set stream [open $report {WRONLY CREAT EXCL}]
puts $stream "schema tnca-r40-actual-core-placement/1"
puts $stream "checkpoint $checkpoint"
set failures [list]
for {set core 0} {$core < 3} {incr core} {
    set members [dict get $groups $core]
    set placed [list]
    set unplaced 0
    set constants 0
    if {[llength $members]} {
        set locations [get_property IS_PLACED $members]
        set references [get_property REF_NAME $members]
        foreach cell $members location $locations reference $references {
            # Netlist constants have no physical site; report their exclusion.
            if {$reference in {GND VCC}} {
                incr constants
            } elseif {$location} {
                lappend placed $cell
            } else {
                incr unplaced
            }
        }
    }
    set slrs [list]
    if {[llength $placed]} {
        # Vivado 2023.2 get_slrs explicitly supports cell objects. This queries
        # their actual mapped sites, independently of USER_SLR_ASSIGNMENT.
        set regions [get_slrs -of_objects $placed]
        if {[llength $regions]} {set slrs [lsort -unique [get_property NAME $regions]]}
    }
    set expected [format "SLR%d" $core]
    set ok [expr {[llength $placed] > 0 && $unplaced == 0 && $slrs eq [list $expected]}]
    puts $stream "core $core primitive_count [llength $members] placed [llength $placed] unplaced $unplaced excluded_constants $constants actual_slrs {$slrs} expected $expected pass $ok"
    if {!$ok} {lappend failures $core}
}
puts $stream "failing_cores {$failures}"
close $stream
if {[llength $failures]} {error "actual core placement failed for cores $failures; see $report"}
puts "R40_ACTUAL_CORE_PLACEMENT_PASS"
exit
