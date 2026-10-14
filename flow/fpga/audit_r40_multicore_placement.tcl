# Diagnostic validation on an actual synthesized checkpoint; not routed proof.
# vivado -mode batch -source ... -tclargs DCP IMAGE_TCL REPORT
set checkpoint [file normalize [lindex $argv 0]]
source [file normalize [lindex $argv 1]]
set report [file normalize [lindex $argv 2]]
open_checkpoint $checkpoint
source [file join [file dirname [info script]] place_r40_multicore.tcl]
set expected_maintenance [dict get $TNCA_IMAGE_PARAMS CACHE_MAINTENANCE]
if {$r40_maintenance != $expected_maintenance} {
    error "R40 placement ownership hierarchy disagrees with image capability"
}
set stream [open $report w]
puts $stream "Diagnostic only: no placement or route performed"
puts $stream "Checkpoint: $checkpoint"
puts $stream "CACHE_MAINTENANCE=$r40_maintenance"
for {set slr 0} {$slr < 3} {incr slr} {
    set block [get_pblocks -quiet r40_slr$slr]
    if {[llength $block] != 1} {error "missing SLR$slr pblock"}
    set cells [get_cells -quiet -of_objects $block]
    if {[llength $cells] == 0} {error "empty SLR$slr assignment"}
    puts $stream "SLR$slr assigned cells=[llength $cells]"
    foreach cell $cells {puts $stream "  $cell"}
}
close $stream
puts "R40_MULTICORE_PLACEMENT_CONTRACT_PASS"
exit
