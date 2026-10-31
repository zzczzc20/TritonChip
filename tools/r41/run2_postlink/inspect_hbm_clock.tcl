# Read-only inspection of the routed checkpoint's HBM clock definition.
# This does not change a constraint, write a checkpoint, or establish closure.
lassign $argv checkpoint out
if {![file isfile $checkpoint] || [file exists $out]} {error "checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
set clocks [get_clocks -quiet hbm_aclk]
if {[llength $clocks] != 1} {error "expected exactly one HBM clock"}
set f [open $out/hbm_clock_properties.txt {WRONLY CREAT EXCL}]
foreach property [lsort [list_property $clocks]] {
    if {![catch {get_property $property $clocks} value]} {
        puts $f "$property\t$value"
    }
}
close $f
report_timing -from $clocks -to $clocks -max_paths 30 -path_type full_clock_expanded -file $out/hbm_paths.rpt
report_clock_interaction -file $out/clock_interaction.rpt
puts "HBM clock inspection complete"
