# path_probe.tcl DCP OUTFILE  -- open a routed checkpoint and report, for a list of (from-pattern, to-pattern) pairs, the worst path with fanouts, and the highest-fanout nets.
#   vivado -mode batch -source path_probe.tcl -tclargs routed.dcp out.rpt PAIRFILE
# PAIRFILE: one pair per line "name from_cells to_cells" (cells as get_cells patterns without spaces).
set dcp [lindex $::argv 0]; set out [lindex $::argv 1]; set pairs [lindex $::argv 2]
open_checkpoint $dcp
set fh [open $out w]
puts $fh "=== high fanout nets"
close $fh
report_high_fanout_nets -max_nets 40 -timing -file $out.hfn
set f [open $pairs r]
while {[gets $f line] >= 0} {
    if {[string trim $line] eq "" || [string index $line 0] eq "#"} continue
    lassign $line name from to
    set fc [get_cells -quiet -hier -filter "NAME =~ $from"]
    set tc [get_cells -quiet -hier -filter "NAME =~ $to"]
    if {[llength $fc] == 0 || [llength $tc] == 0} { set fh [open $out a]; puts $fh "=== $name: no cells ($from / $to)"; close $fh; continue }
    report_timing -from $fc -to $tc -max_paths 1 -nworst 1 -file $out.$name
}
puts DONE
