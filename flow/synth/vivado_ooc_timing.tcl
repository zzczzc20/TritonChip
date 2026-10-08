#==============================================================================
# vivado_ooc_timing.tcl -- post-synthesis timing of one module, out of context
#
#   [OOC_GENERICS="NAME=V ..."] vivado -mode batch -nojournal -nolog \
#          -source flow/synth/vivado_ooc_timing.tcl \
#          -tclargs <top> <period_ns> <part> <out_dir> <rtl_dir> <file> [<file> ...]
#
# Synthesis only, no placement or routing: minutes for a unit like the matrix
# engine, against hours for a v++ link. The delays are Vivado's estimates
# before routing, so they are optimistic on wires -- on the U280 the worst
# clk_dev path was 73 % route -- and are for ranking paths and checking that a
# change removed the levels it was meant to. The frequency a link closes at
# is still the link's to say.
#
# Every input and output is left unconstrained, so only register-to-register
# paths are reported: a module's ports are registers or wires of its parent,
# and timing them here would be guessing at the parent's logic.
#==============================================================================
set top    [lindex $::argv 0]
set period [lindex $::argv 1]
set part   [lindex $::argv 2]
set out    [lindex $::argv 3]
set rtl    [lindex $::argv 4]
set files  [lrange $::argv 5 end]

file mkdir $out
foreach f $files {
    read_verilog [file join $rtl $f]
}
set_param general.maxThreads 8
# Parameters of the top, as NAME=VALUE pairs in OOC_GENERICS (e.g.
# "FP_LAT=3"): the same unit at another configuration without a wrapper.
set gen [list]
if {[info exists ::env(OOC_GENERICS)]} {
    foreach g [split $::env(OOC_GENERICS) " "] {
        if {$g ne ""} { lappend gen -generic $g }
    }
}
synth_design -top $top -part $part -mode out_of_context \
    -include_dirs [list [file join $rtl common] [file join $rtl noc]] \
    -flatten_hierarchy rebuilt {*}$gen
create_clock -name clk -period $period [get_ports clk]

report_utilization -file [file join $out ${top}_util.rpt]
report_timing_summary -no_header -no_detailed_paths -file [file join $out ${top}_summary.rpt]
# One worst path per endpoint, unique start/end pins: the list of distinct
# critical paths, not 50 bits of the same bus.
report_timing -from [all_registers] -to [all_registers] -max_paths 60 -nworst 1 \
    -unique_pins -sort_by group -file [file join $out ${top}_paths.rpt]
puts "DONE $top"
