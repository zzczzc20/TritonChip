# R42 lineelig: out-of-context synthesis and place and route of one unit at 10 ns in an SLR0 pblock (the screen_r41_* recipe).
#   vivado -mode batch -source screen_r42_unit.tcl -tclargs TOP INCLUDE_DIR FILELIST OUT [NAME=VALUE ...]
# FILELIST: one source path per line (absolute); NAME=VALUE: top-level generics.
if {$argc < 4} {error "usage: screen_r42_unit.tcl TOP INCLUDE_DIR FILELIST OUT [NAME=VALUE ...]"}
set top  [lindex $argv 0]
set inc  [file normalize [lindex $argv 1]]
set fl   [lindex $argv 2]
set out  [file normalize [lindex $argv 3]]
file mkdir $out
set_param general.maxThreads 6
set gen {}
foreach kv [lrange $argv 4 end] {
    lassign [split $kv =] k v
    lappend gen -generic "$k=$v"
}
puts "R42_UNIT $top $gen"
set_property include_dirs [list $inc] [current_fileset]
set fh [open $fl r]
foreach f [split [string trim [read $fh]] "\n"] {read_verilog -sv $f}
close $fh
synth_design -top $top -part xcu280-fsvh2892-2L-e -mode out_of_context {*}$gen
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock unit]
resize_pblock $pb -add SLR0
add_cells_to_pblock $pb -top
set_property CONTAIN_ROUTING 1 $pb
report_utilization -file $out/synth_util.rpt
report_timing_summary -file $out/synth_timing.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/routed_util.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file $out/routed_timing.rpt
report_timing -max_paths 10 -nworst 1 -file $out/routed_paths.rpt
write_checkpoint -force $out/routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "R42_UNIT_SCREEN $top wns=$wns whs=$whs"
