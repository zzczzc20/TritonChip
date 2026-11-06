# R41 G2: out-of-context screen of one unit at 10 ns in SLR0.
#   vivado -mode batch -source unit_ooc.tcl -tclargs TOP OUT_DIR ROUTE(0|1) FILE.v [FILE.v ...]
# TOP is a register-to-register module (regwrap.py, or a hand wrapper).
set top [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
set route [lindex $argv 2]
set files [lrange $argv 3 end]
file mkdir $out
set_param general.maxThreads 4
foreach f $files { read_verilog -sv $f }
synth_design -top $top -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
report_utilization -file $out/synth_util.rpt
report_utilization -hierarchical -file $out/synth_util_hier.rpt
report_timing_summary -file $out/synth_timing.rpt
if {$route} {
  set pb [create_pblock unit_pb]
  resize_pblock $pb -add SLR0
  add_cells_to_pblock $pb -top
  opt_design
  place_design
  phys_opt_design
  route_design
  report_utilization -file $out/routed_util.rpt
  report_timing_summary -max_paths 10 -file $out/routed_timing.rpt
  set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
  puts "R41_UNIT_OOC top=$top wns_ns=$wns"
}
puts "R41_UNIT_OOC_DONE $top"
