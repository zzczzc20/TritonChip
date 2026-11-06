# Free routed screen: shared contiguous-vector planner and finite landing slots.
if {$argc != 2} {error "usage: screen_r41_lsu_line.tcl RTL_ROOT OUT"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
set_param general.maxThreads 4
set_property include_dirs [list [file join $rtl common]] [current_fileset]
read_verilog -sv [file join $rtl core/tnca_lsu_line.v]
synth_design -top tnca_lsu_line -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock line_lsu]
resize_pblock $pb -add SLR1
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
report_drc -file $out/routed_drc.rpt
write_checkpoint -force $out/routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "R41_LINE_LSU_SCREEN wns=$wns whs=$whs"
if {$wns < 0 || $whs < 0} {error "line LSU screen failed timing"}
