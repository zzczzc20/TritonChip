# Free routed OOC screen of a candidate stream-capable line outlet.
# vivado -mode batch -source screen_r41_hbm_outlet.tcl -tclargs RTL OUT SLOTS
if {$argc != 3} {error "usage: screen_r41_hbm_outlet.tcl RTL OUT SLOTS"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set slots [lindex $argv 2]
if {$slots ni {32 64}} {error "screen slots must be 32 or 64"}
file mkdir $out
set_param general.maxThreads 4
read_verilog $rtl
synth_design -top tnca_line_axi -part xcu280-fsvh2892-2L-e -mode out_of_context \
    -generic [list DATA_W=512 OUT_N=$slots PIPELINED_ISSUE=1 WRITE_RESPONSES=1]
create_clock -name kernel -period 10.000 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
report_utilization -file $out/synth_util.rpt
report_timing_summary -file $out/synth_timing.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/routed_util.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file $out/routed_timing.rpt
report_timing -max_paths 20 -file $out/routed_paths.rpt
report_drc -file $out/routed_drc.rpt
write_checkpoint -force $out/routed.dcp
set worst [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK $worst]
puts "HBM_OUTLET_SCREEN slots=$slots period_ns=10 wns_ns=$wns"
if {$wns < 0} {error "HBM outlet screen failed 100 MHz setup timing"}
