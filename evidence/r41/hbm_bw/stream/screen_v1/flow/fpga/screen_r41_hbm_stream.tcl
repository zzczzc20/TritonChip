# Free routed screen of one ordered burst stream lane in SLR1.
# vivado -mode batch -source screen_r41_hbm_stream.tcl -tclargs RTL_ROOT OUT PERIOD
if {$argc != 3} {error "usage: screen_r41_hbm_stream.tcl RTL_ROOT OUT PERIOD"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set period [lindex $argv 2]
if {$period ni {10 5}} {error "screen period must be 10 or 5 ns"}
file mkdir $out
set_param general.maxThreads 4
foreach f {common/tnca_fifo.v fpga/tnca_stream_fifo.v fpga/tnca_stream_axi.v} {
 read_verilog -sv [file join $rtl $f]
}
synth_design -top tnca_stream_axi -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -name kernel -period $period [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock stream_lane]
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
report_timing -max_paths 30 -file $out/routed_paths.rpt
report_drc -file $out/routed_drc.rpt
write_checkpoint -force $out/routed.dcp
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
set wns [get_property SLACK $setup]
set whs [get_property SLACK $hold]
puts "HBM_STREAM_SCREEN period_ns=$period wns_ns=$wns whs_ns=$whs"
if {$wns < 0 || $whs < 0} {error "HBM stream lane screen failed timing"}
