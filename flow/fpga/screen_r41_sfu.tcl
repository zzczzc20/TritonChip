# Free parallel-SFU screen; no image link or card access.
if {$argc != 3} {error "usage: screen_r41_sfu.tcl RTL_ROOT OUT LANES"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set lanes [lindex $argv 2]
if {$lanes ni {1 2 4 8}} {error "unsupported SFU replication"}
file mkdir $out
set_param general.maxThreads 4
set_property include_dirs [list [file join $rtl common]] [current_fileset]
foreach f {core/tnca_sfu.v gen/tnca_sfu_exp2_xls.v gen/tnca_sfu_rcp_xls.v gen/tnca_sfu_rsqrt_xls.v gen/tnca_sfu_log2_xls.v} {
 read_verilog -sv [file join $rtl $f]
}
synth_design -top tnca_sfu -part xcu280-fsvh2892-2L-e -mode out_of_context -generic LANES=$lanes
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock sfu]
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
puts "R41_SFU_SCREEN lanes=$lanes wns=$wns whs=$whs"
if {$wns < 0 || $whs < 0} {error "SFU screen failed timing"}
