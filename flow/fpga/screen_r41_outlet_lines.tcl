# R41 G2: free routed OOC screen of one image outlet (tnca_line_axi) at the
# run-1 image's parameters, placed in SLR0 where the run-1 image put all three.
#   vivado -mode batch -source screen_r41_outlet_lines.tcl -tclargs RTL_ROOT LINE_LUTRAM OUT
# Image parameters: AXI_DATA_W 256, AXI_OUT_N 8 (ID_W 3), MEMORY_CONTROLS 1
# (DYNAMIC_LIMIT), PIPELINED_ISSUE 0, HBM_STRIPED 0, SRC_W 2, TAG_W 3,
# MEM_BASE 0x20000 (tnca_multicore_axi). Ports are left to OOC partition pins
# with 1 ns of I/O delay each side. Not a substitute for a linked image.
if {$argc != 3} {error "usage: screen_r41_outlet_lines.tcl RTL_ROOT LINE_LUTRAM OUT"}
set root [file normalize [lindex $argv 0]]
set lut [lindex $argv 1]
set out [file normalize [lindex $argv 2]]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
read_verilog -sv $root/rtl/fpga/tnca_line_axi.v
synth_design -top tnca_line_axi -part xcu280-fsvh2892-2L-e -mode out_of_context \
    -generic DATA_W=256 -generic OUT_N=8 -generic ID_W=3 -generic DYNAMIC_LIMIT=1 \
    -generic PIPELINED_ISSUE=0 -generic HBM_STRIPED=0 -generic SRC_W=2 -generic TAG_W=3 \
    -generic LINE_LUTRAM=$lut
create_clock -name kernel -period 10.000 [get_ports clk]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay 1.000 -clock kernel [all_outputs]
set pb [create_pblock outlet_pb]
resize_pblock $pb -add SLR0
add_cells_to_pblock $pb -top
report_utilization -file $out/synth_util.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/routed_util.rpt
report_timing_summary -max_paths 20 -file $out/routed_timing.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "R41_OUTLET_LINES_SCREEN line_lutram=$lut period_ns=10 wns_ns=$wns"
