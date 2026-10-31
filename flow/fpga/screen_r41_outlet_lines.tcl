# R41 G2: free routed OOC screen of one image outlet (tnca_line_axi) at the
# run-1 image's parameters, placed in SLR0 where the run-1 image put all three.
#   vivado -mode batch -source screen_r41_outlet_lines.tcl -tclargs RTL_ROOT LINE_LUTRAM OUT
#          [DATA_W OUT_N PIPELINED_ISSUE HBM_STRIPED]     (R41 G4: other images' outlets)
# Image parameters: AXI_DATA_W 256, AXI_OUT_N 8 (ID_W 3), MEMORY_CONTROLS 1
# (DYNAMIC_LIMIT), PIPELINED_ISSUE 0, HBM_STRIPED 0, SRC_W 2, TAG_W 3,
# MEM_BASE 0x20000 (tnca_multicore_axi). Ports are left to OOC partition pins
# with 1 ns of I/O delay each side. Not a substitute for a linked image.
if {$argc != 3 && $argc != 7} {error "usage: screen_r41_outlet_lines.tcl RTL_ROOT LINE_LUTRAM OUT \[DATA_W OUT_N PIPELINED_ISSUE HBM_STRIPED\]"}
set dw 256; set outn 8; set pipe 0; set striped 0; set tagw 3
# the 7-argument form is the R41 images' home tag (HOME_MSHR_N 32: 5 bits)
if {$argc == 7} {lassign [lrange $argv 3 6] dw outn pipe striped; set tagw 5}
set idw [expr {$outn > 1 ? int(ceil(log($outn)/log(2))) : 1}]
set root [file normalize [lindex $argv 0]]
set lut [lindex $argv 1]
set out [file normalize [lindex $argv 2]]
if {[file exists $out]} {error "output must be fresh: $out"}
file mkdir $out
set_param general.maxThreads 4
read_verilog -sv $root/rtl/fpga/tnca_line_axi.v
synth_design -top tnca_line_axi -part xcu280-fsvh2892-2L-e -mode out_of_context \
    -generic DATA_W=$dw -generic OUT_N=$outn -generic ID_W=$idw -generic DYNAMIC_LIMIT=1 \
    -generic PIPELINED_ISSUE=$pipe -generic HBM_STRIPED=$striped -generic SRC_W=2 -generic TAG_W=$tagw \
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
puts "R41_OUTLET_LINES_SCREEN line_lutram=$lut data_w=$dw out_n=$outn pipelined=$pipe striped=$striped period_ns=10 wns_ns=$wns"
