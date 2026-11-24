# R42 lineelig: out-of-context synthesis and place and route of tnca_tensor at the run-2 image geometry (10 ns, SLR0).
#   vivado -mode batch -source screen_r42_tensor.tcl -tclargs RTL_COMMON_DIR TENSOR_V OUT [NAME=VALUE ...]
# The image's elaboration of the unit (tnca_core -> tnca_tensor with the line engine): 4 contexts, 8 barriers,
# line slots 64 (LINE_SLOTS 64 -> port tag 8 bits, MTAG_W 7), HAS_XPOSE 2, 64-bit word walk behind a 512-bit port.
# NAME=VALUE pairs override or add top parameters (LINE_HALF 0/1, NTAG, ...).
if {$argc < 3} {error "usage: screen_r42_tensor.tcl RTL_COMMON_DIR TENSOR_V OUT [NAME=VALUE ...]"}
set common [file normalize [lindex $argv 0]]
set src    [file normalize [lindex $argv 1]]
set out    [file normalize [lindex $argv 2]]
file mkdir $out
set_param general.maxThreads 6
array set P {RCTX_W 2 BAR_W 3 MEM_DW 64 MEM_BYTES 8 TAG_W 4 NTAG 16 HAS_XPOSE 2 MUL_PIPE 1 LINE_ENGINE 1 LSLOT_N 64 LSLOT_W 6 CMD_N 8 CMD_W 3 PTAG_W 7 PORT_DW 512 PORT_BYTES 64}
foreach kv [lrange $argv 3 end] {
    lassign [split $kv =] k v
    set P($k) $v
}
set gen {}
foreach k [lsort [array names P]] {lappend gen -generic "$k=$P($k)"}
puts "R42_PARAMS [array get P]"
set_property include_dirs [list $common] [current_fileset]
read_verilog -sv $src
synth_design -top tnca_tensor -part xcu280-fsvh2892-2L-e -mode out_of_context {*}$gen
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock tensor]
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
puts "R42_TENSOR_SCREEN wns=$wns whs=$whs"
