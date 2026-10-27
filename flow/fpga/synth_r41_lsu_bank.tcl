# Free unit synthesis: finite LSU bank area before wide-path integration.
if {$argc != 4} {error "usage: synth_r41_lsu_bank.tcl RTL_ROOT OUT UNITS WIDTH"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set units [lindex $argv 2]
set width [lindex $argv 3]
if {$units ni {4 8} || $width ni {64 512}} {error "unsupported LSU screen"}
set bytes [expr {$width / 8}]
set shift [expr {$width == 512 ? 6 : 3}]
file mkdir $out
set_param general.maxThreads 4
set_property include_dirs [list [file join $rtl common]] [current_fileset]
read_verilog -sv [file join $rtl core/tnca_lsu.v]
read_verilog -sv [file join $rtl core/tnca_lsu_bank.v]
synth_design -top tnca_lsu_bank -part xcu280-fsvh2892-2L-e -mode out_of_context -generic [list UNITS=$units NUM_BEATS=[expr {16/$units}] BEAT_W=4 MEM_DW=$width MEM_BYTES=$bytes MEM_SHIFT=$shift RCTX_W=2]
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION == IN && NAME != clk && NAME != rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
report_utilization -hierarchical -file $out/synth_util.rpt
report_timing_summary -file $out/synth_timing.rpt
report_drc -file $out/synth_drc.rpt
write_checkpoint -force $out/synth.dcp
puts "R41_LSU_BANK_SYNTH units=$units width=$width total_tags=16"
