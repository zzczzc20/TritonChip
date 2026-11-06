# Free OOC synthesis checkpoint for the 32-lane stream bank.
if {[llength $argv] != 2} {error "expected RTL_ROOT OUT_DIR"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
set_param general.maxThreads 4
foreach f {common/tnca_fifo.v fpga/tnca_stream_fifo.v fpga/tnca_stream_axi.v fpga/tnca_stream_bank.v} {
    read_verilog -sv [file join $rtl $f]
}
synth_design -top tnca_stream_bank -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -name kernel -period 10 [get_ports clk]
write_checkpoint -force $out/synth.dcp
report_utilization -file $out/utilization.rpt
