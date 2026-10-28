# Free wide-home screen; source ~/x86_link_env.sh before invoking on x86.
if {$argc<2 || $argc>5} {error "usage: screen_r41_wide_home.tcl RTL_ROOT OUT ?HIT_CHAIN? ?FAST_FILL_READ? ?BACKGROUND_READ_REQUESTS?"}
set chain 0
if {$argc>=3} {set chain [lindex $argv 2]}
set fast_fill 0
if {$argc>=4} {set fast_fill [lindex $argv 3]}
set background 0
if {$argc==5} {set background [lindex $argv 4]}
if {$background ni {0 1}} {error "BACKGROUND_READ_REQUESTS must be 0 or 1"}
if {$fast_fill ni {0 1}} {error "FAST_FILL_READ must be 0 or 1"}
if {$chain ni {0 1}} {error "HIT_CHAIN must be 0 or 1"}
set rtl [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
set_param general.maxThreads 4
set_property include_dirs [list [file join $rtl common]] [current_fileset]
read_verilog -sv [file join $rtl nuca/tnca_nuca_slice.v]
set generics {WORD_BITS=512 WORD_BYTES=64 FULL_LINE_TAG=1 SETS=32 SET_W=5 WAYS=8 WAY_W=3 TAG_W=5 CACHE_DATA_PIPELINE=1 REGISTER_VICTIM_OWNER=1 REGISTER_VICTIM_SELECT=1 REGISTER_VICTIM_RANK=1 DCO_REGISTERED_RANGES=1}
lappend generics HIT_CHAIN=$chain FAST_FILL_READ=$fast_fill BACKGROUND_READ_REQUESTS=$background
synth_design -top tnca_nuca_slice -part xcu280-fsvh2892-2L-e -mode out_of_context -generic $generics
create_clock -name kernel -period 10 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks kernel]
set_input_delay 1.000 -clock kernel [get_ports -filter {DIRECTION==IN && NAME!=clk && NAME!=rst_n}]
set_output_delay 1.000 -clock kernel [all_outputs]
set_false_path -from [get_ports rst_n]
set pb [create_pblock wide_home]
resize_pblock $pb -add SLR1
add_cells_to_pblock $pb -top
set_property CONTAIN_ROUTING 1 $pb
report_utilization -hierarchical -file $out/synth_util.rpt
report_timing_summary -file $out/synth_timing.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -hierarchical -file $out/routed_util.rpt
report_timing_summary -report_unconstrained -check_timing_verbose -file $out/routed_timing.rpt
report_drc -file $out/routed_drc.rpt
write_checkpoint -force $out/routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
puts "R41_WIDE_HOME_SCREEN wns=$wns whs=$whs"
if {$wns<0 || $whs<0} {error "wide home screen failed timing"}
