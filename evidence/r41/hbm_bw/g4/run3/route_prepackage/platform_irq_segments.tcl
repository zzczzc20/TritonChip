set_param general.maxThreads 2
set base $::env(HOME)/r41run3
open_checkpoint $base/build_8d237223_retry1/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
set reg {level0_i/blp/blp_i/PLP/ss_fic/inst/user_fanout_firewall_and_regslices/axi_firewall_ctrl_user/inst/gen_mi.mi_r_error_i_reg}
set report [open $base/cdc_deep_audit_20261030/segments.txt w]
foreach target [list $reg/D $reg/Q] {
 set pin [get_pins -quiet $target]
 puts $report "TARGET $target PINCOUNT [llength $pin]"
 if {[llength $pin] != 1} {error "target pin missing"}
 set ns [get_nets -quiet -segments -of_objects $pin]
 puts $report "SEGMENTS [llength $ns] $ns"
 foreach n $ns {
  foreach p [get_pins -quiet -leaf -of_objects $n] {
   set c [get_cells -quiet -of_objects $p]
   puts $report "PIN $p DIR [get_property DIRECTION $p] CELL $c REF [get_property REF_NAME $c]"
   if {[get_property DIRECTION $p] eq "OUT" && [string match LUT* [get_property REF_NAME $c]]} {
    puts $report "DRIVER_INIT [get_property INIT $c]"
    foreach input [get_pins -quiet -of_objects $c -filter {DIRECTION == IN}] {
     puts $report "INPUT $input SEGMENTS [get_nets -quiet -segments -of_objects $input]"
    }
   }
  }
 }
}
close $report
exit
