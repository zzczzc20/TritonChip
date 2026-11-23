# netlist.tcl -- post-synthesis functional netlist of tnca_kfuse_lane for a netlist simulation (Vivado unisim models).
#   OOC_GENERICS="G=4 W=0 ..." vivado -mode batch -source netlist.tcl -tclargs <out_dir> <files...>
set out   [lindex $::argv 0]
set files [lrange $::argv 1 end]
file mkdir $out
foreach f $files { read_verilog -sv $f }
set gen [list]
if {[info exists ::env(OOC_GENERICS)]} {
    foreach g [split $::env(OOC_GENERICS) " "] { if {$g ne ""} { lappend gen -generic $g } }
}
synth_design -top tnca_kfuse_lane -part xcu280-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy rebuilt {*}$gen
write_verilog -mode funcsim -force [file join $out netlist.v]
report_utilization -file [file join $out netlist_util.rpt]
puts "DONE"
