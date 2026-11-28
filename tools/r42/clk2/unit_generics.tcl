# unit_generics.tcl -- elaborate tnca_tile (RTL elaboration only) at the given generics and print the generics of u_mtx u_vfpu u_bar u_txf u_valu u_mul
#   vivado -mode batch -nojournal -nolog -source unit_generics.tcl -tclargs <root> <out.txt> "<generics>"
set root [file normalize [lindex $argv 0]]
set out [lindex $argv 1]
set gen [list]
foreach g [split [lindex $argv 2] " "] { if {$g ne ""} { lappend gen -generic $g } }
set_param general.maxThreads 4
set sources [list]
foreach dir {common core gen nuca noc tile} { foreach s [lsort [glob $root/rtl/$dir/*.v]] {lappend sources $s} }
read_verilog -sv $sources
synth_design -rtl -top tnca_tile -part xcu280-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy none -include_dirs [list $root/rtl/common $root/rtl/noc] {*}$gen
set fh [open $out w]
foreach u {u_mtx u_vfpu u_bar u_txf u_valu u_mul} {
    set c [get_cells -hier -filter "NAME =~ */$u || NAME == $u"]
    foreach cc $c { puts $fh "UNIT $u cell=[get_property NAME $cc] ref=[get_property REF_NAME $cc]"
    foreach pr [list_property $cc] { catch { puts $fh "$pr = [get_property $pr $cc]" } } }
}
close $fh
puts DONE
