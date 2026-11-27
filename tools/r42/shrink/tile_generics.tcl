# tile_generics.tcl -- elaborate the kernel top at an image's parameters and print the generics of tile 0 (R42-shrink).
#   vivado -mode batch -nojournal -nolog -source tile_generics.tcl -tclargs <root> <image.tcl> <out.txt>
set root [file normalize [lindex $argv 0]]
source [file normalize [lindex $argv 1]]
set out [lindex $argv 2]
set_param general.maxThreads 4
set sources [list]
foreach dir {common core gen nuca noc tile fpga} { foreach s [lsort [glob $root/rtl/$dir/*.v]] {lappend sources $s} }
foreach s [lsort [glob $root/rtl/*.v]] {lappend sources $s}
lappend sources $root/rtl/board/tnca_multicore_hbm_xo.v
read_verilog -sv $sources
set_property include_dirs [list $root/rtl/common $root/rtl/noc] [current_fileset]
set generics [list]
dict for {name value} $TNCA_IMAGE_PARAMS {lappend generics "$name=$value"}
synth_design -rtl -top tnca_multicore_hbm_xo -part xcu280-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy none -generic $generics
set fh [open $out w]
foreach c [get_cells -hier -filter {REF_NAME == tnca_tile}] {
    puts $fh "CELL [get_property NAME $c]"
    foreach pr [list_property $c] { catch { puts $fh "$pr = [get_property $pr $c]" } }
}
close $fh
puts DONE
