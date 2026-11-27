# R42 lineelig: RTL elaboration (no synthesis) of the board kernel top at an image's parameters, in Vivado: checks that the
# image parameters (TENSOR_LINE_HALF, TENSOR_TAG_W, MTX_ABUF, ...) reach tnca_tensor / tnca_matrix through every level.
#   vivado -mode batch -source elab_r42_image.tcl -tclargs RTL_ROOT IMAGE_TCL OUT
if {$argc != 3} {error "usage: elab_r42_image.tcl RTL_ROOT IMAGE_TCL OUT"}
set rtl [file normalize [lindex $argv 0]]
source [lindex $argv 1]
set out [file normalize [lindex $argv 2]]
file mkdir $out
set_param general.maxThreads 6
set_property include_dirs [list [file join $rtl common] [file join $rtl noc]] [current_fileset]
foreach d {common core gen nuca noc tile fpga board} {
    foreach f [lsort [glob -nocomplain [file join $rtl $d *.v]]] {read_verilog -sv $f}
}
foreach f [lsort [glob -nocomplain [file join $rtl *.v]]] {read_verilog -sv $f}
set gen {}
dict for {k v} $TNCA_IMAGE_PARAMS {lappend gen -generic "$k=$v"}
puts "R42_ELAB generics: $gen"
synth_design -rtl -top tnca_multicore_hbm_xo -part xcu280-fsvh2892-2L-e {*}$gen
foreach c [get_cells -hierarchical -filter {REF_NAME == tnca_tensor}] {
    puts "R42_ELAB_TENSOR $c LINE_ENGINE=[get_property LINE_ENGINE $c] LINE_HALF=[get_property LINE_HALF $c] NTAG=[get_property NTAG $c] TAG_W=[get_property TAG_W $c] LSLOT_N=[get_property LSLOT_N $c] PTAG_W=[get_property PTAG_W $c]"
}
foreach c [get_cells -hierarchical -filter {REF_NAME == tnca_matrix}] {
    puts "R42_ELAB_MATRIX $c ABUF=[get_property ABUF $c]"
}
puts "R42_ELAB_DONE"
