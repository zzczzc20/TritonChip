# cell_census.tcl -- primitive census of a synthesized tile checkpoint, grouped by array name (R42-shrink M0).
#   vivado -mode batch -nojournal -nolog -source cell_census.tcl -tclargs <synth.dcp> <out.csv> [top]
# One row per (normalised hierarchical name, primitive): path with index brackets collapsed, bit/word suffixes of RAM and
# register arrays stripped; columns: name,ref,count. Read-only; writes no checkpoint.
set dcp [lindex $::argv 0]
set out [lindex $::argv 1]
set_param general.maxThreads 4
open_checkpoint $dcp
set fh [open $out w]
puts $fh "name,ref,count"
array set cnt {}
foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
    set ref [get_property REF_NAME $c]
    set n [get_property NAME $c]
    regsub -all {\[[0-9]+\]} $n {[]} n
    regsub {_[0-9]+_[0-9]+_[0-9]+(_[A-Za-z0-9]+)?$} $n {_RAMSLICE} n
    regsub {_reg(_[0-9]+)?(_[0-9]+)?$} $n {_reg} n
    regsub {\.[A-Za-z0-9_]+$} $n {} nn_dummy
    set key "$n,$ref"
    if {[info exists cnt($key)]} { incr cnt($key) } else { set cnt($key) 1 }
}
foreach k [lsort [array names cnt]] { puts $fh "$k,$cnt($k)" }
close $fh
puts DONE
