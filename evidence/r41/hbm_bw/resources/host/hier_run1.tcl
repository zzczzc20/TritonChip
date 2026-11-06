# Read-only resource analysis of the run-1 routed DCP (R41 hbm_bw G2).
# One open_checkpoint, 2 threads. Never writes a checkpoint, never touches a board.
#   vivado -mode batch -nojournal -log OUT.log -source hier_run1.tcl -tclargs ROUTED_DCP FRESH_OUT
# Outputs:
#   util_hier_d20.rpt      report_utilization -hierarchical, depth 20, whole design
#   util_slr.rpt           report_utilization -slr
#   leaf_cells.tsv         one row per placed primitive: SLR, REF_NAME, PRIMITIVE_TYPE, cell name
#                          (all cells of the design, platform included)
lassign $argv checkpoint out
if {![file isfile $checkpoint] || [file exists $out]} {error "existing checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
proc stamp {m} {puts "R41RES [clock format [clock seconds] -gmt 1 -format %H:%M:%S] $m"}
stamp opened
report_utilization -hierarchical -hierarchical_depth 20 -file $out/util_hier_d20.rpt
report_utilization -slr -file $out/util_slr.rpt
stamp util
set fh [open $out/leaf_cells.tsv w]
puts $fh "slr\tref_name\tprimitive_type\tcell"
foreach slr [get_slrs] {
    set n 0
    set cells [get_cells -quiet -of_objects [get_sites -of_objects $slr]]
    foreach c $cells pt [get_property PRIMITIVE_TYPE $cells] rn [get_property REF_NAME $cells] {
        puts $fh "$slr\t$rn\t$pt\t$c"
        incr n
    }
    stamp "$slr cells $n"
}
close $fh
stamp done
