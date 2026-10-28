# R41 G2: count the performance-counter registers that survive kernel synthesis.
#   vivado -mode batch -source count_perf_cells.tcl -tclargs SYNTH_DCP
open_checkpoint [lindex $argv 0]
foreach pat {*u_core/perf_* *u_slice/perf_* *adapter/read_lines* *adapter/write_lines* *adapter/read_beats* *adapter/write_beats* *perf_*} {
  set c [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && PRIMITIVE_GROUP == REGISTER && NAME =~ $pat"]
  puts "R41_PERF_CELLS pattern=$pat registers=[llength $c]"
}
