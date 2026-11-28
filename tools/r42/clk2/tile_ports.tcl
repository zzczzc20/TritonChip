#==============================================================================
# tile_ports.tcl -- R42 clk2: time the paths that tile_screen.tcl leaves unconstrained (register -> tile output port, tile input port -> register).
# The tile screen has no port delays, so a path that continues into another unit outside the tile (here: transfer engine -> address -> network
# interface -> mesh input FIFO) is cut at the tile boundary and not timed. This opens a routed tile checkpoint, gives every output an output
# delay of 0 and every input an input delay of 0, and writes the paths to / from the ports: the arrival time at the port is what the tile
# leaves for the shell.
#   vivado -mode batch -nojournal -source tile_ports.tcl -tclargs ROUTED_DCP OUT_DIR THREADS
#==============================================================================
set dcp [file normalize [lindex $::argv 0]]
set out [file normalize [lindex $::argv 1]]
set thr [lindex $::argv 2]
file mkdir $out
set_param general.maxThreads $thr
open_checkpoint $dcp
set outs [all_outputs]
set ins  [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0 $outs
set_input_delay  -clock clk 0 $ins
proc dump {file paths} {
    set fh [open $file w]
    puts $fh "slack,start,end,levels,delay,logic,net"
    foreach p $paths {
        puts $fh [join [list [get_property SLACK $p] [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] \
            [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p] [get_property DATAPATH_LOGIC_DELAY $p] \
            [get_property DATAPATH_NET_DELAY $p]] ","]
    }
    close $fh
}
dump $out/to_ports.csv   [get_timing_paths -to   $outs -max_paths 50000 -nworst 1 -unique_pins -setup]
dump $out/from_ports.csv [get_timing_paths -from $ins  -max_paths 50000 -nworst 1 -unique_pins -setup]
report_timing -to $outs -max_paths 40 -nworst 1 -unique_pins -file $out/to_ports_paths.rpt
report_timing -from $ins -max_paths 20 -nworst 1 -unique_pins -file $out/from_ports_paths.rpt
puts DONE
