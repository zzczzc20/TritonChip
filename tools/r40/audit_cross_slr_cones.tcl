# Structural audit of SLR-spanning combinational control cones on a
# synthesized (unplaced) TNCA kernel checkpoint, e.g. the synth.dcp written by
# flow/fpga/synth_r40_multicore.tcl or the v++ kernel run
# ulp_tnca_u50_1_0_synth_1/ulp_tnca_u50_1_0.dcp. Analysis only: no opt/place/
# route, no checkpoint writes. Delays here are pre-placement estimates; the
# evidence is connectivity and logic depth, not slack.
#
# vivado -mode batch -nojournal -log OUT.log -source audit_cross_slr_cones.tcl \
#   -tclargs SYNTH_DCP FRESH_OUTPUT_DIR [PLACE_HOOK_TCL]
#
# With PLACE_HOOK_TCL (flow/fpga/place_r40_multicore.tcl, TNCA_IMAGE_CONFIG in
# the environment) the hook's SLR pblocks are created in memory so that every
# reported path is labelled with the SLR groups of its endpoints. This also
# dry-runs the placement hook against the new hierarchy before any link.
#
# Attempt-3 failure mechanism (routed 100 MHz report): one combinational path
# memory_active_ports_reg -> memory mode response mux -> home link idle ->
# all_idle -> host_allow -> tile0 idle host mux -> core c_ready -> LSU/txf,
# crossing SLR1->SLR2->SLR1->SLR0. A candidate that changes the mechanism has
#   (1) no tile endpoint in the combinational fanout of memory_active_ports,
#   (2) host_allow reaching only registers of the host mux (no core endpoint),
#   (3) home_link_idle reaching no tile endpoint combinationally.
if {[llength $argv] ni {2 3}} {error "expected SYNTH_DCP FRESH_OUTPUT_DIR ?PLACE_HOOK_TCL?"}
lassign $argv checkpoint out hook
if {![file isfile $checkpoint] || [file exists $out]} {error "existing checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
if {[llength [get_clocks -quiet]] == 0} {
    create_clock -name ap_clk -period 10 [get_ports ap_clk]
}
set hook_status none
if {$hook ne ""} {
    if {[catch {source $hook} msg]} {
        set hook_status "failed: $msg"
    } else {
        set hook_status ok
    }
}
proc r40_unit {name} {
    if {[regexp {g_tile\[(\d+)\]\.u_tile/u_core/([^/]+)} $name -> t u]} {return "tile$t/core/$u"}
    if {[regexp {g_tile\[(\d+)\]\.u_tile/([^/]+)} $name -> t u]} {return "tile$t/$u"}
    if {[regexp {g_home_link\[(\d+)\]} $name -> h]} {return "home_link$h"}
    if {[regexp {g_outlet\[(\d+)\]\.([^/]+)} $name -> p u]} {return "outlet$p/$u"}
    if {[regexp {/control/} "/$name"]} {return control}
    if {[regexp {(g_dispatch[^/]*|g_cache_handoff[^/]*|g_code\[\d+\][^/]*|g_local_tmu[^/]*)} $name -> u]} {return $u}
    return other
}
proc r40_slr {cell} {
    set pb [get_pblocks -quiet -of_objects $cell]
    if {[llength $pb] == 0} {return -}
    return [join [lsort [get_property NAME $pb]] ,]
}
proc r40_fanout_census {label pins out} {
    set f [open [file join $out $label.tsv] {WRONLY CREAT EXCL}]
    puts $f "unit\tendpoints"
    if {[llength $pins] == 0} {
        puts $f "MISSING\t0"
        close $f
        return [dict create missing 1]
    }
    set eps [all_fanout -quiet -flat -endpoints_only $pins]
    set counts [dict create]
    foreach p $eps {dict incr counts [r40_unit [get_property NAME $p]]}
    dict for {u c} $counts {puts $f "$u\t$c"}
    close $f
    return $counts
}
set map_q [get_pins -quiet -hierarchical -filter {NAME =~ *control/memory_active_ports_reg*/Q}]
set allow_nets [get_nets -quiet -hierarchical -filter {NAME =~ *host_allow*}]
set allow_pins [get_pins -quiet -of_objects $allow_nets -filter {DIRECTION == OUT}]
set idle_nets [get_nets -quiet -hierarchical -filter {NAME =~ *home_link_idle*}]
set idle_pins [get_pins -quiet -of_objects $idle_nets -filter {DIRECTION == OUT}]
set c1 [r40_fanout_census fanout_memory_active_ports $map_q $out]
set c2 [r40_fanout_census fanout_host_allow $allow_pins $out]
set c3 [r40_fanout_census fanout_home_link_idle $idle_pins $out]
foreach {label query} [list \
        from_memory_active_ports [list -from $map_q] \
        through_host_allow [list -through $allow_nets] \
        through_home_link_idle [list -through $idle_nets]] {
    if {[llength [lindex $query 1]] == 0} {continue}
    set paths [get_timing_paths {*}$query -max_paths 200 -nworst 1 -unique_pins]
    set f [open [file join $out paths_$label.tsv] {WRONLY CREAT EXCL}]
    puts $f "logic_levels\testimated_slack_ns\tstart_unit\tend_unit\tstart_pblock\tend_pblock\tstartpoint\tendpoint"
    foreach p $paths {
        set sp [get_property STARTPOINT_PIN $p]
        set ep [get_property ENDPOINT_PIN $p]
        set sc [get_cells -quiet -of_objects [get_pins -quiet $sp]]
        set ec [get_cells -quiet -of_objects [get_pins -quiet $ep]]
        puts $f [join [list [get_property LOGIC_LEVELS $p] [get_property SLACK $p] \
            [r40_unit $sp] [r40_unit $ep] [r40_slr $sc] [r40_slr $ec] $sp $ep] \t]
    }
    close $f
}
set s [open [file join $out summary.tsv] {WRONLY CREAT EXCL}]
puts $s "checkpoint\t$checkpoint"
puts $s "place_hook\t$hook_status"
puts $s "memory_active_ports_q_pins\t[llength $map_q]"
puts $s "host_allow_nets\t[llength $allow_nets]"
puts $s "home_link_idle_nets\t[llength $idle_nets]"
foreach {label counts} [list memory_active_ports $c1 host_allow $c2 home_link_idle $c3] {
    set tile_eps 0
    dict for {u c} $counts {if {[string match tile* $u]} {incr tile_eps $c}}
    puts $s "${label}_tile_endpoints\t$tile_eps"
}
puts $s "scope\tunplaced netlist connectivity/logic depth; not routed timing"
close $s
puts "R40_CROSS_SLR_CONE_AUDIT_DONE"
exit
