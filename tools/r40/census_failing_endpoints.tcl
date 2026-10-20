# Read-only census of every failing setup endpoint of the TNCA kernel clock in
# an existing routed checkpoint (R40 final-implementation planning).
#
# Never runs synth/opt/place/route/bitstream and never writes a checkpoint.
# Run only when the FPGA host is not measuring board performance: the measured
# cost on the attempt-3 routed DCP is ~11-12 min open_checkpoint (~13.6 GB),
# ~10 min first timing update (~16.2 GB peak), then the queries below.
#
# vivado -mode batch -nojournal -log OUT.log -source census_failing_endpoints.tcl \
#   -tclargs ROUTED_DCP FRESH_OUTPUT_DIR [MAX_PATHS]
#
# Outputs (all new files in FRESH_OUTPUT_DIR):
#   failing_endpoints.tsv   one row per failing endpoint (worst path, -nworst 1)
#   design_analysis.rpt     report_design_analysis timing table (SLR crossings,
#                           high-fanout, logic levels) for the same population
#   design_analysis_csv/    the same table as CSV when the option is supported
#   top_paths_full.rpt      full clock-expanded detail of the 200 worst paths
#   logic_levels.rpt        logic-level distribution of failing paths
#   summary.tsv             counts that must match report_timing_summary
# Group the TSV with tools/r40/group_failing_paths.py.
if {[llength $argv] ni {2 3}} {error "expected ROUTED_DCP FRESH_OUTPUT_DIR ?MAX_PATHS?"}
lassign $argv checkpoint out max_paths
if {$max_paths eq ""} {set max_paths 25000}
if {![string is integer -strict $max_paths] || $max_paths < 1} {error "MAX_PATHS must be a positive integer"}
if {![file isfile $checkpoint] || [file exists $out]} {error "existing checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
set clock_name clk_kernel_00_unbuffered_net
set clk [get_clocks -quiet $clock_name]
if {[llength $clk] != 1} {error "expected one kernel clock $clock_name"}
set period [get_property PERIOD $clk]
puts "R40_CENSUS_PERIOD period_ns=$period"
# Population: worst path to every failing endpoint of the kernel path group.
set paths [get_timing_paths -setup -group $clock_name -max_paths $max_paths -nworst 1 \
    -unique_pins -slack_lesser_than 0]
set n [llength $paths]
puts "R40_CENSUS_PATHS count=$n max_paths=$max_paths"
proc r40_slr_of_pin {pin} {
    set cell [get_cells -quiet -of_objects [get_pins -quiet $pin]]
    if {[llength $cell] != 1} {return "?"}
    set slr [get_slrs -quiet -of_objects $cell]
    if {[llength $slr] != 1} {return "?"}
    return [get_property NAME $slr]
}
set f [open [file join $out failing_endpoints.tsv] {WRONLY CREAT EXCL}]
puts $f "slack_ns\tstartpoint\tendpoint\tlogic_levels\tdatapath_ns\tlogic_ns\tnet_ns\tskew_ns\tstart_slr\tend_slr"
set tns 0.0
foreach p $paths {
    set slack [get_property SLACK $p]
    set tns [expr {$tns + $slack}]
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    puts $f [join [list $slack $sp $ep \
        [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p] \
        [get_property DATAPATH_LOGIC_DELAY $p] [get_property DATAPATH_NET_DELAY $p] \
        [get_property SKEW $p] [r40_slr_of_pin $sp] [r40_slr_of_pin $ep]] \t]
}
close $f
# report_design_analysis adds SLR-crossing, high-fanout and primitive columns.
if {[catch {report_design_analysis -timing -setup -max_paths $max_paths -show_all \
        -file [file join $out design_analysis.rpt]} msg]} {
    puts "R40_CENSUS_WARNING design_analysis_failed $msg"
}
if {[catch {report_design_analysis -timing -setup -max_paths $max_paths -show_all \
        -csv [file join $out design_analysis_csv]} msg]} {
    puts "R40_CENSUS_WARNING design_analysis_csv_unsupported $msg"
}
if {[catch {report_design_analysis -logic_level_distribution -logic_level_dist_paths $max_paths \
        -file [file join $out logic_levels.rpt]} msg]} {
    puts "R40_CENSUS_WARNING logic_level_distribution_failed $msg"
}
report_timing -of_objects [lrange $paths 0 199] -path_type full_clock_expanded \
    -file [file join $out top_paths_full.rpt]
set s [open [file join $out summary.tsv] {WRONLY CREAT EXCL}]
puts $s "checkpoint\t$checkpoint"
puts $s "kernel_period_ns\t$period"
puts $s "failing_endpoints_enumerated\t$n"
puts $s "enumerated_tns_ns\t$tns"
puts $s "complete\t[expr {$n < $max_paths ? 1 : 0}]"
puts $s "scope\tread-only routed checkpoint census; no implementation or board access"
# Residual census: -nworst 1 hides any failing path of an endpoint whose
# worst path runs through the known attempt-3 control cone. To expose what
# would remain once that cone is registered, the cone is excluded IN MEMORY
# ONLY for this read-only analysis session (never written to an XDC or
# checkpoint; this is not a design constraint and qualifies nothing).
set cone_from [get_cells -quiet -hierarchical -filter {NAME =~ *inst/device/control/memory_active_ports_reg*}]
set cone_through [get_nets -quiet -hierarchical -filter {NAME =~ *inst/device/system/* && NAME =~ *host_allow*}]
puts $s "residual_exclusion_from_cells\t[llength $cone_from]"
puts $s "residual_exclusion_through_nets\t[llength $cone_through]"
close $s
if {[llength $cone_from] > 0 || [llength $cone_through] > 0} {
    if {[llength $cone_from] > 0} {set_false_path -from $cone_from}
    if {[llength $cone_through] > 0} {set_false_path -through $cone_through}
    update_timing
    set rpaths [get_timing_paths -setup -group $clock_name -max_paths $max_paths -nworst 1 \
        -unique_pins -slack_lesser_than 0]
    set r [open [file join $out residual_failing_endpoints.tsv] {WRONLY CREAT EXCL}]
    puts $r "slack_ns\tstartpoint\tendpoint\tlogic_levels\tdatapath_ns\tlogic_ns\tnet_ns\tskew_ns\tstart_slr\tend_slr"
    set rtns 0.0
    foreach p $rpaths {
        set slack [get_property SLACK $p]
        set rtns [expr {$rtns + $slack}]
        set sp [get_property STARTPOINT_PIN $p]
        set ep [get_property ENDPOINT_PIN $p]
        puts $r [join [list $slack $sp $ep \
            [get_property LOGIC_LEVELS $p] [get_property DATAPATH_DELAY $p] \
            [get_property DATAPATH_LOGIC_DELAY $p] [get_property DATAPATH_NET_DELAY $p] \
            [get_property SKEW $p] [r40_slr_of_pin $sp] [r40_slr_of_pin $ep]] \t]
    }
    close $r
    report_timing -of_objects [lrange $rpaths 0 199] -path_type full_clock_expanded \
        -file [file join $out residual_top_paths_full.rpt]
    puts "R40_CENSUS_RESIDUAL count=[llength $rpaths] tns=$rtns (what-if exclusion of the known cone; analysis only)"
}
puts "R40_CENSUS_DONE count=$n tns=$tns"
exit
