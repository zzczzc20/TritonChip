#==============================================================================
# lane_ooc.tcl -- out-of-context synthesis, placement and routing of one MAC
# lane (pmac_ooc_top), U280 part. Never touches a card.
#
#   OOC_GENERICS="LANE=1 VARIANT=2 ..." vivado -mode batch -nojournal -nolog \
#       -source lane_ooc.tcl -tclargs <period_ns> <part> <out> <rtl> <files...>
#
# The clock is constrained before synthesis (timing-driven synthesis), on an
# ideal source (no HD.CLK_SRC): skew and clock insertion are not modelled.
# Reports, post-synthesis and post-route (<out>/synth_* and route_*):
#   *_util.rpt, *_util_hier.rpt (the lane instance alone), *_summary.rpt,
#   *_paths.rpt (20 worst paths, unique endpoints), *_stages.tsv (the worst
#   path INTO each pipeline boundary: IN_REG, c0..c3, the accumulator register).
#==============================================================================
set period [lindex $::argv 0]
set part   [lindex $::argv 1]
set out    [lindex $::argv 2]
set rtl    [lindex $::argv 3]
set files  [lrange $::argv 4 end]
file mkdir $out
foreach f $files {
    if {[string match /* $f]} { read_verilog -sv $f } else { read_verilog -sv [file join $rtl $f] }
}
set_param general.maxThreads 4
set gen [list]
if {[info exists ::env(OOC_GENERICS)]} {
    foreach g [split $::env(OOC_GENERICS) " "] { if {$g ne ""} { lappend gen -generic $g } }
}
set xdc [file join $out clk.xdc]
set fh [open $xdc w]; puts $fh "create_clock -name clk -period $period \[get_ports clk\]"; close $fh
read_xdc -mode out_of_context $xdc

synth_design -top pmac_ooc_top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
    -include_dirs [list [file join $rtl common]] {*}$gen

proc cells {pat} { return [get_cells -quiet -hier -filter "NAME =~ $pat && IS_SEQUENTIAL"] }

proc stage_report {tag out} {
    # boundary name, endpoint pattern(s)
    set groups {
        {in_reg   {*u_lane/g_in_reg.*}}
        {c0       {*u_lane/g_c0.*}}
        {c1       {*u_lane/g_c1.*}}
        {c2       {*u_lane/g_c2.* *u_lane/u_fma/g_l2.s0_q*}}
        {c3       {*u_lane/g_c3.* *u_lane/u_fma/g_l2.s1_q*}}
        {acc      {y_q_reg*}}
    }
    set fh [open [file join $out ${tag}_stages.tsv] w]
    puts $fh "boundary\tendpoints\tslack_ns\tdatapath_ns\tlogic_ns\tnet_ns\tlevels\tstart\tend"
    set rf [file join $out ${tag}_stages.rpt]
    file delete -force $rf
    foreach g $groups {
        lassign $g name pats
        set tc {}
        foreach p $pats { foreach c [cells $p] { lappend tc $c } }
        if {[llength $tc] == 0} { puts $fh "$name\t0\tNA\tNA\tNA\tNA\tNA\t\t"; continue }
        set p [get_timing_paths -quiet -to $tc -max_paths 1 -nworst 1 -setup]
        if {[llength $p] == 0} { puts $fh "$name\t[llength $tc]\tNA\tNA\tNA\tNA\tNA\t(no timed path)\t"; continue }
        puts $fh [join [list $name [llength $tc] [get_property SLACK $p] [get_property DATAPATH_DELAY $p] \
            [get_property DATAPATH_LOGIC_DELAY $p] [get_property DATAPATH_NET_DELAY $p] \
            [get_property LOGIC_LEVELS $p] \
            [get_property PARENT_CELL [get_property STARTPOINT_PIN $p]] \
            [get_property PARENT_CELL [get_property ENDPOINT_PIN $p]]] "\t"]
        set rh [open $rf a]; puts $rh "==== $name"; close $rh
        report_timing -of_objects $p -append -file $rf
    }
    close $fh
}

proc reports {tag out} {
    report_utilization -file [file join $out ${tag}_util.rpt]
    report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out ${tag}_util_hier.rpt]
    report_timing_summary -no_header -no_detailed_paths -file [file join $out ${tag}_summary.rpt]
    report_timing -max_paths 20 -nworst 1 -unique_pins -file [file join $out ${tag}_paths.rpt]
    stage_report $tag $out
}

reports synth $out
opt_design
place_design
phys_opt_design
route_design
reports route $out
puts "DONE"
