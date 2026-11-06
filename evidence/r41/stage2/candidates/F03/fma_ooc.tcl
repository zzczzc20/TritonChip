#==============================================================================
# fma_ooc.tcl -- the unit OOC screen of the FMA-datapath candidates F03/F08/F09
#
#   OOC_SV=1 OOC_GENERICS="..." vivado -mode batch -nojournal -nolog \
#       -source fma_ooc.tcl -tclargs <top> <period> <part> <out> <rtl> <files...>
#
# Runs flow/synth/vivado_ooc_timing.tcl unchanged (same arguments), then, in
# the same session, reports the worst register-to-register path of each FMA
# pipeline stage and of each census family by name, one TSV line each
# (<top>_families.tsv). Stage families are matched structurally: any p<N>_
# register of an XLS-scheduled FMA instance (the census tree), or the s0_q /
# s1_q cut registers of the latency-2 instance since R41 Stage 2. The by-name
# families resolve only where the census register names still exist.
# With FAMILIES_ONLY=1 and a DCP as the only argument after <top> <period>
# <part> <out>, it opens that checkpoint instead of synthesising.
# Synthesis only; never touches a card.
#==============================================================================
# OOC_TCL is the tree's flow/synth/vivado_ooc_timing.tcl.
if {[info exists ::env(FAMILIES_ONLY)] && $::env(FAMILIES_ONLY)} {
    open_checkpoint [lindex $::argv 4]
} else {
    source $::env(OOC_TCL)
}

set top [lindex $::argv 0]
set out [lindex $::argv 3]

proc cells {pat} { return [get_cells -quiet -hier -filter "NAME =~ $pat && IS_PRIMITIVE"] }

# name {from-pattern} {to-pattern}
set fams {}
switch -glob $top {
    tnca_vector_fpu {
        lappend fams [list s0_red_busy_to_p0   {red_busy_reg}                 {*u_xls/p0_*}]
        lappend fams [list s0_red_small_wins   {red_busy_reg}                 {*u_xls/p0_small_wins_reg*}]
        lappend fams [list s0_red_res_sign_mag {red_busy_reg}                 {*u_xls/p0_res_sign_mag_reg*}]
        lappend fams [list s0_red_sub          {red_busy_reg}                 {*u_xls/p0_sub_671_reg*}]
        lappend fams [list s0_any_to_p0        {}                             {*u_xls/p0_*}]
        lappend fams [list s1_p0_to_p1         {*u_xls/p0_*}                  {*u_xls/p1_*}]
        lappend fams [list s1_sticky_to_sh_mag {*u_xls/p0_sticky_al_reg*}     {*u_xls/p1_sh_mag_reg*}]
        lappend fams [list s2_p1_to_p2         {*u_xls/p1_*}                  {*u_xls/p2_*}]
        lappend fams [list sL_last_to_acc      {*u_xls/p*}                    {acc_reg*}]
        lappend fams [list sL_sh_mag_to_acc    {*u_xls/p1_sh_mag_reg*}        {acc_reg*}]
        lappend fams [list cvt_to_dly          {cvt_*}                        {dly_v_reg*}]
        lappend fams [list c0_red_busy_to_s0q  {red_busy_reg*}                {*g_l2.s0_q_reg*}]
        lappend fams [list c0_any_to_s0q       {}                             {*g_l2.s0_q_reg*}]
        lappend fams [list c1_s0q_to_s1q       {*g_l2.s0_q_reg*}              {*g_l2.s1_q_reg*}]
        lappend fams [list c2_s1q_to_acc       {*g_l2.s1_q_reg*}              {acc_reg*}]
        lappend fams [list c2_s1q_to_any       {*g_l2.s1_q_reg*}              {}]
    }
    tnca_matrix {
        lappend fams [list s0_inreg_to_p0      {*g_in_reg.*}                  {*u_xls/p0_*}]
        lappend fams [list s0_a_q_small_wins   {*g_in_reg.a_q_reg*}           {*u_xls/p0_small_wins_reg*}]
        lappend fams [list s1_p0_to_p1         {*u_xls/p0_*}                  {*u_xls/p1_*}]
        lappend fams [list s1_sub_to_sh_mag    {*u_xls/p0_sub_671_reg*}       {*u_xls/p1_sh_mag_reg*}]
        lappend fams [list s1_sticky_to_sh_mag {*u_xls/p0_sticky_al_reg*}     {*u_xls/p1_sh_mag_reg*}]
        lappend fams [list s2_p1_to_p2         {*u_xls/p1_*}                  {*u_xls/p2_*}]
        lappend fams [list sL_last_to_lane_mem {*u_xls/p*}                    {*u_acc/*lane_mem*}]
        lappend fams [list sL_sh_mag_to_lane_mem {*u_xls/p1_sh_mag_reg*}      {*u_acc/*lane_mem*}]
        lappend fams [list any_to_lane_mem     {}                             {*u_acc/*lane_mem*}]
        lappend fams [list c0_inreg_to_s0q     {*g_in_reg.*}                  {*g_l2.s0_q_reg*}]
        lappend fams [list c1_s0q_to_s1q       {*g_l2.s0_q_reg*}              {*g_l2.s1_q_reg*}]
        lappend fams [list c2_s1q_to_lane_mem  {*g_l2.s1_q_reg*}              {*u_acc/*lane_mem*}]
    }
    default {
        lappend fams [list s1_p0_to_p1         {*p0_*}                        {*p1_*}]
        lappend fams [list s1_sticky_to_sh_mag {*p0_sticky_al_reg*}           {*p1_sh_mag_reg*}]
        lappend fams [list s2_p1_to_p2         {*p1_*}                        {*p2_*}]
        lappend fams [list c1_s0q_to_s1q       {*g_l2.s0_q_reg*}              {*g_l2.s1_q_reg*}]
    }
}

set fh [open [file join $out ${top}_families.tsv] w]
puts $fh "family\tslack_ns\tdatapath_ns\tlogic_ns\tnet_ns\tlevels\tstart\tend"
set rf [file join $out ${top}_families.rpt]
file delete -force $rf
foreach f $fams {
    lassign $f name from to
    if {$to eq ""} {
        set tc [all_registers]
    } else {
        set tc [cells $to]
    }
    if {$from eq ""} {
        set fc [all_registers]
    } else {
        set fc [cells $from]
    }
    if {[llength $fc] == 0 || [llength $tc] == 0} {
        puts $fh "$name\tNA\tNA\tNA\tNA\tNA\t(no cells: [llength $fc] from, [llength $tc] to)\t"
        continue
    }
    set p [get_timing_paths -quiet -from $fc -to $tc -max_paths 1 -nworst 1 -setup]
    if {[llength $p] == 0} {
        puts $fh "$name\tNA\tNA\tNA\tNA\tNA\t(no timed path)\t"
        continue
    }
    set dp [get_property DATAPATH_DELAY $p]
    set lg [get_property DATAPATH_LOGIC_DELAY $p]
    set nt [get_property DATAPATH_NET_DELAY $p]
    puts $fh [join [list $name [get_property SLACK $p] $dp $lg $nt \
        [get_property LOGIC_LEVELS $p] \
        [get_property PARENT_CELL [get_property STARTPOINT_PIN $p]] \
        [get_property PARENT_CELL [get_property ENDPOINT_PIN $p]]] "\t"]
    set rh [open $rf a]; puts $rh "==== $name"; close $rh
    report_timing -of_objects $p -append -file $rf
}
close $fh
puts "FAMILIES $top"
