# PLACE_DESIGN.TCL.PRE for the three-tile U280 platform binding.
# Logical RTL stays platform independent. This binding requires two stages.
# A missing hierarchy is fatal; do not silently produce a single-SLR image.
proc r40_place {suffix slr} {
    set matches [list]
    foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}] {
        set name [get_property NAME $cell]
        if {[string match "*/$suffix" $name]} {lappend matches $cell}
    }
    if {[llength $matches] != 1} {
        error "R40 placement: expected one $suffix, found [llength $matches]"
    }
    set_property USER_SLR_ASSIGNMENT SLR$slr $matches
    puts "R40_PLACEMENT $matches SLR$slr"
}
# Escape literal generate brackets for string match (Tcl glob syntax).
proc r40_index {name index} {return [format {%s\[%d\]} $name $index]}
set y [r40_index g_y 0]
for {set tile 0} {$tile < 3} {incr tile} {
    set t [r40_index g_tile $tile]
    r40_place "u_system/$t.u_tile" $tile
    set x [r40_index g_x $tile]
    r40_place "u_system/u_mesh/$y.$x.u_r" $tile
    set c [r40_index g_command_link $tile]
    set h [r40_index g_home_link $tile]
    for {set stage 0} {$stage < 2} {incr stage} {
        set s [r40_index g_stage $stage]
        r40_place "u_system/g_dispatch/$c.commands/$s.queue" [expr {$stage == 0 ? 1 : $tile}]
        r40_place "u_system/g_dispatch/$c.g_pipeline.completions/$s.queue" [expr {$stage == 0 ? $tile : 1}]
        set home_slr [expr {$stage == 0 ? $tile : 1}]
        r40_place "u_system/$h.g_pipeline.link/$s.request" $home_slr
        set responses [list]
        foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}] {
            set name [get_property NAME $cell]
            if {[string match "*/u_system/$h.g_pipeline.link/$s.rd_reg*" $name] ||
                [string match "*/u_system/$h.g_pipeline.link/$s.rv_reg*" $name]} {
                lappend responses $cell
            }
        }
        if {[llength $responses] == 0} {error "R40 placement: missing home response pipeline $tile/$stage"}
        set_property USER_SLR_ASSIGNMENT SLR$home_slr $responses
    }
    foreach direction {e w} {
        set peer [expr {$direction eq "e" ? $tile+1 : $tile-1}]
        if {$peer < 0 || $peer > 2} {continue}
        for {set vc 0} {$vc < 2} {incr vc} {
            set v [r40_index g_v $vc]
            for {set stage 0} {$stage < 2} {incr stage} {
                set s [r40_index g_stage $stage]
                r40_place "u_system/u_mesh/$y.$x.$v.g_$direction.link/$s.queue" [expr {$stage == 0 ? $tile : $peer}]
            }
        }
    }
}
# HBM-facing links/adapters stay in SLR0. This must be checked in routed DCP.
foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}] {
    set name [get_property NAME $cell]
    if {[regexp {/g_outlet\[[0-2]\]\.adapter$} $name]} {
        set_property USER_SLR_ASSIGNMENT SLR0 $cell
        puts "R40_PLACEMENT $cell SLR0"
    }
}
