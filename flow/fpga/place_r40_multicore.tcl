# PLACE_DESIGN.TCL.PRE for the three-tile U280 platform binding.
# Logical RTL stays platform independent. This binding requires two stages.
# A missing hierarchy is fatal; do not silently produce a single-SLR image.
set r40_hier_cells [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}]
set r40_hier_names [get_property NAME $r40_hier_cells]
set r40_leaf_cells [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}]
set r40_leaf_names [get_property NAME $r40_leaf_cells]
proc r40_assign {cells slr} {
    # USER_SLR_ASSIGNMENT is ignored on leaves. Pblocks constrain both
    # hierarchical groups and primitive response/code storage cells.
    set name r40_slr$slr
    set block [get_pblocks -quiet $name]
    if {[llength $block] == 0} {
        set block [create_pblock $name]
        resize_pblock $block -add SLR$slr
    }
    add_cells_to_pblock $block $cells
}
proc r40_place {suffix slr} {
    global r40_hier_cells r40_hier_names
    set matches [list]
    foreach cell $r40_hier_cells name $r40_hier_names {
        if {[string match "*/$suffix" $name]} {lappend matches $cell}
    }
    if {[llength $matches] != 1} {
        error "R40 placement: expected one $suffix, found [llength $matches]"
    }
    r40_assign $matches $slr
    puts "R40_PLACEMENT $matches SLR$slr"
}
# Escape literal generate brackets for string match (Tcl glob syntax).
proc r40_index {name index} {return [format {%s\[%d\]} $name $index]}
proc r40_place_responses {stage_suffix slr} {
    global r40_leaf_cells r40_leaf_names
    set responses [list]
    foreach cell $r40_leaf_cells name $r40_leaf_names {
        if {[string match "*/$stage_suffix.rd_reg*" $name] ||
            [string match "*/$stage_suffix.rv_reg*" $name]} {
            lappend responses $cell
        }
    }
    if {[llength $responses] == 0} {error "R40 placement: missing response pipeline $stage_suffix"}
    r40_assign $responses $slr
}
proc r40_guide_response_crossing {link_suffix} {
    global r40_leaf_cells r40_leaf_names
    set sender [r40_index g_stage 1]
    set receiver [r40_index g_stage 0]
    set pairs 0
    set skipped 0
    set sender_pattern "$link_suffix/$sender.rd_reg*"
    set receiver_pattern "$link_suffix/$receiver.rd_reg*"
    foreach cell $r40_leaf_cells name $r40_leaf_names {
        if {(![string match "*/$sender_pattern" $name] &&
             ![string match $sender_pattern $name]) ||
            ![string match FD* [get_property REF_NAME $cell]]} {continue}
        set q [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == Q}]
        set nets [get_nets -segments -quiet -of_objects $q]
        set loads [get_pins -leaf -quiet -of_objects $nets -filter {DIRECTION == IN}]
        # Guide only the direct, single-load payload FF pair screened on U280.
        # Valid/ready and locally consumed payloads retain ordinary placement.
        if {[llength $loads] != 1 || [get_property REF_PIN_NAME $loads] ne "D"} {
            incr skipped
            continue
        }
        set destination [get_cells -quiet -of_objects $loads]
        if {![string match FD* [get_property REF_NAME $destination]] ||
            (![string match "*/$receiver_pattern" [get_property NAME $destination]] &&
             ![string match $receiver_pattern [get_property NAME $destination]])} {
            incr skipped
            continue
        }
        # Preserve Vivado object handles; concat can stringify a collection
        # into an invalid object argument when the hierarchy contains brackets.
        set_property USER_SLL_REG TRUE $cell
        set_property USER_SLL_REG TRUE $destination
        incr pairs
    }
    if {$pairs == 0} {error "R40 placement: no eligible response FF pairs in $link_suffix"}
    puts "R40_RESPONSE_SLL_GUIDANCE link=$link_suffix pairs=$pairs skipped=$skipped"
    # This is placement guidance. Final LOC/BEL and setup/hold still need audit.
}
set y [r40_index g_y 0]
set r40_maintenance 0
foreach name $r40_hier_names {
    if {[string match {*/u_system/g_cache_handoff.ownership} $name]} {set r40_maintenance 1}
}
if {$r40_maintenance} {r40_place "u_system/g_cache_handoff.ownership" 1}
# Coherent local TMU state must remain beside its consuming private slice.
# Match retained local state hierarchy even if an intermediate wrapper rebuilds.
set r40_local_tmu_count 0
for {set tile 0} {$tile < 3} {incr tile} {
    set suffix "[r40_index g_replica $tile].local_tmu"
    set matches [list]
    foreach cell $r40_hier_cells name $r40_hier_names {
        if {[string match "*/u_system/*/$suffix" $name]} {lappend matches $cell}
    }
    if {[llength $matches] > 1} {error "R40 placement: duplicate local TMU $suffix"}
    if {[llength $matches] == 1} {
        r40_assign $matches $tile
        incr r40_local_tmu_count
        puts "R40_PLACEMENT_LOCAL_TMU $matches SLR$tile"
    }
}
if {$r40_local_tmu_count != 0 && $r40_local_tmu_count != 3} {
    error "R40 placement: incomplete coherent local TMU replicas ($r40_local_tmu_count)"
}
if {[info exists TNCA_IMAGE_PARAMS] && [dict exists $TNCA_IMAGE_PARAMS DCO_LOCAL_REPLICAS]} {
    set expected_local [expr {[dict get $TNCA_IMAGE_PARAMS DCO_LOCAL_REPLICAS] ? 3 : 0}]
    if {$r40_local_tmu_count != $expected_local} {
        error "R40 placement: local TMU hierarchy disagrees with image configuration"
    }
}

for {set tile 0} {$tile < 3} {incr tile} {
    set t [r40_index g_tile $tile]
    r40_place "u_system/$t.u_tile" $tile
    # Private instruction RAM is owned by the shell, outside u_tile.
    # Keep RAMs and their read mux/registers beside the consuming core.
    set code [r40_index g_code $tile]
    set private_code [list]
    foreach cell $r40_leaf_cells name $r40_leaf_names {
        if {[string match "*/shell/$code.*" $name] &&
            ![string match "*/shell/$code.commands/*" $name] &&
            ![string match "*/shell/$code.replies/*" $name]} {lappend private_code $cell}
    }
    if {[llength $private_code] == 0} {error "R40 placement: private code RAM missing for tile $tile"}
    r40_assign $private_code $tile
    puts "R40_PLACEMENT_PRIVATE_CODE tile=$tile cells=[llength $private_code] SLR$tile"
    # Source command stage is central; private RAM and first reply stage are local.
    for {set stage 0} {$stage < 2} {incr stage} {
        set s [r40_index g_stage $stage]
        r40_place "shell/$code.commands/$s.queue" [expr {$stage == 0 ? 1 : $tile}]
        # Only core zero returns read data. Other reply payloads may be optimized
        # to zero width; their valid/count state still belongs to the queue.
        r40_place "shell/$code.replies/$s.queue" [expr {$stage == 0 ? $tile : 1}]
    }

    set x [r40_index g_x $tile]
    r40_place "u_system/u_mesh/$y.$x.u_r" $tile
    set c [r40_index g_command_link $tile]
    set h [r40_index g_home_link $tile]
    for {set stage 0} {$stage < 2} {incr stage} {
        set s [r40_index g_stage $stage]
        r40_place "u_system/g_dispatch.$c.commands/$s.queue" [expr {$stage == 0 ? 1 : $tile}]
        r40_place "u_system/g_dispatch.$c.g_pipeline.completions/$s.queue" [expr {$stage == 0 ? $tile : 1}]
        set home_slr [expr {$stage == 0 ? $tile : 1}]
        r40_place "u_system/$h.g_pipeline.link/$s.request" $home_slr
        r40_place_responses "u_system/$h.g_pipeline.link/$s" $home_slr
        if {$r40_maintenance} {
            r40_place "u_system/g_cache_handoff.$h.command/$s.queue" [expr {$stage == 0 ? 1 : $tile}]
            r40_place "u_system/g_cache_handoff.$h.response/$s.queue" [expr {$stage == 0 ? $tile : 1}]
        }

    }
    if {$tile != 1} {r40_guide_response_crossing "u_system/$h.g_pipeline.link"}
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
# One or three active outlets use the same tile/home/PC capacity.
set adapters [list]
foreach cell $r40_hier_cells name $r40_hier_names {
    if {[regexp {/g_outlet\[([0-2])\]\.adapter$} $name unused port]} {
        r40_assign $cell 0
        lappend adapters $cell
        puts "R40_PLACEMENT $cell SLR0"
        set outlet [r40_index g_outlet $port]
        for {set stage 0} {$stage < 2} {incr stage} {
            set s [r40_index g_stage $stage]
            set slr [expr {$stage == 0 ? 1 : 0}]
            r40_place "system/$outlet.link/$s.request" $slr
            r40_place_responses "system/$outlet.link/$s" $slr
        }
        r40_guide_response_crossing "system/$outlet.link"
    }
}
if {[llength $adapters] < 1 || [llength $adapters] > 3} {
    error "R40 placement: expected 1..3 HBM-facing adapters, found [llength $adapters]"
}
