# Read-only classification of actual SLR crossings at the three tile boundaries.
# vivado -mode batch -source THIS_FILE -tclargs CHECKPOINT NEW_OUTPUT
# Counts are boundary pins and unique connected nets, not SLL routing capacity.
# Count each canonical connected net once. These are endpoint-region crossings,
# not routed SLL usage; SLICE X is a driver position, not an SLL column number.
proc r40_driver_group {driver} {
    if {[regexp {^(.*?/u_system/(g_tile\[[0-9]+\]\.u_tile/(u_core|u_slice)|u_tmu))/} $driver whole group]} {return $group}
    return [join [lrange [split $driver /] 0 end-1] /]
}
proc r40_rank_crossings {cache out} {
    set modules [dict create]
    set columns [dict create]
    dict for {net entry} $cache {
        if {![dict get $entry crossing]} {continue}
        set group [r40_driver_group [lindex [dict get $entry drivers] 0]]
        set regions [dict get $entry regions]
        dict incr modules [list $group $regions]
        set seen [dict create]
        foreach loc [dict get $entry locs] {
            if {[regexp {^SLICE_X([0-9]+)Y} $loc whole x]} {dict set seen $x 1}
        }
        foreach x [dict keys $seen] {dict incr columns [list $group $x]}
    }
    set ranked [list]
    dict for {key count} $modules {lappend ranked [list $count $key]}
    set f [open $out/cross_slr_module_ranking.tsv {WRONLY CREAT EXCL}]
    puts $f "driver_module\tactual_endpoint_slrs\tunique_connected_crossing_nets"
    foreach row [lsort -integer -decreasing -index 0 $ranked] {
        puts $f [join [concat [lindex $row 1] [list [lindex $row 0]]] "\t"]
    }
    close $f
    set f [open $out/cross_slr_driver_columns.tsv {WRONLY CREAT EXCL}]
    puts $f "driver_module\tdriver_slice_x_NOT_SLL_column\tunique_crossing_nets"
    foreach key [lsort [dict keys $columns]] {
        puts $f [join [concat $key [list [dict get $columns $key]]] "\t"]
    }
    close $f
}

if {[llength $argv] != 2} {error "expected CHECKPOINT NEW_OUTPUT"}
set checkpoint [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file isfile $checkpoint] || [file exists $out]} {
    error "checkpoint must exist and output must be fresh"
}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
set tiles [get_cells -hierarchical -quiet -filter {NAME =~ "*u_system/g_tile*.u_tile"}]
if {[llength $tiles] != 3} {error "expected three retained tile boundaries, found [llength $tiles]"}
set detail [open $out/crossings.tsv {WRONLY CREAT EXCL}]
puts $detail "tile\tpin\tdirection\tactual_slrs\tcanonical_net\tleaf_cells\tdriver_cells\tdriver_locs"
set routed_columns [open $out/routed_sll_columns.tsv {WRONLY CREAT EXCL}]
puts $routed_columns "canonical_net\tactual_endpoint_slrs\trouted_laguna_tiles\tmatched_route_nodes"
set cache [dict create]
set constant_pins 0
set counts [dict create]
set unique [dict create]
set considered 0
set excluded 0
foreach tile $tiles {
    set tile_name [get_property NAME $tile]
    foreach pin [get_pins -quiet -of_objects $tile] {
        set name [get_property NAME $pin]
        set port [file tail $name]
        # Dedicated clock/reset fanout is reported as excluded; do not attribute
        # it to fabric SLL demand by counting every clock sink as a data net.
        if {[regexp -nocase {(^|_)(clk|clock|rst|reset)(_|\[|$)} $port]} {
            incr excluded
            continue
        }
        incr considered
        if {$considered > 20000} {error "boundary-pin execution budget exceeded; partial reports retained"}
        set nets [get_nets -segments -quiet -of_objects $pin]
        if {![llength $nets]} {continue}
        set canonical [lindex [lsort [get_property NAME $nets]] 0]
        if {![dict exists $cache $canonical]} {
            set driver_pins [get_pins -leaf -quiet -of_objects $nets -filter {DIRECTION == OUT}]
            set drivers [list]
            set references [list]
            set names [list]
            set locs [list]
            if {[llength $driver_pins]} {set drivers [get_cells -quiet -of_objects $driver_pins]}
            if {[llength $drivers]} {
                set references [get_property REF_NAME $drivers]
                set names [get_property NAME $drivers]
                set locs [get_property LOC $drivers]
            }
            set constant [expr {[llength $drivers] > 0}]
            foreach reference $references {
                if {$reference ni {GND VCC}} {set constant 0}
            }
            if {$constant} {
                dict set cache $canonical [dict create crossing 0 constant 1]
            } else {
                set leaf_pins [get_pins -leaf -quiet -of_objects $nets]
                set cells [list]
                set regions [list]
                set region_names [list]
                if {[llength $leaf_pins]} {set cells [get_cells -quiet -of_objects $leaf_pins]}
                if {[llength $cells]} {set regions [get_slrs -quiet -of_objects $cells]}
                if {[llength $regions]} {set region_names [lsort -unique [get_property NAME $regions]]}
                set entry [dict create crossing [expr {[llength $region_names] > 1}] constant 0]
                dict set entry regions $region_names
                dict set entry cells [llength $cells]
                dict set entry drivers $names
                dict set entry locs $locs
                if {[llength $region_names] > 1} {
                    # Only existing routed nodes; absent nodes in this failed
                    # checkpoint do not imply absent crossing demand.
                    set lag_nodes [get_nodes -quiet -of_objects $nets -filter {NAME =~ "*LAG*"}]
                    set lag_tiles [list]
                    if {[llength $lag_nodes]} {
                        set lag_tiles [lsort -unique [get_property NAME [get_tiles -quiet -of_objects $lag_nodes]]]
                    }
                    puts $routed_columns [join [list $canonical $region_names $lag_tiles [llength $lag_nodes]] "\t"]
                    flush $routed_columns
                }
                dict set cache $canonical $entry
            }
        }
        set entry [dict get $cache $canonical]
        if {[dict get $entry constant]} {incr constant_pins; continue}
        if {![dict get $entry crossing]} {continue}
        set region_names [dict get $entry regions]
        set direction [get_property DIRECTION $pin]
        set group [regsub {\[[0-9]+\]$} $port {}]
        set key [list $tile_name $group $direction $region_names]
        dict incr counts $key
        dict set unique $canonical 1
        set driver_names [dict get $entry drivers]
        set driver_locs [dict get $entry locs]
        set cell_count [dict get $entry cells]
        puts $detail [join [list $tile_name $port $direction $region_names $canonical $cell_count $driver_names $driver_locs] "\t"]
    }
    flush $detail
}
close $detail
close $routed_columns
set summary [open $out/summary.tsv {WRONLY CREAT EXCL}]
puts $summary "tile\tport_group\tdirection\tactual_slrs\tboundary_pin_crossings"
foreach key [lsort [dict keys $counts]] {
    puts $summary [join [concat $key [list [dict get $counts $key]]] "\t"]
}
close $summary
r40_rank_crossings $cache $out
# Actual primitive placements, grouped independently from pblock assignments.
# Include internal core/slice logic and all preserved elastic stage state.
set primitives [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE}]
set primitive_names [get_property NAME $primitives]
set placements [dict create]
foreach cell $primitives name $primitive_names ref [get_property REF_NAME $primitives] {
    if {[regexp {^(.*?/u_system/g_tile\[[0-2]\]\.u_tile/(u_core|u_slice))/} $name whole module]} {
        dict lappend placements $module $cell
    }
    if {[regexp {^(FD|RAM)} $ref] && [regexp {^(.*?/g_stage\[[0-9]+\])([./])} $name whole stage]} {
        set kind other
        if {[regexp {(rv_reg|valid[^/]*reg)} $name]} {set kind valid}
        if {[regexp {(cnt_reg|count_reg|ptr_reg)} $name]} {set kind occupancy_ready}
        if {[regexp {(mem_reg|rd_reg)} $name]} {set kind payload}
        dict lappend placements "$stage:$kind" $cell
    }
}
set place_report [open $out/actual_placement.tsv {WRONLY CREAT EXCL}]
puts $place_report "scope\tprimitive_count\tplaced_count\tunplaced_nonconstant\tactual_slrs\tsample_actual_locs"
foreach scope [lsort [dict keys $placements]] {
    set members [dict get $placements $scope]
    set placed [list]
    set missing 0
    foreach cell $members loc [get_property LOC $members] ref [get_property REF_NAME $members] {
        if {$ref in {GND VCC}} {continue}
        if {$loc eq ""} {incr missing} else {lappend placed $cell}
    }
    set slrs [list]
    set samples [list]
    if {[llength $placed]} {
        set slrs [lsort -unique [get_property NAME [get_slrs -quiet -of_objects $placed]]]
        set samples [lrange [get_property LOC $placed] 0 7]
    }
    puts $place_report [join [list $scope [llength $members] [llength $placed] $missing $slrs $samples] "\t"]
}
close $place_report
report_control_sets -verbose -file $out/control_sets.rpt
set protected [open $out/protected_cells.tsv {WRONLY CREAT EXCL}]
puts $protected "cell\tprimitive\tloc"
foreach cell [get_cells -hierarchical -quiet -filter {DONT_TOUCH == 1}] {
    puts $protected "[get_property NAME $cell]\t[get_property REF_NAME $cell]\t[get_property LOC $cell]"
}
close $protected
set status [open $out/status.txt {WRONLY CREAT EXCL}]
puts $status "scope diagnostic_only_actual_endpoint_regions_not_column_capacity"
puts $status "checkpoint $checkpoint"
puts $status "tiles [llength $tiles] considered_boundary_pins $considered excluded_clock_reset_pins $excluded excluded_constant_pins $constant_pins"
puts $status "unique_crossing_nets [dict size $unique]"
puts $status "routed_sll_column_coverage partial_existing_LAG_nodes_only_missing_nodes_are_not_zero_demand"
puts $status "reports_complete 1"
close $status
puts "R40_BOUNDARY_DIAGNOSIS_DONE"
exit
