# Read-only classification of actual SLR crossings at the three tile boundaries.
# vivado -mode batch -source THIS_FILE -tclargs CHECKPOINT NEW_OUTPUT
# Counts are boundary pins and unique connected nets, not SLL routing capacity.
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
set summary [open $out/summary.tsv {WRONLY CREAT EXCL}]
puts $summary "tile\tport_group\tdirection\tactual_slrs\tboundary_pin_crossings"
foreach key [lsort [dict keys $counts]] {
    puts $summary [join [concat $key [list [dict get $counts $key]]] "\t"]
}
close $summary
set status [open $out/status.txt {WRONLY CREAT EXCL}]
puts $status "scope diagnostic_only_actual_endpoint_regions_not_column_capacity"
puts $status "checkpoint $checkpoint"
puts $status "tiles [llength $tiles] considered_boundary_pins $considered excluded_clock_reset_pins $excluded excluded_constant_pins $constant_pins"
puts $status "unique_crossing_nets [dict size $unique]"
puts $status "reports_complete 1"
close $status
puts "R40_BOUNDARY_DIAGNOSIS_DONE"
exit
