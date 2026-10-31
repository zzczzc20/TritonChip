# Exploratory read-only timing sensitivity, not bitstream signoff.
# Recreate the existing 450 MHz generated clock first, then analyze 433.4 MHz.
# No checkpoint or image is written. Compare the recreated 450 MHz timing with
# the original routed report before interpreting the changed-constraint run.
lassign $argv checkpoint out
if {![file isfile $checkpoint] || [file exists $out]} {error "checkpoint and fresh output required"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $checkpoint
set clock [get_clocks -quiet hbm_aclk]
if {[llength $clock] != 1 || ![get_property IS_GENERATED $clock]} {error "HBM clock is not the expected generated clock"}
set master [get_clocks -quiet io_clk_freerun_00_clk_p]
set source [get_pins -quiet [get_property SOURCE $clock]]
set sink [get_pins -quiet [get_property SOURCE_PINS $clock]]
if {[llength $master] != 1 || [llength $source] != 1 || [llength $sink] != 1 ||
    [get_property MASTER_CLOCK $clock] ne "io_clk_freerun_00_clk_p" ||
    abs([get_property PERIOD $master]-10.0) > 0.001 ||
    abs([get_property PERIOD $clock]-2.222) > 0.001} {error "unexpected routed clock definition"}
foreach {label mhz} {recreated450 450.0 exploratory433p4 433.4} {
    set period [expr {1000.0/$mhz}]
    set shift [list 0.0 [expr {$period/2.0-5.0}] [expr {$period-10.0}]]
    create_generated_clock -name hbm_aclk -source $source -master_clock $master \
        -edges {1 2 3} -edge_shift $shift $sink
    set current [get_clocks -quiet hbm_aclk]
    if {[llength $current] != 1 || abs([get_property PERIOD $current]-$period) > 0.001 ||
        [get_property MASTER_CLOCK $current] ne "io_clk_freerun_00_clk_p"} {error "recreated clock identity differs"}
    report_timing_summary -delay_type min_max -max_paths 20 -file $out/${label}_timing.rpt
    report_clock_interaction -file $out/${label}_interaction.rpt
    puts "CLOCK_SENSITIVITY $label [get_property PERIOD $current]"
}
puts "Exploratory clock sensitivity complete; no closure is asserted"
