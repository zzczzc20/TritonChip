# Called with an already-open routed checkpoint and output directory in $out.
# The trace is evidence for review; it is not a CDC disposition.
set irq_src {level0_i/blp/blp_i/PLP/ss_fic/inst/user_fanout_firewall_and_regslices/axi_firewall_ctrl_user/inst/gen_mi.mi_r_error_i_reg}
set irq_dst {level0_i/blp/blp_i/PLP/ss_dma/inst/axi_intc_aggregated_mgmt/U0/INTC_CORE_I/INTR_DETECT_GEN[0].ASYNC_GEN.intr_ff_reg[0]}
set irq_trace [open [file join $out platform_irq_trace.txt] w]
foreach {label target} [list source $irq_src/C destination $irq_dst/D] {
    set pin [get_pins -quiet $target]
    puts $irq_trace "$label pin_count [llength $pin]"
    if {[llength $pin] != 1} {close $irq_trace; error "$label pin ambiguous"}
    puts $irq_trace "$label pin $pin"
    puts $irq_trace "$label cell [get_cells -of_objects $pin]"
    puts $irq_trace "$label clocks [get_clocks -quiet -of_objects $pin]"
    puts $irq_trace "$label properties [report_property -return_string [get_cells -of_objects $pin]]"
}
puts $irq_trace "destination fanin [all_fanin -flat -startpoints_only -to [get_pins $irq_dst/D]]"
puts $irq_trace "destination fanout [all_fanout -flat -endpoints_only -from [get_pins $irq_dst/D]]"
close $irq_trace
set irq_logic [open [file join $out platform_irq_logic.txt] w]
foreach {label target} [list sourceD $irq_src/D destinationD $irq_dst/D] {
    set pin [get_pins -quiet $target]
    if {[llength $pin] != 1} {close $irq_logic; error "$label pin ambiguous"}
    set pins [all_fanin -flat -to $pin]
    puts $irq_logic "\n$label fanin_pin_count [llength $pins]"
    foreach c [lsort -unique [get_cells -of_objects $pins]] {
        puts $irq_logic "cell $c ref [get_property REF_NAME $c]"
        if {[string match LUT* [get_property REF_NAME $c]]} {puts $irq_logic "INIT [get_property INIT $c]"}
        foreach p [get_pins -of_objects $c] {
            set net [get_nets -quiet -of_objects $p]
            if {[llength $net]} {puts $irq_logic "pin $p net $net"}
        }
    }
}
close $irq_logic
