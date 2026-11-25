# The placed design's congestion, attributed to cells, while the link goes on routing (a target's
# `post_place_tcl`; v++ runs it as the implementation run's PLACE_DESIGN.TCL.POST). It only reports: it
# changes nothing in the design. The report is written under a temporary name and renamed when whole, next
# to the run's log (impl_1/), so a reader never sees half of one; a failure is logged and written to
# tnca_placed_congestion.failed, and the link goes on as if the hook were not there.
set tnca_out [file join [pwd] tnca_placed_congestion.rpt]
if {[catch {
    report_design_analysis -congestion -file "$tnca_out.part"
    file rename -force "$tnca_out.part" $tnca_out
} tnca_err]} {
    puts "INFO: \[tnca-harness\] placed congestion report not written: $tnca_err"
    catch {
        set tnca_f [open [file join [pwd] tnca_placed_congestion.failed] w]
        puts $tnca_f $tnca_err
        close $tnca_f
    }
}
