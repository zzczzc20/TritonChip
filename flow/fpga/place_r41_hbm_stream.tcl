# Preserve the three-core placement contract, then bind the stream CU near
# HBM. Combined packing and routed timing must be screened before a full run.
source [file join [file dirname [info script]] place_r40_multicore.tcl]
set stream_banks [get_cells -hierarchical -quiet -filter {REF_NAME =~ tnca_stream_bank*}]
if {[llength $stream_banks] != 1} {
 error "R41 stream placement: expected one stream bank, found [llength $stream_banks]"
}
r40_assign $stream_banks 0
puts "R41_STREAM_PLACEMENT $stream_banks SLR0"
set stream_lanes [get_cells -hierarchical -quiet -filter {REF_NAME =~ tnca_stream_axi*}]
if {[llength $stream_lanes] != 32} {
 error "R41 stream placement: expected 32 stream lanes, found [llength $stream_lanes]"
}
