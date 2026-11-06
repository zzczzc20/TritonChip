# Free OOC kernel screen. The same arguments and image configuration are
# used by synth_r40_multicore.tcl; bound threads for the shared host.
set_param general.maxThreads 4
source [file join [file dirname [info script]] synth_r40_multicore.tcl]
