# Free kernel synthesis screen; no platform implementation or board access.
# Arguments match synth_r40_multicore.tcl: ROOT CONFIG OUT.
set_param general.maxThreads 4
source [file join [file dirname [info script]] synth_r40_multicore.tcl]
