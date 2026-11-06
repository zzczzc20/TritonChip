# Isolated probe only: source the production hook and record its expected
# refusal because this tiny kernel has no TNCA hierarchy. Never ship this hook.
set_param general.maxThreads 6
set original [file join [file dirname [info script]] place_r40_multicore.real.tcl]
set rc [catch {source $original} message options]
if {!$rc} {error "TINY_PROBE unexpected success: production placement requires TNCA hierarchy"}
if {![string match "R40 placement:*" $message]} {return -options $options $message}
puts "TINY_PROBE_EXPECTED_PLACEMENT_REFUSAL: $message"
puts "TINY_PROBE: production hierarchy placement is NOT validated"
