set_param general.maxThreads 6
set root $::env(HOME)/r41run2/recovery_postroute_20261029
set input $::env(HOME)/r41run2/build_1f0a5aff/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
open_checkpoint $input
report_timing_summary -delay_type min_max -max_paths 10 -file $root/before_timing.rpt
report_route_status -file $root/before_route.rpt
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
report_timing_summary -delay_type min_max -max_paths 20 -file $root/after_timing.rpt
report_route_status -file $root/after_route.rpt
report_cdc -details -file $root/after_cdc.rpt
write_checkpoint -force $root/recovered_routed.dcp
exit
