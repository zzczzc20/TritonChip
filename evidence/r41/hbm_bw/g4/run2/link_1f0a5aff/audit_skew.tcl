set_param general.maxThreads 6
set root $::env(HOME)/r41run2/recovery_postroute_20261029
open_checkpoint $root/recovered_routed.dcp
report_bus_skew -file $root/after_bus_skew.rpt
check_timing -verbose -file $root/after_check_timing.rpt
exit
