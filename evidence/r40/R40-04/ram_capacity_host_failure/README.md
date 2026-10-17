# RAM-cache capacity host failure — unresolved

The complete barrier-corrected HBM regression passes 108/109 named checks;
RAM-cache capacity's Python child returns 1 with only its CUDA-runtime
warning surviving. The independent exact-child retry retains the same
exit and complete output. An instrumented retry confirms registration,
24 MiB pad allocation and input upload succeed. Its graph call waits in
the runtime worker's transport reply loop; a faulthandler run subsequently
exits by SIGSEGV (-11). All attempts remain preserved.

This is not a passing capacity/robustness result or a diagnosed RTL
arithmetic error. An exact uninstrumented child is now running with a
172800-second host observation budget; the original workload, numerical
checks and hardware cycle deadline are unchanged. The host-wait hypothesis
is not yet verified. hbm_tests.py now retains complete child scripts and
stdout/stderr for every attempt, including timeouts.
