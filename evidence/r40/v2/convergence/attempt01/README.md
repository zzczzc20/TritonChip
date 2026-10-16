# Additional implementation attempt 1

Candidate hypothesis: registered unsigned range endpoints and balanced owner
selection reduce the diagnosed metadata cone; reducing cache sets from 64 to
32, retaining eight ways, reduces asynchronous tag selection/storage and local
packing pressure. The failed placed design used 97.59%/96.54%/86.54% of CLBs.
No tag-bank pipeline is introduced. Three physical cores/outlets, shared
HBM[0:31], 100 MHz, two-stage links, ownership and code-queue preservation remain
explicit. Runtime memory controls provide same-image port/outstanding ablations.
Every final comparison must use this same cache geometry and full frozen shapes.

`image.json` and `image.tcl` are the exact package parameters. The launch command
is `bash flow/fpga/run_r40_attempt01.sh /home/zzhouch/tnca_build/r40_attempt01_20261016`.
The runner packages once and starts Vitis link once; it never retries or loads
the board. `SSI_SpreadSLLs` is retained from the failed run: this is an RTL and
geometry hypothesis, not a directive search.

Known risk: runtime-control snapshot 09 exposed a reservation-X assertion;
snapshot 11 passed but did not explain that failure. The user requested starting
the first bounded attempt before the remaining session allowance expires.
Physical build proceeds concurrently with this functional investigation, not as
an assertion that candidate qualification is complete. The earlier 64-set OOC
screen still misses 100 MHz; smaller geometry cannot be assumed to fix timing.

Current blocker: no completed candidate placement/route evidence. Next command:
`tail -40 /home/zzhouch/tnca_build/r40_attempt01_20261016/build/runner.log` on the
FPGA host, followed by the vendor run log named there. Success requires actual
placement, full route, 100 MHz setup/hold, CDC and a loadable image before the
unscaled board checkpoint. Failure preserves raw reports/checkpoints and uses
the final remaining attempt only for a specifically diagnosed correction.
All required gate statuses and performance values remain unchanged/not-measured.
