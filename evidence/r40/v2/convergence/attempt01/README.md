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
`tail -60 /home/zzhouch/tnca_build/r40_attempt01_20261016/xo/link.log` on the
FPGA host, followed by the vendor run log named there. Success requires actual
placement, full route, 100 MHz setup/hold, CDC and a loadable image before the
unscaled board checkpoint. Failure preserves raw reports/checkpoints and uses
the final remaining attempt only for a specifically diagnosed correction.
All required gate statuses and performance values remain unchanged/not-measured.

## Completed local screen and launch observation

Actual TMU plus full-line-tag slice, with identical ideal 10 ns OOC constraints:
registered-range 64 sets used 78,543 LUTs and 33,818 FF; 32 sets used 49,905 LUTs
and 19,419 FF. Estimated WNS changed from -6.037 to -5.811 ns, still failing.
This is evidence for reduced resources, not 100 MHz qualification or a prediction
that routing will succeed. `geometry_screen.tar.gz` preserves the exact sources,
script, log and raw reports. `geometry_functional.tar.gz` preserves both
replacement-policy random differential runs; its explicit limitations are in
`geometry_functional.json`.

Detached runner PID 894704 started at 2026-10-16 03:18:04 UTC. Vitis link started
at 03:18:36 UTC. The packaged XO SHA256 is
`ae9b958a7e54c1744b76ea3f2aa6f4cad8675b55e2aff29f4f1d3f84d110e9e8`.
XO is an intermediate, not a loadable image. The immutable source archive SHA256
is `9050487dbff7675c859e3401849ee023137c4607f3ccb17654585def72c20869`;
source/configuration identities are in `input_identity.json` and `launch.json`.

Main kernel synthesis was confirmed running at 03:24:31 UTC (process 900339,
run `ulp_tnca_u50_1_0_synth_1`); the vendor launched synthesis at 03:23:25 UTC.
`synthesis_started.json` preserves the live kernel log and parent launch log.
The process has a detached runner and persistent stdout/stderr; session expiry
will not stop the build. There is no automatic second attempt. Required gate
inventory remains five capability passes and 31 not-measured entries.

## Implementation progress observed at 06:36 UTC

The complete Vitis synthesis step finished and placement started; route and
qualification are still pending. `kernel_utilization_synth.rpt` is the actual
kernel synthesis report: 867,764 LUTs, 273,821 FF, 214.5 BRAM tiles and 531 DSPs.
`post_synthesis_observation.json` preserves the linked flow and placement log
snapshot. Earlier OOC and placed resource reports use different stages and
constraints; they are diagnostic context, not a controlled percentage saving.
The observed SLR0/1 crossing demand is 15,396 of 23,040 SLLs. That preliminary
placement message cannot establish final local congestion or routability.
No extra Vivado report process was started while placement consumes memory.
