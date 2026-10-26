# HBM bandwidth status

Updated: 2026-10-26 UTC.

## Goal and current evidence

The target remains a new FPGA image with at least 100 GB/s effective streaming bandwidth, passing 100 MHz timing and measured on the shared U280. No candidate image or target measurement exists yet. The preceding goal turn made progress by recording and pushing the initial capacity audit. The current checkpoint adds bounded-admission controller results and an RTL issue-rate experiment, with synthesis evidence redirecting implementation toward compact burst engines. The target remains unachieved.

The source audit and existing R40 board/model evidence establish an upstream bottleneck. Three 64-bit core request paths admit at most 2.4 GB/s at 100 MHz, before protocol overhead. Three 256-bit outlets provide 9.6 GB/s in each AXI data direction. The prior ports experiment measured no gain from one to three outlets. Its model attributed contention to the home engine and core dependency waits; this attribution is not a new board measurement.

## Live external state

Read-only FPGA-host inspection at 2026-10-26 06:18:28 UTC found the Stage 2 endurance job live (experiment PID 2593930, endurance PID 2593937). KDS reported one context and client PID 2594359 on image UUID 49964b17-e0b4-5bd2-dd40-7fc37b1205d3. No card access or programming was attempted. Available host memory was 35 GB, below the 38 GB full-link threshold. No HBM-task full run has started; the handoff records one of five total runs consumed.

## Engineering direction

See `capacity_analysis.md` for quantitative elimination of inadequate options. The first implementation candidate must use independent stream lanes and bypass the serialized word/home path for supported contiguous transfers, while retaining the existing path for all other operations. Selection remains provisional pending native execution integration, semantic review and area screens. The controller service experiment in `model/README.md` now rules out eight-line windows and the unchanged two-cycle issue path for a 24-lane design. A disconnected traffic generator does not satisfy the goal.

## Next work and gates

1. Completed the first controller-service sweep: 54 bounded-admission cases and ten unchanged bridge regressions pass. At 16,384 lines, 32-line windows and one-cycle issue give 6.045 GB/s read and 6.063 GB/s write per PC. Ideal 24-PC scaling is 145 GB/s; this is not a chip prediction. Pipelined outlet issue now passes 78 functional configurations, eight sustained stream cases and two three-core execution configurations. Free 32/64-slot OOC routed screens are live on the FPGA host, PIDs 2597251/2597252, in `~/r41_hbmbw_outlet_v2/`. The 32-slot synthesis costs 17,514 LUTs per lane, ruling out straightforward 24-way replication under the existing image area/packing. Next implement a compact ordered burst stream path with BRAM buffering; see `outlet/README.md`.
2. Define how compiler-generated contiguous copy, write and read-reduce commands enter the stream path and maintain cache ownership, ordering, masks, exceptions and instruction numeric semantics. Do not claim generic acceleration from a microbenchmark-only engine.
3. Implement and regress the chosen path; obtain unit and tile synthesis/place-and-route screens before seeking a full-run allocation from the lead.
4. Obtain explicit new-image loading approval through the lead and respect the existing board queues and 15-minute idle rule. Run manifest-bound streaming and original-suite measurements, then regenerate the roofline.

No permission is needed to continue analysis, implementation or free screens. Full-run and programming approvals are later gates required by the handoff.
