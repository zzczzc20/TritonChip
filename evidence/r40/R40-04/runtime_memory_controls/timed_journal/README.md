# Timed mailbox diagnostics: retained capture and replay

Required gates are unchanged. This package is functional diagnostic evidence;
board timing, bandwidth, instrumentation overhead and performance remain
not-measured. No production RTL or frozen contract was changed.

Snapshot 24 captured and successfully replayed 713 TMU transport mailbox batches,
including byte DMA through the link interface. Every response byte and batch
completion time matched. Its normal stop was retained and replayed. Response
bit mutation and completion-time mutation each fail at batch 1; removing the
terminal record is refused before simulation. The runtime-control RTL/host suite
and the TMU transport driver with journaling disabled also passed. The initial
default-driver invocation used the system Python without Triton and failed at
import; the corrected invocation uses `~/tc_env/torchenv/bin/python`. Both
outcomes are retained, and the import failure is not treated as a DUT failure.

Snapshot 25 captured the original 4,097-element resident native Torch graph
with the four outlet/outstanding configurations. The first three configurations
(3/8, 1/8, 1/1) passed numerical and outlet checks. Configuration 3/1 failed:
core 1, context 2, PC `0x1408`, cycle 332427, simulated time 3324375000 ps;
reservation, issue control and writeback delay contain unknown values. The host
exit was 1. The capture ends during batch 4921 with abnormal terminal record
`T 3324375 0`; the original DUT failure is retained rather than suppressed.
This independently captured failure resembles snapshot 09's symptom; it does
not by itself establish the same producing cause or repair either failure.

The exact snapshot-25 harness is retained. It precedes snapshot 24's trailing-
whitespace EOF-reader correction; that difference affects normal replay
completion only. Do not silently replace this harness during failure replay.
The failed source snapshot, original host and simulator logs, and the complete
journal are included. Preserve the original journal when adding diagnostic
probes: file I/O can alter live host delivery, while replay holds the captured
simulated input schedule fixed. A DUT replay excludes host/reference semantics.

Artifacts and SHA256:

- `failure_capture.tar.gz`: `f82846e6b27676e79ea207772d509540c035bdbeebdc2856b6afc788a1d8a7ed`.
- `validation.tar.gz`: `69a4c051fa17a751741554e3dbae1e6814f19e55fe49d2291992a8265e1b40ad`.
- `mutations.tar.gz`: `a2b272a74b5db47e79664ed12da320044f48cd3daa86750d9b23c5474b0483c9`.

The validation archive contains the successful capture and exact compiled
harness, raw replay and mutation reports, controls reports, and the tested
replay/Verilog sources. The mutations archive includes the modified journal
inputs and corrected-interpreter default-driver report. `observations.json`
retains process exit codes and bounded original logs.

Current blocker: the first source of unknown reservation/decode state is not
explained. The exact-harness failure replay completed with the identical first assertion,
including simulated time, cycle, core, context and PC. `failure_replay.json`
retains the raw result; `reproduction.json` checks the original and replay first
assertions verbatim. The replay exited 1, as a reproduced DUT failure must.
Its retained command on the jump host was:

```sh
source ~/tc_env/env.sh
cd /tmp/r40_mmio_journal24
python3 tools/r40/replay_mmio_journal.py \
  --journal /tmp/r40_mmio_journal25_native/sim/build/u50-sim/link-1his1__i/journal \
  --out /tmp/r40_mmio_journal25_failure_replay --cap-seconds 1800
```

The first monitor (snapshot 26) stopped earlier on an unknown decoded delay in
a killed instruction (`kill=1`, `books=0`, reservation still zero), at 430200 ns.
That is not evidence for the later reservation corruption: a killed instruction
does not book writeback. `first_unknown_launch.json` retains this observation.

The next experiment is launched in the separate snapshot
`/tmp/r40_mmio_journal27_first_unknown`. Its monitor still stops on unknown
reservation or valid state, but checks decoded delay only for an explicitly
valid, non-killed instruction. This filters inactive decode noise without
masking unknown architectural state or changing DUT behavior. The experiment
scripts retain both monitor versions and exact launch commands.
It compiles only an Icarus diagnostic harness and uses the explicit diagnostic
override on the same recorded inputs. Next observation command on the jump host:

```sh
cat /tmp/r40_mmio_journal27_build.exit
tail -40 /tmp/r40_mmio_journal27_first_unknown_replay.log
```

If the first unknown is located, retain the report and producing-state evidence
before proposing a correction and mutation. If compilation, replay timing or
responses diverge, retain the first diagnostic and correct the monitor/replay
boundary before changing DUT behavior. A timeout remains inconclusive. The
monitor's intentional fatal is not a passing application verdict. Neither branch
authorizes another full FPGA implementation. See the
[usage and uncertainty guidance](../../../../../tools/performance_harness/mmio_journal.md).
