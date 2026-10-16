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

## Located cause and candidate repair

`diagnostic_replays_27_29.json.gz` retains the full strict replay reports and
short cycle traces. `root_cause_prefix.json` compares the original and candidate
frames. At cycle 332423, context 2 exits while its younger instruction contains
unspecified opcode bits. The decoder correctly marks that instruction illegal
and clears its write enables, but retains raw opcode classification. The core
also selects functional units from those classification bits, so the killed
instruction makes the shared stall unknown. Another context is being fetched;
fetch-valid, D-valid, R-valid and finally the writeback reservation become X.

The candidate assigns inert SYS/NOP classification and zero modifier bits only
when the decoder has already determined the instruction is illegal. The illegal
flag remains asserted, so a non-squashed illegal instruction still traps. This
is not a blanket unknown-to-zero conversion. On the same recorded input times,
the candidate keeps the critical stall and valid signals known and completes
the originally failing batch. The unchanged replay oracle then refuses the
partial capture: the original recording has no successful response or later
inputs for that batch. This expected rejection is **not** relabeled replay-pass
or a complete application result.

`decoder_core_regression.json.gz` retains decoder legality checks, the old-source
mutation which fails the new unknown/illegal-classification checks, and complete
`make -j3 core core-vrf core-slowmem` logs. These include illegal-trap isolation
and descriptor faults. The first standalone decoder driver incorrectly selected
SystemVerilog for this Verilog-2005 testbench (`expect` is a reserved keyword);
its failure is preserved, and the corrected driver uses the Makefile's `-g2005`.

Current blocker: full native four-configuration numerical validation of the
repair, with recording enabled and disabled, is still running. Next commands
on the jump host:

```sh
cat /tmp/r40_illegal30_native.exit /tmp/r40_illegal32_default.exit
tail -12 /tmp/r40_illegal30_native.log
tail -12 /tmp/r40_illegal32_default.log
```

Success requires all four numerical/outlet/staging checks and exit 0 in each
run; retain the complete logs, source hashes and recorded journal. Failure
retains the first assertion or numerical discrepancy and its input history
before further changes. Neither outcome substitutes for the full native 77,
HBM 14 or frozen model on the final immutable software/image identity.

The active FPGA implementation still uses cf330dc and does not contain the
later decoder repair. Preserve that distinction when evaluating the image;
these simulator results cannot qualify that hardware or spend another full
implementation attempt. Snapshot 09 lacks a complete timed history, so no
claim of exact replay of that older failure is made.

See the [usage and uncertainty guidance](../../../../../tools/performance_harness/mmio_journal.md).
