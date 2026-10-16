# Timed MMIO/DMA capture and replay

Use this workflow when a multicore RTL failure changes or disappears after
adding probes. It records the mailbox inputs at the simulated times when the
board-top stand-in consumes them, every completed response, and the terminal
simulation time. Replay removes live host delivery variability for that one
captured experiment. It does not establish that other schedules are correct.

Enable `journal: true` in the `fpga-multicore-sim` backend configuration together
with `board_top: true`. Recording is disabled by default. The following bounded
functional drivers also accept the option:

```sh
python3 tools/r40/tmu_control_rtl_tests.py --journal --out NEW_CAPTURE_OUTPUT
python3 tools/r40/native_memory_control_tests.py --journal
```

The TMU driver reports `journal_directory` in its report. The native driver
prints `MMIO_JOURNAL` after its warm-up graph; an earlier failing run still leaves
its journal under `sim/build/u50-sim/link-*/journal`. Preserve the original host
log and its exit status as well as the journal, `regs.log`, compiled harness,
and immutable source snapshot. Metadata records the harness path and SHA256.
A successful DUT replay does not certify the original host/reference checks.

Run replay with the matching Icarus runtime on PATH:

```sh
python3 tools/r40/replay_mmio_journal.py \
  --journal CAPTURE_DIRECTORY --out NEW_REPLAY_OUTPUT --cap-seconds 1800
```

A relocated harness requires `--harness COPY.vvp` and the captured SHA256 must
match. An intentionally modified diagnostic harness requires
`--diagnostic-harness NEW.vvp`; the report records the override and new hash.
Keep the original harness and first replay before recompiling diagnostic probes.
Use `--capture-log FILE` if the original simulator log was relocated.

Replay checks every response byte and each batch completion time. It retains
the idle interval after the final batch: a DUT assertion can occur there.
The terminal record distinguishes graceful closure from abnormal termination.
A live or truncated capture without that record is refused. Graceful simulator
closure does not imply that the application's reference checks passed.
A captured abnormal stop always produces a nonzero replay verdict; compare the
original and replay assertion, simulated time, core, context, PC, and reservation
state to determine whether the same failure was reproduced. An expected failure
is never labeled an acceptance pass. Retain mismatching responses, shifted
completion times, and missing-terminal mutations alongside successful replay.

Journal file operations introduce no explicit simulated delay, but their real
wall-clock cost changes when the host supplies subsequent work. Thus capture
can expose or hide a schedule-sensitive failure. Replay preserves the resulting
schedule rather than reconstructing an unrecorded earlier failure. Capturing
also adds graceful-stop waiting, bounded to ten seconds, at simulator closure.
Wall-time overhead and its uncertainty remain **not-measured** until a matched
experiment characterizes them. Do not rank candidates by capture/replay wall
time or use these runs for board bandwidth, frequency, or speedup claims.
