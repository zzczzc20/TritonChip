# User-authorized conditional operating-frequency acceptance

On 2026-10-18 the user explicitly authorized preserving the attempt03 image
and using 91 MHz to conditionally close as many R40 gates as possible. After
functional/workload acceptance, frequency optimization should be attempted and
the achieved frequency reported without substitution or rounding into a pass.

This authorization is supplemental evidence. It does not edit the frozen
R40v2 contract or turn the original 100 MHz gate into a pass. Each result must
state whether it is contract-qualified, conditionally qualified under this
authorization, failed, or not measured. Historical results retain their original
image, clock and software identities.

## Execution order

1. Preserve the original routed checkpoint, generated bitstream and complete
   xclbin with hashes after the existing implementation terminates. Retain the
   original link outcome even if its 100 MHz clock guard rejects a downclocked
   image. Do not rerun implementation or overwrite these artifacts.
2. Audit the unchanged routed netlist at the proposed operating clock, including
   all-clock setup/hold, unconstrained paths, CDC, route legality and DRC.
   `audit_r40_operating_clock.tcl` performs analysis only. Changed constraints
   alone do not change physical clock frequency or establish board qualification.
3. Read the packaged clock topology and verify the actual data-clock mapping.
   Preserve any separately packaged operating-clock derivative and verify the
   embedded BITSTREAM identity. Measure the loaded clock using the shell counter;
   metadata and requested frequency are insufficient. If exact 91 MHz fails
   timing or cannot be programmed, record that result and establish a safe actual
   operating point before workload testing. Never round a negative margin to zero.
4. On the qualified operating image and immutable software, run the original
   unscaled >=64 MiB Torch checkpoint with all three physical cores and outlets,
   actual multi-PC HBM allocation, ownership and drained visibility. Preserve
   first-failure evidence. Success unlocks native77, HBM14, frozen models and
   subsequent required experiments; it does not substitute for them.
5. Preserve this usable baseline while diagnosing frequency limitations. Any
   additional full implementation still consumes the one remaining budget and
   requires a source-bound root-cause correction and matched validation first.

Current blocker: the original implementation has not yet produced a terminal
marker and packaged xclbin at the latest observation. The routed 100 MHz report
has WNS -1.002 ns, TNS -6679.026 ns and clean hold. A 91 MHz pass is not inferred
from these numbers. Next command: inspect `build/attempt.exit`, image clock
metadata and the existing observer state. Terminal success advances to the
unchanged-netlist clock audit. Failure preserves all generated artifacts and
checks whether the usable bitstream can be packaged without implementation.

## Prepared conditional checkpoint interface

`board_checkpoint.py --operating-clock-evidence REVIEWED_CLOCK.json` uses the
reviewed, image-bound measured frequency instead of its historical 100 MHz
identity. It retains `gate_status=not-measured` and explicitly labels the
conditional acceptance basis; a numerical checkpoint pass does not promote a
gate automatically. The original code/data graph and all traffic, memory-map
and visibility assertions remain intact.

The `tnca-qualified-operating-clock/1` file must contain the exact image SHA,
positive integer `clock_hz` no greater than 91,000,000, qualification
`conditional-r40v2-20261018`, finite nonnegative `setup_wns_ns` and `hold_whs_ns`,
`route_legal=true`, `cdc_review=passed`, and
`clock_source=measured-board-counter`. `evidence_files` must bind the reviewed
`timing`, `route`, `cdc`, and `board_clock` files by path and SHA256. This is a
reviewed qualification record, not an automatically inferred pass from report
existence. It cannot be created honestly until the reports and board clock
measurement exist. Evidence is rechecked before sealing checkpoint success.

Current running preservation/audit command and PID are retained in
`conditional_91mhz_launch.json`. Its state is
`~/tnca_build/r40_attempt03_conditional_91mhz/state.json`. It waits for both the
original implementation and its serial report observer before opening the
checkpoint. No extra implementation or board load has been started.
