# Public Inductor tuning: functional evidence, physical validation pending

`TNCA_TUNE=1` now has a consumer in the common `CachingAutotuner.run` path for native TNCA tensors.
With the switch off, the first legal compiler configuration remains in use.
Pointwise candidates are generic block retiles, including conservative 16-lane and smaller legal blocks;
neither workload IDs nor model names participate in generation or selection.

The parent compiles the baseline before opening a graph tuning budget and records its compile cost separately.
Alternative compilation runs in device-free processes from the generated Inductor source, with the parent's
resolved physical-chunk contract. Workers cannot open a Session and never receive tensor handles.
The existing generic supervisor owns their process group and original graph deadline.
`TNCA_TUNE_SECONDS` defaults to 600 and cannot exceed it; `TNCA_TUNE_CANDIDATES` defaults to 4 and cannot
exceed 16 per kernel. The shared graph budget includes alternative compilation and device measurement.

Measurements use resident tensors in the original Session. Native warmup and measurement logs are isolated
from caller logs. A measurement scope requires idle device work and a backend that snapshots live launch
deadlines; it reserves time for drain, restores limits, and refuses concurrent submissions from another thread.
A host wait expiry is never treated as cancellation. Undrainable work poisons the Session; its epoch and handles
are not silently changed. Device loss propagates so the caller can explicitly recover and recreate invalid state.

Only qualified physical common-clock endpoints rank production candidates. The selector requires the actual
loaded image file identity, reviewed image-bound operating clock, enabled device endpoints and actual memory map.
SystemC's uncalibrated timing domain retains the baseline. Missing metadata, capture, mutated arguments,
aliased storages, unsupported descriptor metadata and insufficient budget also retain conservative execution.
Descriptors remain enabled in compilation; descriptor-argument tuning is currently refused.

The persistent selection key includes generated source, concrete shapes/strides/dtypes/scalars and placements,
product/compiler source identity, profile, capabilities, image/RTL identity, actual memory mapping, clock proof
and every candidate artifact hash. Upstream's winner cache is bypassed for TNCA tuning because it lacks these
hardware identities. Formal execution uses only the selected standard TNCA launcher.

## Evidence

`kickoff/public-tuning-v2-evidence.tgz` contains a source-fingerprinted experiment with four completed steps:

- Session limits, delayed completion drain, undrainable-work poison and refusal checks;
- a real generated 8-lane candidate compiled in a worker and executed through the common launcher;
  an intentionally hung compiler with an actual child process was terminated by its original deadline;
- the public Torch/Inductor hook on SystemC retained conservative execution without spawning tuning work;
- the public hook on the three-core MMIO RTL model measured two real configurations using device endpoints,
  selected XBLOCK16 (237 cycles) over XBLOCK8 (322 cycles), isolated its samples, and preserved numerical output.

The RTL test explicitly injects a functional RTL identity in place of physical image qualification.
Its nominal frequency is a timing normalizer, not a measured board clock. It does not establish board performance.
The incremental tuning time in that functional test was 11.73 seconds; baseline compile was separately recorded.
The first attempted test's cold-log assertion incorrectly counted standard compilation helper copies as tuning
samples. The corrected assertion identifies the actual selected kernel, while the formal log must contain exactly
one launch. The original failure is preserved remotely.

`kickoff/tuning-refusals-v1.log` adds alias-view, mutation, capture, unknown-descriptor and disabled-clock negatives.
`measurement-scope-tests.log`, `autotune-workers-after-compile.log` and `autotune-policy-after-compile.log`
retain the four scope, six worker and fourteen policy/cache checks.
The fresh staged-source AOT regression in `aot-after-measurement-scope-fresh.log` passes all nine cases.

The complete source of the public campaign precedes the final extra metadata-refusal guards and the
argument-compatible cache wrapper; the additional refusal and AOT checks cover those edits.
An actual public-path persistent-cache hit, complete cross-device boundary regression and physical-image
selection/drain/error qualification remain to be completed before S1-03 can close.
