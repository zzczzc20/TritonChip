# Memory-service diagnostic and preserved residuals

`kickoff/memory-service-v1-evidence.tgz` retains the three-core standard Torch
graph `(a+b)*0.5`, 4097 FP32 elements, fixed seed 4006. All three outlet/depth
configurations pass independently on the native model and the actual board-top
MMIO RTL stand-in. Inputs, wrappers, artifacts, placements, control readbacks,
raw counters and model runtime files are preserved. The warm output stays
resident; numeric CPU checks occur after the measured counter window.

The device-cycle metric sums completed launch windows. ABI uploads and their
maintenance remain in the native call path, but prelaunch maintenance cycles
are outside launch timing; full-call counter deltas include its traffic.
Neither this metric nor its ratios are resident-call latency or board timing.

| Ports / outstanding | Historical coherent model | Explicit maintenance, B40 cache geometry | Explicit maintenance, RTL small cache geometry | MMIO RTL |
|---|---:|---:|---:|---:|
| 1 / 1 | 14,902 | 22,011 | 29,646 | 37,429 |
| 1 / 8 | 13,443 | 17,798 | 18,933 | 19,132 |
| 3 / 8 | 13,699 | 17,800 | 19,519 | 21,623 |

These are separate declared model domains. No old result was removed.
For the small-cache cohort, total model read/write lines are respectively
856/256, 859/258 and 854/256. RTL totals are 862/257, 862/257 and 860/257.
The raw reports remain authoritative for per-outlet distribution and exact
model residuals. More outlets do not win this diagnostic: RTL 3/8 is slower
than 1/8, while increasing outstanding depth from 1 to 8 helps substantially.

The source audit explains why this is not a physical-memory accuracy test:

- The stand-in defaults to `SETS=4, WAYS=2`; the B40 model profile uses 32/8.
- Its AXI test slave uses `LATENCY=25+p*19`, ID-dependent delay,
  `WRITE_DELAY=15`, randomized ready signals and interleaved returns.
- Its pipeline defaults, 10 ns simulation clock and simulated BO base differ
  from the B40 board profile and recorded physical mapping domain.

The cache-only profile declaration was saved before its executions, with the
RTL/test-slave/backend hashes and explicit unmatched domains. Only source-derived
sets/ways changed; no cycle parameter was fitted to these observations. The
explicit-maintenance cohort uses zero opaque host-fit costs. An independent
A/B identity audit verifies equal inputs, placements, artifacts, product source,
model binary, model config and map for coherent versus maintenance mode.

The comparator requires all six unique domain/configuration cases, matching
product source and actual inputs, positive completed cycles, correct backend
domains, unchanged shader within each domain, and one host policy per model
cohort. All three complete cohorts compare successfully; missing cases,
duplicate labels and mixed host policies are rejected. Two initial setup
failures are preserved: an invalid backend name and the wrong environment key
for backend arguments. Runtime capability guards were retained.

The model remains uncalibrated. These observations explain cache/ownership
effects and exercise real outlet admission; they do not qualify the synthetic
AXI slave as a physical HBM controller. Existing product defaults are retained.
Board measurement, physical prediction residuals and final Stage 1 closure
remain outstanding.
