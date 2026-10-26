# Complete pipeline target from an empty device cache

`TNCA_CACHE` points to a fresh `mktemp` directory. The target prepares v5 once
before any launch, then completes with exit 0: barrier boundary/exhaustion,
parser 4/4, options 18/18, 16 ring diagnostic cases across stages 2/3, and
eight standard native Torch cases with actual overlap 1272/1664 cycles.

Source/runtime/RTL: `3299288`; explicit preparation fix: `af6c169`. Jump host
root `~/tritonchip-r40-pipeline3`, locked Torch/Triton/Icarus. `run.log` and
`summary.json` preserve the cold-cache run. It verifies the CI setup fix,
not performance sign-off. The GitHub run must independently pass.
