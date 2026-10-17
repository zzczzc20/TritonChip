# R40-01 (in progress): the tnca device, 2026-10-10

`gates.json` holds the evidence the gate tools wrote. `tools/r40/acceptance.py --wp R40-01` gives:

| Gate | Status | Why |
|---|---|---|
| R40-01.mixed-device | capability-pass | `tools/r40/device_tests.py`, 16/16 on a host with CUDA (GB10, simulator) and 16/16 on one without (the board host, U280): 32/32 checks |
| R40-01.device | not-measured | 68 of the 77 tiny/small/medium points run on the device and pass, with no host fallback. 9 are capacity-exceeded: their tensors do not fit resident in the B39 image's 384 KiB on-chip heap |
| R40-01.device-time | not-measured | The frozen comparison needs all 77 (`ablate.py` refuses a frozen point that did not run). The diagnostic over the 68 is S_geo 1.0015 against B39, 52 cycle-identical, worst 0.983 (resnet_block tiny, explained in `doc/r40_tnca_device.md`) |

The two `not-measured` gates are signed off after R40-04 gives the device
external memory, as the R40 contract foresees (R40-01 acceptance 1: "HBM
就绪前先跑容量可容纳的 smoke，完整签收在 R40-04 后补齐").

The board run: u280_r39h, software of this commit's parent tree,
`archbench.run --set TNCA_TORCH_DEVICE=native --warmup 5 --timed 10`. The
records are under `sim/build/r40/native1/`. On tiny, request latency is
3.59× B39's CPU path (geometric mean, range 1.66–17×), because no tensor
crosses the host interface per launch.

R40 resource-property regression checkpoint: reporting TNCA's truthful zero
shared-memory capacity exposed Triton's ordinal-only cached limit. After a
TNCA:0 kernel load, CUDA:0 incorrectly inherited zero and refused a CUDA
kernel requiring 1152 B. The locked-version compatibility hook now queries
the active backend at kernel load. Standard native graph, alternating
CUDA/TNCA and resource-property checks pass 3/3; CUDA receives its actual
limit, and TNCA remains zero. The full device regression is being repeated.
See mixed_backend_cache.log. No complete native/performance gate is claimed.

Native compile-capacity checkpoint (2026-10-11): compile-only now uses the
active Session heap rather than the CPU migration staging limit. With the
32 MiB RTL HBM profile, all 77 frozen tiny/small/medium points compile; the
records retain `not-measured`, null correctness/timing and no executed
kernels. This does not close either device gate.

The unchanged large LayerNorm point needs 4,202,496 argument bytes. Native
HBM compile succeeds against the reported 33,554,432-byte heap; the CPU
migration control still reports `capacity-exceeded` against 393,216 bytes.
See native_large_compile_capacity.*, cpu_large_compile_capacity.* and
native_hbm_compile77.*. Shapes, frozen thresholds and CPU limits are unchanged.

2026-10-11 full regression attempt: CUDA hidden passes 16/16. CUDA visible
passes 14/16; import_orders and mixed_devices fail with CUDA context
allocation errors. A standalone one-element CUDA allocation also fails
with cuDevicePrimaryCtxRetain returning CUDA_ERROR_OUT_OF_MEMORY.
At that checkpoint the CUDA-visible rerun awaited a working context; the
logs retain both failures. Child diagnostics now retain bounded stderr
and stdout so the underlying exception is visible instead of only the
last Dynamo hint. No failed case is changed into a skip.

2026-10-12: the full CUDA-visible native device regression passes 16/16
after preparing the current RTL cache. This includes all three import
orders, CUDA/TNCA alternation in one process, worker registration,
numerical compiled graphs and view/storage/copy behavior. See
native_device_cuda_20261012.log. Prior CUDA-context allocation failures
remain recorded; they are not converted to skips. Full HBM and physical
performance gates remain open.
