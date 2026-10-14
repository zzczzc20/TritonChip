# Thread-safe device process launch

Source: parent `90195c8` plus the child-launch patch in this commit. Functional
tests ran in `~/tritonchip-r40-childexec` on the ARM jump host using the locked
Torch/Triton environment and Icarus 12. Local lifecycle tests use Python 3.

The file transport and U50 simulated register link previously invoked ctypes
from a Python `preexec_fn`, despite the runtime and Torch using threads.
They now exec a small standalone launcher which installs PR_SET_PDEATHSIG,
checks that the owning parent still exists, and execs the target with unchanged
arguments. The device keeps its PID and its separate process session. There
is no Python callback between fork and exec in the threaded owner.

Retained checks:

- `lifecycle.log`: 4/4; literal arguments/exit status, 32 starts from eight
  threads, changed parent before startup, child termination after owner SIGKILL.
- `backend.log`: 3/3; real file/socket bindings execute the same artifact with
  correct output and the same 204 device cycles; backend boundary checks pass.
- `recovery.log`: 3/3 forced-address-reuse epochs; stale storage/events refused
  and old destructors preserve new allocations.
- `fpga_wrapper.log`: R36-FPGA-01/02 pass, including synthesis checks.
- `u50_link.log`: real register-link harness opens, answers a register read,
  and closes through the updated launcher.

An initial backend scan objected to the word `Popen` in the new helper's comment;
the comment was clarified and the complete backend test rerun passed. The helper
contains no binding-specific commands or knowledge of RTL.

This removes an unsafe launch mechanism; it does **not** establish the cause of
the historical capacity-child failure or instrumented host SIGSEGV. Those
records remain in R40-04. No residency/performance/robustness gate is promoted by
these checks; the complete HBM regression and board acceptance remain required.
