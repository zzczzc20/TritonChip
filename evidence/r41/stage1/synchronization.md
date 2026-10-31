# Synchronization audit

`tl.debug_barrier()` remains an acquire/release device fence in the common
frontend. Scheduling retains this memory effect between stores and reloads.
The core implements the fence by draining ordinary LSU requests and pending
writeback. It does not complete asynchronous transfers or rendezvous roles;
those operations require their actual mbarrier protocol. No synchronization
weakening is enabled by this audit.

`kickoff/barrier-ordering-v2-evidence.tgz` records the source-fingerprinted
remote experiment from `experiments/barrier-ordering.json`, its raw logs,
eight generated assembly/hex programs and the actual RTL bench hashes:

- Store/fence/permuted-reload produces the independent expected values with
  both identical and distinct input/output addresses on fast, slow and
  out-of-order response benches (six programs).
- The two-role, one-slot, two-iteration pipeline preserves all 32 expected
  words with proper release arrivals. Replacing consumer release arrivals
  with ordinary fences reaches the explicit 5,000-cycle timeout. This is an
  expected negative, not a passing execution of the broken protocol.
- Asynchronous reads and writes retain ownership across an ordinary fence.
  Code generation rejects premature access in both directions; waiting for
  the corresponding mbarrier makes those same cases legal.

These results establish compiler ordering and the exercised RTL protocols.
They do not establish physical-board behavior or justify removing fences
from other graphs. The final software still requires its original board
regression and related compiler/runtime checks before Stage 1 acceptance.
