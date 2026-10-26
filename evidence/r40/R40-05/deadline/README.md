# Global deadline stop-and-drain

`cmd_deadline` is a latched 64-bit global launch budget in device cycles;
zero disables it. Expiry requests dispatcher abort, stops new reservations
and acceptance immediately, and cancels unaccepted queue entries. Accepted
programs continue to drain. The normal completion record returns status 2
only after the accepted programs have completed, with the original launch ID
and matching issued/completed counts. Timeout remains visible on `timed_out`
until the next valid grid starts. Global abort does not reset cores or fabricate
memory completion. A simultaneous attributable core fault takes precedence.

An infinitely running accepted program cannot be declared drained. In that
case timed_out stays asserted and host/launch admission stays closed; platform
recovery must explicitly quiesce/reset and advance its epoch. The public runtime
recovery binding is not implemented by this checkpoint.

The standalone controller checks the exact boundary, latched budget, one-shot
abort, persistent diagnostic, disabled budget and reset. Dispatcher tests cover
all 1/2/3/4/8 core counts with global abort under mixed latency/backpressure,
queue cancellation, drain and relaunch. The real persistent shell test adds a
20-cycle deadline, verifies partial issue and full drain, then executes both
normal grids and checks all output words. This is RTL functional evidence,
not complete R40-05 or board acceptance.

The final 1/3/8-core shell runner passed, including timeout, subsequent normal
grids and all result downloads. Coherent-host regression and all four legacy
2x2 system cases passed. The standalone deadline target also passed.
