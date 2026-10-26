# R40-09 compile-switch identity checkpoint — not accepted

AOT resolves eight allocation/lowering switches before cache lookup:
bank_alloc, window_reuse, packed_window_unpack, index_opt,
sparse_tensormaps, schedule, rctx_spmd and rctx_roles. Explicit zero overrides
a nonzero environment. Off remains absent from the default artifact options;
role partition modes 0/1/2 have distinct keys, including forced mode 2.
Invalid values are rejected before compilation.

The JIT forwards its resolved selection even when explicitly disabled under
an enabled environment. AOT tests pass 4/4; captured JIT-to-AOT option and
hash checks pass 36/36. Default artifacts remain byte-identical, 447/447.
The rctx_spmd_shared/retry policy and full runtime/RTL/clock/memory-map
configuration identity still need separate completion. This checkpoint is
not an autotuner, budget/holdout measurement or R40-09 acceptance.
