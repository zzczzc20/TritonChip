# Private resource rebalancing study

## Fixed-budget observations

`pipeline-model-rtl.tgz` contains actual native standard Torch mm/addmm
executions for K = 32, 48 and 64 under Mode B, single-role two-stage software
pipelining and Mode A two-stage pairing. `pipeline_closeout.py` verifies the
18 numeric rows, identical inputs and output bits, actual artifacts, role
mappings, model identity and limits. All use the same model binary/configuration,
three-core profile and a two-context role mask. These grid-one points use one
program and are not a three-core scaling study.

Each context has a 64-vector-register architectural limit. On Mode B each
program uses its own context's base; the other configured context is idle for
these grid-one points. RPAIR gives both roles of the Mode A program the same
64-register base, while two contexts are reserved for the pair. Thus every
compared program has one 64-register vector name space and the same two-context
launch budget. The unused physical context space is not silently reassigned.

The emitted resource plans show asymmetric work. For the first registered
paired case the producer has four static descriptor loads and no MMA or
accumulator reads; the consumer has two static MMAs, 16 epilogue accumulator
reads and no descriptor load. The precise counts for every case and both
roles are retained in `pipeline-summary.json`. They are static emitted commands,
not dynamic execution counts or a measured private-register demand per role.
The manifest reports a nine-register private peak for each paired role, but
this field does not prove an independent optimal allocation for each role.

For this shape the baseline has 16 window registers; both two-stage designs
have 32. Two eight-register operands in each of two simultaneously live slots
need 32 window registers before ordinary temporaries. The existing awaited
window-lifetime allocator already reuses disjoint windows when its structured
control-flow proof applies. The single-role odd-trip drain is conservatively
kept in fixed windows because that allocator rejects its extra branch; claiming
reuse across that branch without a control-flow lifetime proof would be unsafe.

## Static asymmetry decision

The candidate considered is a per-kernel unequal producer/consumer private VRF
split while retaining the existing shared transfer windows. The paired roles
currently share a vector base, and the compiler keeps their temporary names
apart; scalar/predicate files remain per context (`rtl/core/tnca_core.v`, RPAIR
and RSLICE comments). Merely changing an allocation constant cannot enlarge
one role's architectural 64-register index range or move unused physical
registers from its partner. Existing RPAIR aliasing does not prove a larger
register pool. The static per-kernel proposal is therefore **not selected**:
the measured representative kernels fit the existing limits, and no safe
capacity or performance improvement has been established.

A future implementation would need a separately specified capability,
allocator and ABI/RTL semantics, including per-role bounds, aliasing,
synchronization, fault handling, and physical timing/resource screening.
Dynamic register reallocation is not proposed. This is a bounded negative
Study decision, not a claim that asymmetric allocation could never help other
workloads. The physical board comparison and complete P2 pipeline gate remain
pending the lead's approved-image measurement.
