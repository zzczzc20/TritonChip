# Persistent tuning cache checkpoint, not accepted

`PersistentCache(directory)` can be passed to the existing native device-event
selector. Records include a schema, search key and payload checksum. Writers
fsync a temporary file and atomically replace the per-search file; readers
discard damaged records. A reopened cache uses the same complete identity,
candidate and resource-capability search key. Cache hits recheck the stored
identity and the selected candidate's legality before skipping measurement.

`policy.log`: 12 tests pass, including reopen, damaged-record recovery,
image invalidation, thread and independent-process publication, and the
existing event/budget/resource policy. These synthetic records prove policy
behavior, not performance. The native integration test now reopens the cache
before its second selection. `native.log` records the completed three-core
RTL integration: unchanged standard Torch graph, candidates 1/3 cores,
8 cores refused before measurement, common device-event spans 3253/1900
cycles, reopened cache hit without remeasurement, and one separate formal
resident launch with zero staging. This is functional evidence, not holdout.

Concurrent searches can repeat measurement. This cache does not implement
cross-process candidate admission or kill over-budget workers. Automatic
compile candidates, the cost model, frozen holdout, amortization and board
performance remain required for R40-09 acceptance.
