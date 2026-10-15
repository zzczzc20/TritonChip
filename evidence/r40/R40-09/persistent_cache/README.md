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
before its second selection; its current three-core RTL run is still pending.

Concurrent searches can repeat measurement. This cache does not implement
cross-process candidate admission or kill over-budget workers. Automatic
compile candidates, the cost model, frozen holdout, amortization and board
performance remain required for R40-09 acceptance.
