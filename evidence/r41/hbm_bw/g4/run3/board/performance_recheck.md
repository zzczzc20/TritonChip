# Registered performance recheck (pending)

The published preliminary comparisons selected all 39 memory-bound and two
original-suite points whose candidate median cycle count was over 5% higher.
The [registration](../../../../../../spec/r41/perf_recheck/registration.json)
records their identities, input hashes, order and fixed sampling before any
new measurements. Each arm collects three timed calls in a fresh process per
point. This is one recheck block, not an independent reproduction or a claim
that a statistical acceptance gate has passed.

Board-host PID 1146002 is queued after the clean-archive reproduction. It verifies
the predecessor and the exact manifests, image and three historical software
trees. It runs both candidate arms, then restores the authorized run-1 image
only after 15 consecutive minutes of recorded idle probes with the candidate still
loaded, using `program.sh` safety gates. It checks the run-1 clock before the
two old arms. A changed identity, failed stage or load refusal stops the queue
and preserves its evidence. The queue and logs are under
`/home/zzhouch/r41run3/perf_recheck_queue_v3`. The earlier unstarted waiters
were stopped to add idle-probe recording and then to insert independent
reproduction before the image switch. Their initial logs remain in
`perf_recheck_queue` and `perf_recheck_queue_v2`.

No recheck result exists yet. Comparing different complete stacks cannot
attribute the change to one RTL feature. Three calls within a process may not
provide enough independent information for a performance acceptance decision.
