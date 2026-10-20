# Frozen graph metric and raw device clock

The frozen `graph-makespan` metric now accepts the producer's
`device-graph-makespan` entry only with complete raw launch endpoints and
cycle units. Every sample is recomputed, including launch count, clock
domain, reset epoch and the complete outer interval. Explicit active-cycle
sums are rejected for secondary makespan entries as well as primary timing.

The pinned ARM environment passed the device-clock and protocol checks. The
existing original tiny-add RTL record reproduced its unchanged 2,318 and
2,317 cycle spans through both metric names, without another device run.
The initial standalone replay command lacked `tools/r40` in `PYTHONPATH`;
its failed import and the successful corrected command are both retained.

This changes post-measurement interpretation only. It adds no target
instructions, register reads or work inside the measured call. Virtual RTL
cycles do not establish physical timing or a performance gate. No frozen
contract or point definition changed. See `report.json` and `raw.tar.gz`.
