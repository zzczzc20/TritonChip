# Range maintenance and all-home ownership barrier — functional checkpoint

Source: parent `47a486b` plus the cache-maintenance changes committed with this
evidence. ARM jump host, Icarus 12. `cache_maintenance_tests.py --global-grid`
passes 19/19; complete output is retained in `matrix.log`.

The optional NUCA slice maintenance path snapshots a byte range, drains its
accepted requests/MSHRs/responses, writes back dirty overlapping lines and
invalidates clean or dirty overlapping lines. Full-line writeback preserves
dirty bytes outside a partial DMA range. Unrelated cache lines remain valid.
The all-home controller snapshots the same range, admits each home once and
waits for all home completions **and** the platform's memory-drained signal.
`tnca_top` connects it through per-home elastic command/reply links, using
MEM_HOME_STAGES. Grid and coherent host admission are held during maintenance.
Default CACHE_MAINTENANCE=0 preserves legacy behavior.

Coverage:

- Six slice cases: full-line and legacy home-hash tags, memory latency 1/8/31,
  random ready backpressure/reordering, dirty/clean/zero ranges, partial-byte
  host DMA emulation, unchanged neighboring bytes, range snapshot, high address
  bits, range-end overflow, queued/stalled responses and reset before write
  acceptance. Memory requests must stay stable under backpressure.
- Three controller cases: 1/3/8 homes, staggered acceptance, reverse completion,
  exactly-once offers, changed input range, delayed physical drain and reset.
- Eight pressure cases: maintenance disabled/enabled-but-idle, policy 0/74,
  seeds 1/7, 20,000 requests each. Every returned value is checked by the
  existing differential bench: 160,000 requests total, including dirty eviction.
- Two real three-core global-grid cases: NoC/dispatch/home stages 0/2, two
  455-program grids, coherent host readback, dirty result writeback, all 455
  words checked in backing memory, partial-byte DMA emulation and subsequent
  coherent read. Grid/host cannot enter before the delayed drain signal.

The backing-memory updates and drain delay are explicit model stimuli; they
are not an XRT DMA measurement. The slice uses the existing fire-and-forget
writeback contract; only the global barrier may grant ownership after physical
write responses have drained. Neither a slice done pulse nor a global core
idle signal alone authorizes DMA.

Runtime bulk DMA remains refused. MMIO command/capability, shell/platform drain
and error propagation, XRT transfer hooks, board validation and independent
copy/compute overlap remain required. This checkpoint does not pass the full
R40-02/06 gate or change the currently routing immutable image snapshots.
