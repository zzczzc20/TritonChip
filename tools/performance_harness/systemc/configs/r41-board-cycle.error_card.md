# R41 SystemC model, CA mode: what it is and how far to trust it

Model `configs/r41-board-cycle.cfg` (`core_model=1`, `uncore_model=1`,
`uncore_board=1`). This card is qualitative: the CA model has not been run
over the Event Model's calibration items, so it has no error statistics of
that kind yet (see `r41-sim.error_card.md` for the Event Model's).

## What it models

Cycle by cycle, from the RTL of the R41 three-core U280 machine
(`spec/devices/u280-3core-r41.json`, profile
`core-vrf-sync-v5-x8-g2c4-dp-board`):

- each core: fetch, decode, the R stage's hazards and the Role Context
  give-way, X, the writeback port and its arbitration, the banked vector
  register file's ports, the hybrid LSU (line path with 64 slots, word path,
  completion register), the line-wide tensor transfer engine (line loads and
  stores, the TensorMap load, transposed and 1-byte walks), the matrix engine
  (`m.mma.dp`, its five-cycle MAC pipeline, the registered port-3 response),
  the FPU, SFU and reductions;
- the uncore: the grid dispatcher (with its admission stages), the core and
  home network interfaces, the 3x1 mesh with its two-stage links, the
  pipelined home of each core (request buffer, tag/decide/data stages, 64
  MSHRs, the fill unit, the writeback queue, full-line store installs), the
  home and outlet links, and the line AXI adapter (32 slots, one 512-bit beat
  per line);
- the host side as the runtime drives it: DMA transfers that hand their range
  off (write back and invalidate the homes) before each transfer.

## What is approximated

- The memory behind the AXI adapters (HBM and the AXI slave): a fixed
  latency per outlet (25 / 44 / 63 cycles, writes 15 more) with the
  simulation bench's per-ID skew, every AXI ready high and the lowest due
  read answered first. The simulation bench's line model stalls its channels
  at random and picks among due reads at random; the board's HBM is neither.
  Memory-bound cases therefore carry most of the CA model's error.
- The reported device cycle count of each launch is 5 to 6 cycles above the
  simulator's (launch and completion boundary); everything between them is
  per-cycle.

## How it was checked

Against RTL simulation recordings of the R41 base, with the AXI adapters'
accept and response cycles replayed from the recording (the memory above is
then the only thing not modelled): on 12 visible ArchBench points of the
smallest tier, with one and with three active cores, every core's issue,
port requests and responses and every uncore stream (home requests and
responses, memory requests and responses, the adapters' accepts and
responses) equal the recording on every launch.

Standalone (nothing replayed), three active cores, the same 12 visible
points against the same recordings: median absolute error 2.0 %, worst
+17.3 % (a memory-bound matrix-vector product), the rest within +7.7 %; the
model's count was above the simulator's on 10 of 12. This is a spot check on small cases, not a
calibration: larger cases spend a larger share of their time in memory,
where the approximation sits.

## Not exercised by any recording yet

These follow the RTL's text but no recorded case reached them, so they are
unverified: a dirty line's eviction (victim writeback), a request blocked for
want of an MSHR, a target or an unclaimed way, a tag replay, a full response
or writeback queue, the home link refusing a memory request, and the tensor
engine's line stores.

## What it does not follow

Only the configuration's keys: changing RTL parameters other than those in
`r41-board-cycle.cfg` (and the core profile's keys it reads) is invisible to
it unless you change the model's source. It follows the compiler fully (it
runs the compiled program instruction by instruction, register numbers and
banks included).

## Run time

A few seconds per case (process start and compile included); the model
itself runs at roughly 10,000 to 40,000 device cycles per second of host
time, so a case of a few hundred thousand cycles takes tens of seconds.
Slower than the Event Model, faster than the simulator by two orders of
magnitude.

Neither model is the score: the score is the RTL simulation and the board.
