# Ring stages and actual engine admission

`kickoff/pipeline-admission-v2-evidence.tgz` preserves the registered experiment:
four model candidates (baseline and stages 1/2/3), three RTL stage candidates,
and comparison. All seven populations pass standard Torch mm/addmm numerical
checks on 16x16x48 and 17x31x65, with actual inputs, placements, ring plans,
artifacts and raw RTL traces retained. V1's invalid backend/core-mask request
is retained as a failed attempt; the runtime guard was not weakened.

For 16x16x48 mm, model cycles are 3818/3872/3261/3266 for baseline/stages 1/2/3.
RTL stage cycles are 2183/1847/1843, with actual transfer/op1-MMA overlap
0/294/292 cycles. ACC row operations and zero/config commands do not count as
useful MMA overlap. On tail mm, RTL stage cycles are 61503/60725/66467: a third
stage has more observed overlap (540 versus 411 cycles) and still runs slower.
Thus additional overlap alone does not establish a makespan improvement.

Model and RTL shader hashes differ because their physical-chunk compilation
domains differ. Their memory profiles differ too. These are within-domain
candidate observations and RTL mechanism checks, not matched cross-domain
accuracy measurements. Model accumulated queue/service counters overlap in
time and are not exclusive fractions of elapsed cycles.

The supplemental strict comparison requires the exact seven registered labels,
unique workload rows, matching actual inputs and label/stage agreement. Missing
or duplicate candidates refuse. It does not rerun or relabel the original
campaign. Physical-board closure remains outstanding; legal candidates remain
available and Mode A remains default-off.
