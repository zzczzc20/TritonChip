# BF16 strided division: preliminary diagnosis

For `op.div/membound/bf16-large-strided-r32768-c1024`, the previous and
corrected 100 MHz images passed numerical checks and recorded one formal
sample each: 150,164,159 versus 168,853,440 serial-grid cycles, 12.4% slower.
This single-sample observation requires remeasurement before a confirmed
performance conclusion. The paired records are in
`../membound_default_large_comparison.json`.

The `triton_poi_fused_div_slice_0` optimized IR, scheduled IR, and assembly
copied from the two board bundles are byte-identical. Their respective SHA-256
values are `f9f474efc3dcc62f0ab88167f8af3bc4b77d7bf7e3957c213f12ff7535a2c8c3`,
`d37c723ae0fb1cc14d1e9048ecab5545acb55f50faac017482705478504e5fbe`, and
`d0f5f8d65b544945fa935f7ae7b8dc0cf8e672c3f297b59dff143ac193bf043e`.
This rules out a difference in these captured instructions for this kernel;
it does not isolate the effect of hardware, memory layout, surrounding runtime
or measurement conditions. Investigate those after the active board campaign.

Both records contain exactly one launch of this kernel with 2,097,152 grid
programs. Dividing the aggregate cycle difference by the grid size gives
approximately 8.91 cycles per program, including multicore overlap. The
assembly contains two halfword gathers. Run 1 configured
`LSU_INFLIGHT=2`, while the candidate configures `LSU_INFLIGHT=1` alongside
other changes including striped HBM and the hybrid LSU. The in-flight limit
is a concrete hypothesis to isolate, not an established cause or a reason to
change the running image.

The separately completed medium BF16 strided point shows a similar difference:
three old samples 9,369,186/9,374,197/9,374,366 cycles versus three candidate
samples 10,553,709/10,551,033/10,549,707 cycles (median 12.55% slower).
The FP16 medium point also has three samples per image and a 12.56% slower
candidate median. The complete matching data are in
[`../membound_default_medium1_comparison.json`](../membound_default_medium1_comparison.json).
For the BF16 medium point, the old and candidate formal bundles' kernel
assemblies both have SHA-256
`d0f5f8d65b544945fa935f7ae7b8dc0cf8e672c3f297b59dff143ac193bf043e`,
the same as the large-point assembly. This repeat across shapes does not
isolate the hardware change or substitute for the registered recheck.

An earlier controlled RTL simulation of a different small strided add point
found that enabling the hybrid LSU alone raised its measured cycles by 22.59%
relative to the G2 arm, with one sample per arm; see
[`../../../strided/README.md`](../../../strided/README.md). That bench uses synthetic
memory and a different point and configuration, so it does not attribute the
board division result. Its later narrow-pair beat-count candidate is already
present in the frozen `8d237223` RTL: reverse-checking the recorded patch
succeeds and those two RTL files are unchanged between that commit and this
checkout. Reapplying that candidate is therefore not a fix for this image.

A later single-parameter RTL simulation of the small FP16 strided division
point reports 195,477 cycles with one in-flight LSU request versus 123,610
with two. Both numerical checks pass. Its synthetic memory and non-striped
configuration are not the board; see [`rtl_compare/README.md`](rtl_compare/README.md)
for the exact source, parameters and raw evidence.
