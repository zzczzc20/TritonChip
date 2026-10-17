# Owner Selection: SystemVerilog Compatibility Repair

CI run `38730337676` exposed a parser compatibility regression: `matches`,
used as a local variable in the owner function, is a SystemVerilog keyword.
The original policy matrix used Verilog-2005, and Yosys accepted the extracted
function; neither check established compatibility with the SystemVerilog-2012
frontend used by the system simulations. The failed CI log is retained.

The repair renames the variable to `eligible_mask` without changing selection
logic. Source-bound verification of the corrected RTL includes the complete
policy matrix under both language standards, the 20-configuration physical
core/grid RTL regression, the 1–16-entry compositional proof, and the actual
eight-entry unrestricted 64-bit equivalence proof with a tie mutation.

The architecture screen now checks both Icarus language modes before running
proof or mapping. Reintroducing the original variable name is a negative
control: Verilog-2005 accepts it, SystemVerilog-2012 rejects it, and the screen
exits before starting expensive analysis. The corrected candidate passes both
frontends. Its nine-query generic mapping remains 30 levels and 18,675 LUT6
cells versus the reference's 61 levels and 24,030 cells. These remain generic
mapping observations, not FPGA timing or graph performance results.

`summary.json` identifies the corrected source and raw proof reports.
`raw.tar.gz` retains both source/tool versions, negative-control records,
solver inputs, mapping output, policy matrices, and the full grid regression.
The earlier parent evidence remains bound to its original source hash;
this follow-up does not retroactively change those records.
