# Actual placed-core collection readiness

Installed Vivado 2023.2 help confirms get_slrs -of_objects supports cells. The new read-only collector flow/fpga/audit_r40_placed_cores.tcl queries only physically placed primitives within each actual core hierarchy, reports unplaced nonconstant logic, and requires distinct SLR0/1/2 placement. Pblock assignments are not queried. Netlist GND/VCC constants are excluded and counted.

The retained tests use Tcl query stubs to check collector semantics: correct locations, shared SLR refusal, missing core refusal, unplaced logic refusal, and preservation of an existing report. These are not actual placement results. Physical/performance gates remain not-measured.

The new collector is queued as a separate read-only follow-up after the existing post-route reports finish successfully. Both observers preserve failed implementation/audit results and perform no placement/route retry. The frozen software archive and live implementation sources are unchanged. An actual report and input hashes will be retained only after the final routed checkpoint exists.
