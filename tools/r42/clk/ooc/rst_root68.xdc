# root reset as the run-3 image delivers it to a unit: one synchroniser flop driving ~21k endpoints across the SLRs; 6.8 ns of net before the unit-internal distribution
set_input_delay -clock clk 6.8 [get_ports i_rst_n]
