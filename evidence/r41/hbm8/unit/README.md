# tb_line_axi_hbm8 runs (GB10, Icarus Verilog 12.0)

    iverilog -g2012 -I <rtl>/common -o t.vvp tb/fpga/tb_line_axi_hbm8.v <rtl>/fpga/tnca_line_axi.v && vvp t.vvp

* `image54_tnca_line_axi.log`: `<rtl>` = `git archive 54249f7 rtl` (the B40 image source).
* `frozen_tnca_line_axi.log`: `<rtl>` = this tree (frozen Stage 1 RTL, registered aperture copies).
* `mutants.log`: the image file with (1) `bus_address` cut to 33 bits,
  (2) the `req_addr-MEM_BASE<=axi_size-64` limit removed, (3) the alignment
  test removed; each run must stop with the shown FATAL.
