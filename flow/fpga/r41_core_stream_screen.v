module r41_core_stream_screen(input ap_clk, input ap_rst_n);
    (* DONT_TOUCH = "yes" *) tnca_multicore_hbm_xo core(.ap_clk(ap_clk), .ap_rst_n(ap_rst_n));
    (* DONT_TOUCH = "yes" *) tnca_stream_bank stream_bank(.clk(ap_clk), .rst_n(ap_rst_n));
endmodule
