// Diagnostic probe (not RTL, not committed to rtl/ or tb/): a second top that reads the device by
// hierarchical reference and prints engine events and occupancy. Enabled by +gprobe=<file>.
`timescale 1ns/1ps
module gemm_probe;
    integer fh; reg [1023:0] path; reg on; reg ev;
    reg [63:0] cyc, c_run, c_mtx, c_txf, c_both, c_none, c_mma, n_mma, n_txf, n_iss;
    reg run_q; reg mtx_q; reg txf_q; reg [63:0] txf_pc_q;
    initial begin
        on = 0; ev = 0; cyc = 0; run_q = 0; mtx_q = 0; txf_q = 0; txf_pc_q = 0;
        c_run=0; c_mtx=0; c_txf=0; c_both=0; c_none=0; c_mma=0; n_mma=0; n_txf=0; n_iss=0;
        if ($value$plusargs("gprobe=%s", path)) begin fh = $fopen(path, "w"); on = 1; end
        ev = $test$plusargs("gprobe_ev");
    end
    wire clk = tnca_device.dut.u_core.clk;
    wire run = |tnca_device.dut.u_core.running;
    wire mb  = tnca_device.dut.u_core.mtx_busy;
    wire tb  = tnca_device.dut.u_core.txf_busy_any;
    reg  ismma;
    always @(posedge clk) if (on) begin
        cyc <= cyc + 1;
        run_q <= run; mtx_q <= mb; txf_q <= tb;
        if (run && !run_q) begin
            c_run=0; c_mtx=0; c_txf=0; c_both=0; c_none=0; c_mma=0; n_mma=0; n_txf=0; n_iss=0;
            $fdisplay(fh, "L %0d", cyc);
        end
        if (run) begin
            c_run = c_run + 1;
            if (mb) c_mtx = c_mtx + 1;
            if (tb) c_txf = c_txf + 1;
            if (mb && tb) c_both = c_both + 1;
            if (!mb && !tb) c_none = c_none + 1;
            if (mb && ismma) c_mma = c_mma + 1;
        end
        if (tnca_device.dut.u_core.mtx_cmd_valid) begin
            ismma <= (tnca_device.dut.u_core.mtx_cmd_op == 2'd1);
            if (tnca_device.dut.u_core.mtx_cmd_op == 2'd1) n_mma = n_mma + 1;
            if (ev) $fdisplay(fh, "M %0d r%0d op%0d a%0d va%0d vb%0d", cyc, tnca_device.dut.u_core.x_rctx,
                              tnca_device.dut.u_core.mtx_cmd_op, tnca_device.dut.u_core.x_dsti,
                              tnca_device.dut.u_core.x_va_idx, tnca_device.dut.u_core.x_vb_idx);
        end
        if (ev && tnca_device.dut.u_core.mtx_done) $fdisplay(fh, "m %0d", cyc);
        if (tnca_device.dut.u_core.txf_is_xfr) begin
            n_txf = n_txf + 1;
            if (ev) $fdisplay(fh, "T %0d r%0d w%0d st%0d pc%0h", cyc, tnca_device.dut.u_core.x_rctx,
                              tnca_device.dut.u_core.x_dsti, tnca_device.dut.u_core.x_func,
                              tnca_device.dut.u_core.x_pc);
        end
        if (ev && tnca_device.dut.u_core.txf_done) $fdisplay(fh, "t %0d", cyc);
        if (ev && tb != txf_q) $fdisplay(fh, "tb %0d %0d", cyc, tb);
        if (run && tnca_device.dut.u_core.x_valid) begin
            n_iss = n_iss + 1;
            if (ev) $fdisplay(fh, "I %0d r%0d %0h", cyc, tnca_device.dut.u_core.x_rctx, tnca_device.dut.u_core.x_pc);
        end
        if (!run && run_q)
            $fdisplay(fh, "S end=%0d run=%0d mtx=%0d mma_busy=%0d txf=%0d both=%0d none=%0d n_mma=%0d n_txf=%0d x_valid=%0d",
                      cyc, c_run, c_mtx, c_mma, c_txf, c_both, c_none, n_mma, n_txf, n_iss);
    end
endmodule
