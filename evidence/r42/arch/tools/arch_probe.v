// Diagnostic probe (not RTL, not committed to rtl/ or tb/): a second simulation top that reads the single-core
// device by hierarchical reference and prints one line per issued instruction: "I <rctx> <major> <func>",
// plus "L <cycle>" at launch start and "S <cycle> <running cycles>" at launch end. Enabled by +aprobe=<file>.
// Derived from evidence/r42/gemm/tools/gemm_probe.v (same hierarchical names).
`timescale 1ns/1ps
module arch_probe;
    integer fh; reg [1023:0] path; reg on;
    reg [63:0] cyc, c_run, n_iss;
    reg run_q;
    initial begin
        on = 0; cyc = 0; run_q = 0; c_run = 0; n_iss = 0;
        if ($value$plusargs("aprobe=%s", path)) begin fh = $fopen(path, "w"); on = 1; end
    end
    wire clk = tnca_device.dut.u_core.clk;
    wire run = |tnca_device.dut.u_core.running;
    always @(posedge clk) if (on) begin
        cyc <= cyc + 1;
        run_q <= run;
        if (run && !run_q) begin c_run = 0; n_iss = 0; $fdisplay(fh, "L %0d", cyc); end
        if (run) c_run = c_run + 1;
        if (run && tnca_device.dut.u_core.x_valid) begin
            n_iss = n_iss + 1;
            $fdisplay(fh, "I %0d %0d %0d", tnca_device.dut.u_core.x_rctx, tnca_device.dut.u_core.x_major, tnca_device.dut.u_core.x_func);
        end
        if (!run && run_q) $fdisplay(fh, "S %0d %0d %0d", cyc, c_run, n_iss);
    end
endmodule
