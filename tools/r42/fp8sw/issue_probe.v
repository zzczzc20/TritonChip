// Diagnostic probe (simulation only; not RTL, not part of the device): a second top that reads the core by hierarchical
// reference and prints one line per issued instruction ("I <cycle> r<rctx> <pc hex>") between the launch start and the
// core going idle, plus the cycle at which each launch starts ("L") and ends ("S"). Enabled by +iprobe=<file>.
// Used by tools/r42/fp8sw/stall_profile.py to attribute the cycles of a kernel to the instructions that wait.
`timescale 1ns/1ps
module issue_probe;
    integer fh; reg [1023:0] path; reg on;
    reg [63:0] cyc; reg run_q;
    initial begin
        on = 0; cyc = 0; run_q = 0;
        if ($value$plusargs("iprobe=%s", path)) begin fh = $fopen(path, "w"); on = 1; end
    end
    wire clk = tnca_device.dut.u_core.clk;
    wire run = |tnca_device.dut.u_core.running;
    always @(posedge clk) if (on) begin
        cyc <= cyc + 1;
        run_q <= run;
        if (run && !run_q) $fdisplay(fh, "L %0d", cyc);
        if (run && tnca_device.dut.u_core.x_valid)
            $fdisplay(fh, "I %0d r%0d %0h", cyc, tnca_device.dut.u_core.x_rctx, tnca_device.dut.u_core.x_pc);
        if (!run && run_q) $fdisplay(fh, "S %0d", cyc);
    end
endmodule
