// tb_router_miter -- tnca_router vs tnca_router_fast, same random traffic.
// Every cycle: in_ready and out_valid must be equal on all channels, and out_flit
// equal on every channel that has out_valid. Several node positions, NV 1 and 2,
// heavy and light injection, random output back-pressure.
`timescale 1ns/1ps
module tb_router_miter;
    parameter CYCLES = 200000;
    parameter SEED   = 1;
    parameter XINV   = 1;          // drive the flit of an invalid input as X (what a real producer's idle bus may be)
    localparam XW = 2, YW = 2, PW = 40, FW = XW + YW + PW;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer cyc = 0, errs = 0, xfers = 0;
    integer seed = SEED;
    reg [31:0] r;

    genvar x, y, nv;
    generate
      for (nv = 1; nv <= 2; nv = nv + 1) begin : g_nv
        for (y = 0; y < 3; y = y + 1) begin : g_y
          for (x = 0; x < 3; x = x + 1) begin : g_x
            localparam NC = 5*nv;
            reg  [NC-1:0]      iv;
            reg  [NC*FW-1:0]   ifl;
            reg  [NC-1:0]      orr;
            wire [NC-1:0]      ir_a, ir_b, ov_a, ov_b;
            wire [NC*FW-1:0]   of_a, of_b;
            integer pin, pout, k, mism;
            tnca_router #(.XW(XW),.YW(YW),.PAYLOAD_W(PW),.NV(nv),.DEPTH(4),.AW(2),.X_HERE(x),.Y_HERE(y)) a (
                .clk(clk),.rst_n(rst_n),.in_valid(iv),.in_ready(ir_a),.in_flit(ifl),
                .out_valid(ov_a),.out_ready(orr),.out_flit(of_a));
            tnca_router_fast #(.XW(XW),.YW(YW),.PAYLOAD_W(PW),.NV(nv),.DEPTH(4),.AW(2),.X_HERE(x),.Y_HERE(y)) b (
                .clk(clk),.rst_n(rst_n),.in_valid(iv),.in_ready(ir_b),.in_flit(ifl),
                .out_valid(ov_b),.out_ready(orr),.out_flit(of_b));
            // inputs change on the falling edge; compare just before the rising edge
            always @(negedge clk) begin
                pin  = (((cyc / 20000) % 4) == 0) ? 90 : (((cyc / 20000) % 4) == 1) ? 30 : 60;
                pout = (((cyc / 25000) % 3) == 0) ? 90 : (((cyc / 25000) % 3) == 1) ? 20 : 55;
                for (k = 0; k < NC; k = k + 1) begin
                    iv[k]  = ($urandom % 100) < pin;
                    orr[k] = ($urandom % 100) < pout;
                    ifl[k*FW +: 32]      = $urandom;
                    ifl[k*FW+32 +: 12]   = $urandom;
                    if (XINV && !iv[k]) ifl[k*FW +: FW] = {FW{1'bx}};
                end
            end
            always @(posedge clk) if (rst_n) begin
                mism = 0;
                if (ir_a !== ir_b) mism = 1;
                if (ov_a !== ov_b) mism = 1;
                for (k = 0; k < NC; k = k + 1) begin
                    if (ov_a[k] && (of_a[k*FW +: FW] !== of_b[k*FW +: FW])) mism = 1;
                    if (ov_a[k] && orr[k]) xfers = xfers + 1;
                end
                if (mism) begin
                    errs = errs + 1;
                    if (errs < 10) $display("MISMATCH nv=%0d node(%0d,%0d) cyc=%0d ir %b/%b ov %b/%b", nv, x, y, cyc, ir_a, ir_b, ov_a, ov_b);
                end
            end
          end
        end
      end
    endgenerate

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
    end
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc == CYCLES) begin
            if (errs == 0) $display("PASS tb_router_miter cycles=%0d transfers=%0d", cyc, xfers);
            else           $display("FAIL tb_router_miter errs=%0d", errs);
            $finish;
        end
    end
endmodule
