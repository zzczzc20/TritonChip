// tb_uram_lat -- measures the read latency (cycles from the address edge to valid DOUT_B) of one URAM288 (unisim model, xsim)
// for the four combinations of IREG_PRE_B and OREG_B, and RAMB36E2 SDP with and without DOB register (DO_REG).
`timescale 1ns/1ps
module tb;
    reg clk = 0; always #5 clk = ~clk;
    reg [22:0] addr_a, addr_b; reg en_a, en_b, wr_a; reg [71:0] din;
    wire [71:0] dout [0:3];
    genvar g;
    generate for (g = 0; g < 4; g = g + 1) begin : u
        URAM288 #(.IREG_PRE_B(g[0] ? "TRUE" : "FALSE"), .OREG_B(g[1] ? "TRUE" : "FALSE"), .OREG_A("FALSE"), .IREG_PRE_A("FALSE"),
                  .BWE_MODE_A("PARITY_INDEPENDENT"), .BWE_MODE_B("PARITY_INDEPENDENT")) r (
            .CLK(clk), .ADDR_A(addr_a), .ADDR_B(addr_b), .EN_A(en_a), .EN_B(en_b), .RDB_WR_A(wr_a), .RDB_WR_B(1'b0),
            .BWE_A(9'h1ff), .BWE_B(9'h000), .DIN_A(din), .DIN_B(72'd0), .OREG_CE_A(1'b1), .OREG_CE_B(1'b1),
            .OREG_ECC_CE_A(1'b1), .OREG_ECC_CE_B(1'b1), .RST_A(1'b0), .RST_B(1'b0), .SLEEP(1'b0),
            .INJECT_DBITERR_A(1'b0), .INJECT_DBITERR_B(1'b0), .INJECT_SBITERR_A(1'b0), .INJECT_SBITERR_B(1'b0),
            .CAS_IN_ADDR_A(23'd0), .CAS_IN_ADDR_B(23'd0), .CAS_IN_BWE_A(9'd0), .CAS_IN_BWE_B(9'd0), .CAS_IN_DBITERR_A(1'b0), .CAS_IN_DBITERR_B(1'b0),
            .CAS_IN_DIN_A(72'd0), .CAS_IN_DIN_B(72'd0), .CAS_IN_DOUT_A(72'd0), .CAS_IN_DOUT_B(72'd0), .CAS_IN_EN_A(1'b0), .CAS_IN_EN_B(1'b0),
            .CAS_IN_RDACCESS_A(1'b0), .CAS_IN_RDACCESS_B(1'b0), .CAS_IN_RDB_WR_A(1'b0), .CAS_IN_RDB_WR_B(1'b0), .CAS_IN_SBITERR_A(1'b0), .CAS_IN_SBITERR_B(1'b0),
            .DOUT_B(dout[g]));
    end endgenerate
    integer cyc;
    initial begin
        en_a = 0; en_b = 0; wr_a = 0; addr_a = 0; addr_b = 0; din = 0; cyc = 0; #300;   // past the glbl GSR pulse
        repeat (4) @(posedge clk);
        // write value 0xAB..+k at addresses 5 and 6 (port A), one per cycle
        @(negedge clk); en_a = 1; wr_a = 1; addr_a = 5; din = 72'hA5;
        @(negedge clk); addr_a = 6; din = 72'hB6;
        @(negedge clk); en_a = 0; wr_a = 0;
        repeat (3) @(negedge clk);
        // read address 5 for exactly one cycle: the address is presented before edge E0; count edges until DOUT_B shows A5
        @(negedge clk); en_b = 1; addr_b = 5; cyc = 0;
        @(negedge clk); en_b = 0;      // after edge E0 (the edge that sampled addr 5)
        // DOUT visible after edge number N (E0 = 1)
        repeat (8) begin
            #1;
            $display("after edge %0d : dout = %h %h %h %h", cyc + 1, dout[0][7:0], dout[1][7:0], dout[2][7:0], dout[3][7:0]);
            @(negedge clk); cyc = cyc + 1;
        end
        $finish;
    end
endmodule
