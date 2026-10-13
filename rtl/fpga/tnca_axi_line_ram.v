// Dual-port line storage. Valid/fill ownership stays in the bridge.
// Port A receives fill beats; port B reads completed lines or applies
// byte-masked stores. The bridge excludes conflicting same-word accesses.
// No reset or initialization of data: every valid line has been filled.
`default_nettype none
module tnca_axi_line_ram #(
    parameter WORDS = 128,
    parameter ADDR_W = 7
) (
    input wire clk,
    input wire a_we,
    input wire [ADDR_W-1:0] a_addr,
    input wire [63:0] a_wdata,
    input wire b_en,
    input wire [7:0] b_we,
    input wire [ADDR_W-1:0] b_addr,
    input wire [63:0] b_wdata,
    output reg [63:0] b_rdata
);
    (* ram_style = "block" *) reg [63:0] data [0:WORDS-1];
    integer byte_i;
    always @(posedge clk)
        if (a_we) data[a_addr] <= a_wdata;
    always @(posedge clk) begin
        if (b_en) begin
            for (byte_i = 0; byte_i < 8; byte_i = byte_i + 1)
                if (b_we[byte_i])
                    data[b_addr][byte_i*8 +: 8] <= b_wdata[byte_i*8 +: 8];
            b_rdata <= data[b_addr];
        end
    end
endmodule
`default_nettype wire
