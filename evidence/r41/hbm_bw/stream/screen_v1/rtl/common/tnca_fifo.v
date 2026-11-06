//==============================================================================
// tnca_fifo -- synchronous FIFO with valid/ready on both sides
//
// The handshake rule the whole design relies on (arch overview sec 19.1):
// while `valid` is high and `ready` is low the payload does not change, and a
// transfer happens exactly on the cycle both are high.
//==============================================================================

module tnca_fifo #(
    parameter W     = 32,
    parameter DEPTH = 4,
    parameter AW    = 2          // $clog2(DEPTH)
) (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          in_valid,
    output wire          in_ready,
    input  wire [W-1:0]  in_data,

    output wire          out_valid,
    input  wire          out_ready,
    output wire [W-1:0]  out_data,

    output wire [AW:0]   count
);

    reg [W-1:0]  mem [0:DEPTH-1];
    reg [AW-1:0] rptr, wptr;
    reg [AW:0]   cnt;

    assign in_ready  = (cnt < DEPTH);
    assign out_valid = (cnt != 0);
    assign out_data  = mem[rptr];
    assign count     = cnt;

    wire push = in_valid  && in_ready;
    wire pop  = out_valid && out_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            rptr <= {AW{1'b0}};
            wptr <= {AW{1'b0}};
            cnt  <= {(AW+1){1'b0}};
        end else begin
            if (push) begin
                mem[wptr] <= in_data;
                wptr <= (wptr == DEPTH-1) ? {AW{1'b0}} : (wptr + 1'b1);
            end
            if (pop)
                rptr <= (rptr == DEPTH-1) ? {AW{1'b0}} : (rptr + 1'b1);
            case ({push, pop})
            2'b10: cnt <= cnt + 1'b1;
            2'b01: cnt <= cnt - 1'b1;
            default: ;
            endcase
        end
    end

endmodule
