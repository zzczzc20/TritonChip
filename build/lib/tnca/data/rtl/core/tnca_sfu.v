//==============================================================================
// tnca_sfu -- the special function unit (ISA sec 8.8)
//
// Four approximate binary32 functions -- exp2, rcp, rsqrt, log2 -- with a
// published error bound of 2^-22 rather than a specified result. The four
// pipelines themselves are generated from xls/tnca_sfu.x; see doc/xls.md. This
// module is the part that belongs to the core: how many of them there are, and
// how a VLEN-wide vector gets through them.
//
// ONE SLICE, VLEN PASSES
// ----------------------
// There is a single four-function slice, and lanes go through it one per
// cycle. That is a deliberate ratio, not a shortcut. The four generated units
// come to about 60 k gates between them, against 7 k for one binary32 FMA, so
// sixteen SFU lanes would cost more than the entire rest of the vector
// datapath -- for operations that the architecture overview (sec 6.2) puts
// firmly off the GEMM inner loop. The slice is fully pipelined, so the cost of
// the choice is latency on one instruction, not throughput on a stream of
// them.
//
// The four functions are evaluated in parallel on every lane and the result is
// selected. They share no logic -- each has its own coefficient tables -- so
// selecting the input instead would save nothing but the mux.
//
// SEQUENCE
// --------
// `start` latches the source vector, because by the time the last lane is
// issued the execute stage belongs to a different instruction. Then, with
// `cnt` counting cycles from zero:
//
//     cnt < VLEN            lane `cnt` is presented and accepted by the slice
//     cnt >= LAT+1          lane `cnt - LAT - 1` emerges and is captured
//     cnt == VLEN + LAT     the last lane lands; `done` and `vres` are valid
//                           in the following cycle
//
// LAT is the generated units' latency and is checked against them by
// tb_sfu_core: a wrong value there reads the wrong lane out, not merely a
// late one, so it cannot pass by accident.
//
// Masked-off lanes are computed and then not written -- the lane enable at
// writeback drops them. Skipping them would make the instruction's duration
// depend on the mask, which buys a few cycles on an operation that is already
// off the critical path and costs a variable-latency completion.
//==============================================================================
`include "tnca_defs.vh"

module tnca_sfu #(
    parameter VLEN = 16,
    parameter ELEN = 32,
    // Latency of rtl/gen/tnca_sfu_*_xls.v: stages minus one, because they are
    // generated with --flop_inputs=false. All four share it on purpose, so
    // that this sequencer does not have to know which function it is running.
    parameter LAT  = 6
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start,      // 1-cycle issue pulse
    input  wire [7:0]            func,       // TNCA_SFU_*
    input  wire [VLEN*ELEN-1:0]  va,

    output reg                   busy,
    output reg                   done,
    output wire [VLEN*ELEN-1:0]  vres        // valid in the cycle `done` is high
);
    localparam CW = 8;                       // cycle counter width

    integer i;
    genvar  g;

    reg [ELEN-1:0] src [0:VLEN-1];
    reg [ELEN-1:0] res [0:VLEN-1];
    reg [7:0]      op_l;
    reg [CW-1:0]   cnt;

    wire issuing = busy && (cnt < VLEN);
    wire landing = busy && (cnt >= LAT + 1);
    wire [CW-1:0] land_ix = cnt - LAT - 1;

    reg [ELEN-1:0] x;
    always @(*) begin
        x = {ELEN{1'b0}};
        for (i = 0; i < VLEN; i = i + 1)
            if (issuing && (cnt == i[CW-1:0])) x = src[i];
    end

    wire [ELEN-1:0] y_exp2, y_log2, y_rcp, y_rsqrt;
    wire rst = ~rst_n;

    tnca_sfu_exp2_xls  u_exp2 (.clk(clk), .rst(rst), .x(x), .out(y_exp2));
    tnca_sfu_rcp_xls   u_rcp  (.clk(clk), .rst(rst), .x(x), .out(y_rcp));
    tnca_sfu_rsqrt_xls u_rsqrt(.clk(clk), .rst(rst), .x(x), .out(y_rsqrt));
    tnca_sfu_log2_xls  u_log2 (.clk(clk), .rst(rst), .x(x), .out(y_log2));

    reg [ELEN-1:0] y;
    always @(*) begin
        case (op_l)
        `TNCA_SFU_EXP2:  y = y_exp2;
        `TNCA_SFU_RCP:   y = y_rcp;
        `TNCA_SFU_RSQRT: y = y_rsqrt;
        default:         y = y_log2;
        endcase
    end

    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_out
            assign vres[g*ELEN +: ELEN] = res[g];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            cnt  <= {CW{1'b0}};
            op_l <= 8'd0;
            for (i = 0; i < VLEN; i = i + 1) begin
                src[i] <= {ELEN{1'b0}};
                res[i] <= {ELEN{1'b0}};
            end
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    for (i = 0; i < VLEN; i = i + 1)
                        src[i] <= va[i*ELEN +: ELEN];
                    op_l <= func;
                    cnt  <= {CW{1'b0}};
                    busy <= 1'b1;
                end
            end else begin
                if (landing)
                    for (i = 0; i < VLEN; i = i + 1)
                        if (land_ix == i[CW-1:0]) res[i] <= y;
                if (cnt == VLEN + LAT) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
                cnt <= cnt + 1'b1;
            end
        end
    end
endmodule
