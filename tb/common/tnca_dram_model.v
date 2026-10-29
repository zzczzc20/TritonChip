//==============================================================================
// tnca_dram_model -- behavioural line-granular memory behind the NUCA slices
//
// Tagged, variable latency, optional out-of-order responses and optional
// request back-pressure. Everything a slice might wrongly assume about its
// memory is a parameter here, so the assumption fails in a unit test rather
// than in the 2x2 integration.
//==============================================================================
`timescale 1ns/1ps

module tnca_dram_model #(
    parameter ADDR_W     = 64,
    parameter LINE_BYTES = 64,
    parameter LINE_BITS  = 512,
    parameter LINES      = 4096,     // capacity, in lines
    parameter TAG_W      = 4,
    parameter SRC_W      = 4,
    parameter LATENCY    = 10,
    parameter RANDOM_STALL = 0,
    parameter REORDER    = 0,
    // Bandwidth: at most one request (read or write) every INTERVAL cycles.
    // 1 is the unlimited model every bench had before DCO; a cache policy
    // that saves traffic only shows up in time when traffic costs time.
    parameter INTERVAL   = 1,
    // Reads in flight. A read arriving with the pool full is lost, so a
    // requester with more tags than this needs a larger pool.
    parameter POOL       = 32,
    parameter SEED       = 32'hACE1_2345
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [ADDR_W-1:0]        req_addr,
    input  wire                     req_we,
    input  wire [LINE_BITS-1:0]     req_wdata,
    input  wire [TAG_W-1:0]         req_tag,
    input  wire [SRC_W-1:0]         req_src,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [SRC_W-1:0]         rsp_src,
    output reg  [LINE_BITS-1:0]     rsp_rdata
);

    localparam OFF_W = 6;

    reg [LINE_BITS-1:0] mem [0:LINES-1];

    integer i;
    initial for (i = 0; i < LINES; i = i + 1) mem[i] = {LINE_BITS{1'b0}};

    reg [31:0] lfsr;
    always @(posedge clk) begin
        if (!rst_n) lfsr <= SEED;
        else        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // Traffic, for the benches to report: lines read and written.
    integer n_rd = 0, n_wr = 0;

    reg [15:0] gap;
    always @(posedge clk) begin
        if (!rst_n)                                  gap <= 16'd0;
        else if (req_valid && req_ready && INTERVAL > 1) gap <= INTERVAL - 1;
        else if (gap != 16'd0)                        gap <= gap - 16'd1;
    end

    assign req_ready = ((RANDOM_STALL != 0) ? lfsr[5] : 1'b1) && (gap == 16'd0);

    reg [POOL-1:0]      p_valid;
    reg [LINE_BITS-1:0] p_data [0:POOL-1];
    reg [TAG_W-1:0]     p_tag  [0:POOL-1];
    reg [SRC_W-1:0]     p_src  [0:POOL-1];
    reg [15:0]          p_cnt  [0:POOL-1];

    reg [63:0] lidx;
    reg        in_range;
    integer    k, slot, pick, nelig;
    reg        found;

    // Writes only acknowledge; a slice's writeback needs no data back, but it
    // does need to know the line left, so the response still carries its tag.
    always @(posedge clk) begin
        if (!rst_n) begin
            p_valid   <= {POOL{1'b0}};
            rsp_valid <= 1'b0;
        end else begin
            rsp_valid <= 1'b0;

            if (req_valid && req_ready) begin
                if (req_we) n_wr = n_wr + 1; else n_rd = n_rd + 1;
                lidx     = req_addr >> OFF_W;
                in_range = (lidx < LINES);
                slot = -1;
                for (k = POOL-1; k >= 0; k = k - 1)
                    if (!p_valid[k]) slot = k;
                if (req_we) begin
                    if (in_range) mem[lidx[31:0]] <= req_wdata;
                    // a writeback needs no response; it is fire-and-forget here
                end else if (slot >= 0) begin
                    p_valid[slot] <= 1'b1;
                    p_data[slot]  <= in_range ? mem[lidx[31:0]] : {LINE_BITS{1'b0}};
                    p_tag[slot]   <= req_tag;
                    p_src[slot]   <= req_src;
                    p_cnt[slot]   <= (LATENCY == 0) ? 16'd0 : (LATENCY - 1);
                end
            end

            for (k = 0; k < POOL; k = k + 1)
                if (p_valid[k] && p_cnt[k] != 16'd0) p_cnt[k] <= p_cnt[k] - 16'd1;

            nelig = 0;
            for (k = 0; k < POOL; k = k + 1)
                if (p_valid[k] && p_cnt[k] == 16'd0) nelig = nelig + 1;

            if (nelig > 0) begin
                pick  = (REORDER != 0) ? (lfsr[9:0] % nelig) : 0;
                found = 1'b0;
                for (k = 0; k < POOL; k = k + 1) begin
                    if (!found && p_valid[k] && p_cnt[k] == 16'd0) begin
                        if (pick == 0) begin
                            rsp_valid  <= 1'b1;
                            rsp_tag    <= p_tag[k];
                            rsp_src    <= p_src[k];
                            rsp_rdata  <= p_data[k];
                            p_valid[k] <= 1'b0;
                            found      = 1'b1;
                        end else pick = pick - 1;
                    end
                end
            end
        end
    end

    task preload_word;
        input [63:0] byte_addr;
        input [63:0] value;
        reg [63:0] li;
        reg [2:0]  wsel;
        begin
            li   = byte_addr >> OFF_W;
            wsel = byte_addr[5:3];
            mem[li[31:0]][wsel*64 +: 64] = value;
        end
    endtask

endmodule
