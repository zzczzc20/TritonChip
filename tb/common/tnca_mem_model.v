//==============================================================================
// tnca_mem_model -- behavioural tagged memory for bring-up
//
// One flat array serves both the instruction port (asynchronous read, matching
// the core's Stage-1 fetch model) and a tagged data port whose responses may
// return OUT OF ORDER.
//
// Out-of-order responses are not a nicety: arch overview sec 20.3 lists
// "arbitrary sub-request return order" as a required correctness test, and once
// requests go to different NUCA slices through a mesh, ordered returns stop
// being a property anything guarantees. REORDER=1 makes the model deliver
// responses from a pool in pseudo-random order, so an LSU that quietly assumes
// FIFO returns fails here rather than in Stage 3.
//
// LATENCY and RANDOM_STALL are parameters for the same reason: running the same
// test at several latencies is the cheapest way to catch a fixed-latency
// assumption.
//==============================================================================
`timescale 1ns/1ps

module tnca_mem_model #(
    parameter WORDS        = 8192,     // 64-bit words
    parameter MEM_DW       = 64,
    parameter MEM_BYTES    = 8,
    parameter TAG_W        = 3,
    parameter LATENCY      = 1,
    parameter RANDOM_STALL = 0,
    parameter REORDER      = 0,
    parameter SEED         = 32'h1234_5678
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [63:0]              imem_addr,
    output wire [63:0]              imem_data,

    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [63:0]              req_addr,
    input  wire                     req_we,
    input  wire [MEM_BYTES-1:0]     req_be,
    input  wire [MEM_DW-1:0]        req_wdata,
    input  wire [TAG_W-1:0]         req_tag,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [MEM_DW-1:0]        rsp_rdata
);

    localparam POOL = 16;

    reg [MEM_DW-1:0] mem [0:WORDS-1];

    integer i;
    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = {MEM_DW{1'b0}};
    end

    // ---- instruction port ---------------------------------------------------
    wire [63:0] iw = imem_addr >> 3;
    assign imem_data = (iw < WORDS) ? mem[iw[31:0]] : 64'd0;

    // ---- pseudo-random source ------------------------------------------------
    reg [31:0] lfsr;
    always @(posedge clk) begin
        if (!rst_n) lfsr <= SEED;
        else        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    assign req_ready = (RANDOM_STALL != 0) ? lfsr[3] : 1'b1;

    // ---- in-flight pool ------------------------------------------------------
    // Each accepted request lands here with a countdown. When the countdown
    // expires the entry becomes eligible; the responder picks an eligible entry
    // (the lowest, or a pseudo-random one under REORDER) and retires it.
    reg [POOL-1:0]       p_valid;
    reg [MEM_DW-1:0]     p_data  [0:POOL-1];
    reg [TAG_W-1:0]      p_tag   [0:POOL-1];
    reg [7:0]            p_cnt   [0:POOL-1];

    reg [63:0] dw;
    reg        in_range;
    reg [MEM_DW-1:0] rd;

    integer k, slot, pick, nelig;
    reg     found;

    always @(posedge clk) begin
        if (!rst_n) begin
            p_valid   <= {POOL{1'b0}};
            rsp_valid <= 1'b0;
            rsp_tag   <= {TAG_W{1'b0}};
            rsp_rdata <= {MEM_DW{1'b0}};
        end else begin
            rsp_valid <= 1'b0;

            // ---- accept --------------------------------------------------
            if (req_valid && req_ready) begin
                dw       = req_addr >> 3;
                in_range = (dw < WORDS);
                rd       = in_range ? mem[dw[31:0]] : {MEM_DW{1'b0}};
                if (req_we && in_range) begin
                    for (k = 0; k < MEM_BYTES; k = k + 1)
                        if (req_be[k]) mem[dw[31:0]][k*8 +: 8] <= req_wdata[k*8 +: 8];
                end
                slot  = -1;
                for (k = POOL-1; k >= 0; k = k - 1)
                    if (!p_valid[k]) slot = k;
                if (slot >= 0) begin
                    p_valid[slot] <= 1'b1;
                    p_data[slot]  <= rd;
                    p_tag[slot]   <= req_tag;
                    p_cnt[slot]   <= (LATENCY == 0) ? 8'd0 : (LATENCY - 1);
                end
            end

            // ---- age -----------------------------------------------------
            for (k = 0; k < POOL; k = k + 1)
                if (p_valid[k] && p_cnt[k] != 8'd0)
                    p_cnt[k] <= p_cnt[k] - 8'd1;

            // ---- retire one eligible entry -------------------------------
            nelig = 0;
            for (k = 0; k < POOL; k = k + 1)
                if (p_valid[k] && p_cnt[k] == 8'd0) nelig = nelig + 1;

            if (nelig > 0) begin
                if (REORDER != 0) begin
                    pick = lfsr[7:0] % nelig;     // pseudo-random eligible entry
                end else begin
                    pick = 0;                     // oldest-indexed: in order
                end
                found = 1'b0;
                for (k = 0; k < POOL; k = k + 1) begin
                    if (!found && p_valid[k] && p_cnt[k] == 8'd0) begin
                        if (pick == 0) begin
                            rsp_valid  <= 1'b1;
                            rsp_tag    <= p_tag[k];
                            rsp_rdata  <= p_data[k];
                            p_valid[k] <= 1'b0;
                            found      = 1'b1;
                        end else begin
                            pick = pick - 1;
                        end
                    end
                end
            end
        end
    end

    // ---- loading and dumping -------------------------------------------------
    task load_hex;
        input [1023:0] path;
        begin
            $readmemh(path, mem);
        end
    endtask

    task dump_hex;
        input [1023:0] path;
        input integer first;
        input integer count;
        integer fh, j;
        begin
            fh = $fopen(path, "w");
            for (j = first; j < first + count; j = j + 1)
                $fwrite(fh, "%016x\n", mem[j]);
            $fclose(fh);
        end
    endtask

endmodule
