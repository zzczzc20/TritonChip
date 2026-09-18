//==============================================================================
// tnca_mem_model -- behavioural unified memory for Stage-1 bring-up
//
// One flat array serves both the instruction port (asynchronous read, matching
// the core's Stage-1 fetch model) and the data port (valid/ready request,
// valid response, with a configurable fixed latency).
//
// LATENCY is a parameter on purpose: running the same test at latency 0, 1 and
// 8 is the cheapest way to catch a core that accidentally assumes a fixed
// memory response time. RANDOM_STALL additionally drops mem_req_ready at
// pseudo-random cycles so the request handshake is exercised, not just the
// happy path.
//==============================================================================
`timescale 1ns/1ps

module tnca_mem_model #(
    parameter WORDS        = 8192,     // 64-bit words
    parameter MEM_DW       = 64,
    parameter MEM_BYTES    = 8,
    parameter LATENCY      = 1,
    parameter RANDOM_STALL = 0,
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
    output reg                      rsp_valid,
    output reg  [MEM_DW-1:0]        rsp_rdata
);

    reg [MEM_DW-1:0] mem [0:WORDS-1];

    integer i;
    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = {MEM_DW{1'b0}};
    end

    // ---- instruction port ---------------------------------------------------
    wire [63:0] iw = imem_addr >> 3;
    assign imem_data = (iw < WORDS) ? mem[iw[31:0]] : 64'd0;

    // ---- data port ----------------------------------------------------------
    reg [31:0] lfsr;
    always @(posedge clk) begin
        if (!rst_n) lfsr <= SEED;
        else        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    assign req_ready = (RANDOM_STALL != 0) ? lfsr[3] : 1'b1;

    reg [MEM_DW-1:0] pipe_data [0:15];
    reg [15:0]       pipe_vld;

    wire [63:0] dw = req_addr >> 3;
    wire        in_range = (dw < WORDS);

    reg [MEM_DW-1:0] rd;
    integer k;

    always @(posedge clk) begin
        if (!rst_n) begin
            pipe_vld  <= 16'd0;
            rsp_valid <= 1'b0;
            rsp_rdata <= {MEM_DW{1'b0}};
        end else begin
            // shift the response pipeline
            for (k = 15; k > 0; k = k - 1) begin
                pipe_vld[k]  <= pipe_vld[k-1];
                pipe_data[k] <= pipe_data[k-1];
            end
            pipe_vld[0]  <= 1'b0;
            pipe_data[0] <= {MEM_DW{1'b0}};

            if (req_valid && req_ready) begin
                rd = in_range ? mem[dw[31:0]] : {MEM_DW{1'b0}};
                if (req_we && in_range) begin
                    for (k = 0; k < MEM_BYTES; k = k + 1)
                        if (req_be[k]) mem[dw[31:0]][k*8 +: 8] <= req_wdata[k*8 +: 8];
                end
                if (LATENCY == 0) begin
                    rsp_valid <= 1'b1;
                    rsp_rdata <= rd;
                end else begin
                    pipe_vld[0]  <= 1'b1;
                    pipe_data[0] <= rd;
                end
            end else if (LATENCY == 0) begin
                rsp_valid <= 1'b0;
            end

            if (LATENCY > 0) begin
                rsp_valid <= pipe_vld[LATENCY-1];
                rsp_rdata <= pipe_data[LATENCY-1];
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
