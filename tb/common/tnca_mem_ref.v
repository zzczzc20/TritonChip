//==============================================================================
// tnca_mem_ref -- a SECOND memory behind the same adaptation boundary
//
// doc/phase3_6_tasks_and_acceptance.md R36-MEM-02: at least one independent
// reference implementation must complete a real RTL regression through the
// public memory and fetch boundary, covering the latency, backpressure,
// partial writes and collisions the contract allows, with values, ownership
// and drain correct.
//
// WHY A SECOND ONE AT ALL
//
// tb/common/tnca_mem_model.v is the only memory the core has ever run
// against. Everything it happens to do -- the order it retires responses in,
// the fact that `req_ready` is a pure function of one LFSR bit, that a write
// is answered exactly like a read, that the instruction port reads the same
// array a store just wrote -- is indistinguishable, from inside the core,
// from something the contract requires. The way to find out which is which is
// to write a different memory from the port list and see what breaks.
//
// HOW THIS ONE IS DIFFERENT
//
//   storage      a BYTE array, not a word array. Partial writes are byte
//                writes rather than a masked read-modify-write of a word, so
//                a byte enable that is wrong in the adapter shows up here as
//                a wrong byte rather than as the same wrong word twice.
//
//   pipeline     a fixed-depth SHIFT REGISTER, not a pool of counters. A
//                response leaves when it reaches the end, so the latency is
//                exactly LATENCY for every request and two requests accepted
//                on the same... they cannot be: one port, one accept a cycle.
//
//   reorder      a two-deep SWAP stage at the end of the shift register,
//                which inverts the order of adjacent pairs. That is a
//                different way to be out of order from picking randomly
//                among eligible entries, and it is the adversarial one: it
//                guarantees an inversion whenever two responses are ready
//                back to back, rather than producing one with some
//                probability.
//
//   backpressure a run-length pattern, not one LFSR bit: STALL_RUN cycles
//                closed, STALL_RUN open. A ready that toggles every cycle
//                lets a requester that only ever offers on alternate cycles
//                look as though it handles backpressure.
//
//   fetch        registered from the SAME byte array, so an instruction
//                fetched after a store to that address sees the new word. The
//                contract does not promise that; it is recorded here as a
//                property of this model rather than assumed by anything.
//
// It is not independent in the sense of being written by somebody else. It is
// written from the port list rather than from the other model, which is the
// most this repository can honestly claim, and R36-MEM-02's evidence records
// it that way.
//==============================================================================
`timescale 1ns/1ps

module tnca_mem_ref #(
    parameter WORDS        = 8192,     // 64-bit words
    parameter MEM_DW       = 64,
    parameter MEM_BYTES    = 8,
    parameter TAG_W        = 3,
    parameter LATENCY      = 1,
    parameter RANDOM_STALL = 0,
    parameter REORDER      = 0,
    parameter STALL_RUN    = 3,
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
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [MEM_DW-1:0]        req_cmp,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [MEM_DW-1:0]        rsp_rdata
);

`include "tnca_amo.vh"

    localparam integer BYTES = WORDS * MEM_BYTES;
    localparam integer DEPTH = (LATENCY < 1) ? 1 : LATENCY;

    reg [7:0] b [0:BYTES-1];

    integer i;
    initial begin
        for (i = 0; i < BYTES; i = i + 1) b[i] = 8'd0;
    end

    // ---- reading and writing a word out of the byte array --------------------
    function [MEM_DW-1:0] word_at;
        input [63:0] byte_addr;
        integer j;
        begin
            word_at = {MEM_DW{1'b0}};
            for (j = 0; j < MEM_BYTES; j = j + 1)
                word_at[j*8 +: 8] = ((byte_addr + j) < BYTES)
                                    ? b[(byte_addr + j)] : 8'd0;
        end
    endfunction

    // ---- instruction port ----------------------------------------------------
    // Registered, like the SRAM it stands for: the address presented this
    // cycle returns its word on the next, which is the stage tnca_core's
    // fetch buffer is built around.
    wire [63:0] ibase = {imem_addr[63:3], 3'd0};
    reg  [63:0] imem_q;
    always @(posedge clk) imem_q <= word_at(ibase);
    assign imem_data = imem_q;

    // ---- backpressure --------------------------------------------------------
    // A run-length pattern rather than a coin flip. STALL_RUN cycles refusing,
    // then STALL_RUN accepting.
    reg [15:0] phase;
    always @(posedge clk) begin
        if (!rst_n) phase <= 16'd0;
        else if (phase == (2*STALL_RUN - 1)) phase <= 16'd0;
        else phase <= phase + 16'd1;
    end
    wire stalling = (RANDOM_STALL != 0) && (phase < STALL_RUN);

    // The shift register must have room for the entry that would enter this
    // cycle. It always does -- one accept per cycle, one departure per cycle,
    // fixed depth -- so readiness is only ever about the stall pattern.
    assign req_ready = !stalling;

    // ---- the delay line ------------------------------------------------------
    reg              s_valid [0:DEPTH-1];
    reg [TAG_W-1:0]  s_tag   [0:DEPTH-1];
    reg [MEM_DW-1:0] s_data  [0:DEPTH-1];

    // ---- the swap stage ------------------------------------------------------
    // Holds one response back so the next one can overtake it. Under REORDER
    // every adjacent pair of responses comes out inverted, which is the
    // strongest thing a single-port memory can do to an LSU that assumes FIFO
    // returns.
    reg              h_valid;
    reg [TAG_W-1:0]  h_tag;
    reg [MEM_DW-1:0] h_data;

    reg [63:0]       ba;
    reg [MEM_DW-1:0] rd;
    reg [63:0]       amo_res;
    integer          j, k;

    // Counters, so a run can show what the parameters actually did. Three
    // configurations with three identical counts are one configuration run
    // three times.
    integer n_stall_cycles, n_swaps, n_accepted, n_partial;

    initial begin
        n_stall_cycles = 0;
        n_swaps        = 0;
        n_accepted     = 0;
        n_partial      = 0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 0; k < DEPTH; k = k + 1) s_valid[k] <= 1'b0;
            h_valid   <= 1'b0;
            rsp_valid <= 1'b0;
            rsp_tag   <= {TAG_W{1'b0}};
            rsp_rdata <= {MEM_DW{1'b0}};
        end else begin
            rsp_valid <= 1'b0;

            if (req_valid && !req_ready) n_stall_cycles <= n_stall_cycles + 1;

            // ---- accept, and do the memory work now ----------------------
            //
            // The read value is captured at accept, not at retire. A later
            // write to the same address must not change an answer that was
            // already determined -- that is what "the request was accepted"
            // means, and a model that read at retire would hide an ordering
            // bug in the requester.
            for (k = DEPTH-1; k > 0; k = k - 1) begin
                s_valid[k] <= s_valid[k-1];
                s_tag[k]   <= s_tag[k-1];
                s_data[k]  <= s_data[k-1];
            end
            s_valid[0] <= 1'b0;

            if (req_valid && req_ready) begin
                ba = {req_addr[63:3], 3'd0};
                rd = word_at(ba);
                n_accepted <= n_accepted + 1;
                if (req_amo && (ba + MEM_BYTES) <= BYTES) begin
                    amo_res = amo_new(req_aop, req_be, rd, req_wdata,
                                      req_cmp);
                    if (req_aop != `TNCA_AOP_LD)
                        for (j = 0; j < MEM_BYTES; j = j + 1)
                            if (req_be[j]) b[ba + j] <= amo_res[j*8 +: 8];
                end else if (req_we && (ba + MEM_BYTES) <= BYTES) begin
                    if (req_be != {MEM_BYTES{1'b1}})
                        n_partial <= n_partial + 1;
                    for (j = 0; j < MEM_BYTES; j = j + 1)
                        if (req_be[j]) b[ba + j] <= req_wdata[j*8 +: 8];
                end
                s_valid[0] <= 1'b1;
                s_tag[0]   <= req_tag;
                s_data[0]  <= rd;
            end

            // ---- retire --------------------------------------------------
            if (REORDER != 0) begin
                // Hold one back; let the next overtake it.
                if (s_valid[DEPTH-1] && !h_valid) begin
                    h_valid <= 1'b1;
                    h_tag   <= s_tag[DEPTH-1];
                    h_data  <= s_data[DEPTH-1];
                end else if (s_valid[DEPTH-1] && h_valid) begin
                    // The younger one goes first: that is the inversion.
                    rsp_valid <= 1'b1;
                    rsp_tag   <= s_tag[DEPTH-1];
                    rsp_rdata <= s_data[DEPTH-1];
                    n_swaps   <= n_swaps + 1;
                end else if (h_valid) begin
                    rsp_valid <= 1'b1;
                    rsp_tag   <= h_tag;
                    rsp_rdata <= h_data;
                    h_valid   <= 1'b0;
                end
            end else if (s_valid[DEPTH-1]) begin
                rsp_valid <= 1'b1;
                rsp_tag   <= s_tag[DEPTH-1];
                rsp_rdata <= s_data[DEPTH-1];
            end
        end
    end

    // ---- loading, for the simulation binding's `+hex=` ------------------------
    //
    // A word image, byte-scattered on the way in, so the two models take the
    // same file.
    task load_hex;
        input [1023:0] path;
        integer fh, n, idx;
        reg [63:0] w;
        begin
            fh = $fopen(path, "r");
            idx = 0;
            if (fh != 0) begin
                n = $fscanf(fh, "%h", w);
                while (n == 1 && idx < WORDS) begin
                    for (j = 0; j < MEM_BYTES; j = j + 1)
                        b[idx*MEM_BYTES + j] = w[j*8 +: 8];
                    idx = idx + 1;
                    n = $fscanf(fh, "%h", w);
                end
                $fclose(fh);
            end
        end
    endtask

endmodule
