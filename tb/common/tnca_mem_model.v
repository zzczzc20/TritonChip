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
    // Atomics (ISA sec 9.2). In the four-tile system these execute at the home
    // slice; here there is no slice, so the model performs them -- from the
    // same rtl/common/tnca_amo.vh the slice uses, because two definitions that
    // disagreed would mean one of the two test suites was checking the wrong
    // machine.
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [MEM_DW-1:0]        req_cmp,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [MEM_DW-1:0]        rsp_rdata
);

`include "tnca_amo.vh"

    localparam POOL = 16;

    reg [MEM_DW-1:0] mem [0:WORDS-1];

    integer i;
    initial begin
        for (i = 0; i < WORDS; i = i + 1) mem[i] = {MEM_DW{1'b0}};
    end

    // ---- instruction port ---------------------------------------------------
    //
    // Synchronous, like the SRAM it stands for: the address presented this
    // cycle returns its word on the next. tnca_core carries the extra stage
    // and a one-deep fetch buffer for it.
    wire [63:0] iw = imem_addr >> 3;
    reg  [63:0] imem_q;
    always @(posedge clk)
        imem_q <= (iw < WORDS) ? mem[iw[31:0]] : 64'd0;
    assign imem_data = imem_q;

    // ---- pseudo-random source ------------------------------------------------
    //
    // The seed is a parameter AND a plusarg. R1-08 REG-04 asks for the slow
    // and out-of-order models to run under three seeds, and a parameter alone
    // would mean three rebuilds to change a number that controls nothing but
    // stimulus. `+memseed=` reaches this LFSR, which is the thing that decides
    // both the back-pressure pattern and which eligible response is retired
    // first -- so a seed that did not reach here would be a seed in a log
    // line and nowhere else.
    reg [31:0] lfsr;
    reg [31:0] seed_q;
    integer    seed_arg;
    // Counted so a run can show what the seed actually changed. "Three seeds"
    // with three identical stall counts is one test run three times.
    integer    n_stall_cycles, n_reorder, n_multi_eligible;
    reg [63:0] mem_trace;

    initial begin
        seed_q = SEED;
        if ($value$plusargs("memseed=%d", seed_arg))
            seed_q = (seed_arg[31:0] == 32'd0) ? 32'd1 : seed_arg[31:0];
        n_stall_cycles   = 0;
        n_reorder        = 0;
        n_multi_eligible = 0;
        mem_trace      = 64'hcbf29ce484222325;
    end

    always @(posedge clk) begin
        if (!rst_n) lfsr <= seed_q;
        else        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    assign req_ready = (RANDOM_STALL != 0) ? lfsr[3] : 1'b1;

    // The SCHEDULE, not the addresses. A deterministic program issues the
    // same addresses in the same order whatever the seed, so hashing those
    // gave three identical digests for three different runs -- a trace hash
    // that proved the program was deterministic and said nothing about the
    // seed. What the seed changes is when a request is accepted and which
    // eligible response is retired, so that is what is hashed.
    always @(posedge clk) begin
        if (rst_n) begin
            if (req_valid && !req_ready)
                n_stall_cycles <= n_stall_cycles + 1;
            if (req_valid || rsp_valid)
                mem_trace <= (mem_trace ^ {45'd0, req_valid, req_ready,
                                           rsp_valid, rsp_tag})
                             * 64'd1099511628211;
        end
    end

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
    reg [63:0]       amo_res;

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
                if (req_amo && in_range) begin
                    // Accepting is one clock edge, so the read-modify-write is
                    // indivisible here by construction; what it must match is
                    // the slice's arithmetic, and it does because it is the
                    // same function.
                    amo_res = amo_new(req_aop, req_be, rd, req_wdata, req_cmp);
                    if (req_aop != `TNCA_AOP_LD)
                        for (k = 0; k < MEM_BYTES; k = k + 1)
                            if (req_be[k])
                                mem[dw[31:0]][k*8 +: 8] <= amo_res[k*8 +: 8];
                    // `rd` already holds the whole old word, which is what the
                    // response carries: the requester extracts its bytes from
                    // it exactly as it would from a load.
                end else if (req_we && in_range) begin
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
                    // Under REORDER the latency VARIES. With a fixed one,
                    // requests accepted on different cycles expire on
                    // different cycles, so at most one entry was ever
                    // eligible and the "pick a random eligible entry" branch
                    // below picked the only one there was: the out-of-order
                    // model had never once returned anything out of order.
                    // n_reorder stayed at 0 on every run, which is what
                    // R1-08's "must report the count actually triggered"
                    // exists to catch.
                    p_cnt[slot]   <= (LATENCY == 0) ? 8'd0 :
                                     (REORDER != 0)
                                       ? ((LATENCY - 1) + {5'd0, lfsr[10:8]})
                                       : (LATENCY - 1);
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

            if (nelig > 1) n_multi_eligible <= n_multi_eligible + 1;
            if (nelig > 0) begin
                if (REORDER != 0) begin
                    pick = lfsr[7:0] % nelig;     // pseudo-random eligible entry
                    if (pick != 0) n_reorder <= n_reorder + 1;
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

    // A RANGE, for the persistent device: a host upload replaces part of
    // memory and must leave the rest -- including code that is already
    // there and results a previous kernel wrote -- alone. `load_hex` reads a
    // whole image and is right for a one-shot bench and wrong here.
    task load_range;
        input [1023:0] path;
        input integer first;
        input integer count;
        begin
            $readmemh(path, mem, first, first + count - 1);
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
