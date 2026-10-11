//==============================================================================
// tnca_axi_mem_model -- an AXI4 memory slave for simulation (R40-04)
//
// The memory behind rtl/fpga/tnca_axi_bridge.v and tnca_axi_bridge_lb.v in the
// simulated device and in tests: WORDS 64-bit words from BASE, 8-byte INCR
// transactions -- single-beat writes, reads of ARLEN+1 beats -- and timing as
// unhelpful as AXI allows, under plusargs:
//
//   +axi_lat_min=N +axi_lat_max=M   cycles from a channel's acceptance to its
//                                   response (default 1..1)
//   +axi_reorder                    pending reads answered in random order,
//                                   and their beats interleaved (AXI allows
//                                   both across IDs)
//   +axi_bp=P                       each READY dropped with probability P %
//   +axi_seed=S
//
// A read's data is sampled when its R beat is sent and a write applied when
// its B is sent, each as late as the protocol permits, so a requester that
// relies on an order AXI does not promise sees a wrong value. Reads of an
// address outside the memory return zero and writes there are dropped; every
// response is OKAY. Like tnca_mem_model.v, the array starts at zero.
//==============================================================================
`default_nettype none

module tnca_axi_mem_model #(
    parameter WORDS = 1 << 20,
    parameter [63:0] BASE = 64'h0,
    parameter ID_W  = 5,
    parameter QN    = 32
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            awvalid,
    output reg             awready,
    input  wire [63:0]     awaddr,
    input  wire [ID_W-1:0] awid,
    input  wire            wvalid,
    output reg             wready,
    input  wire [63:0]     wdata,
    input  wire [7:0]      wstrb,
    output reg             bvalid,
    input  wire            bready,
    output reg  [ID_W-1:0] bid,
    output wire [1:0]      bresp,
    input  wire            arvalid,
    output reg             arready,
    input  wire [63:0]     araddr,
    input  wire [ID_W-1:0] arid,
    input  wire [7:0]      arlen,
    output reg             rvalid,
    input  wire            rready,
    output reg  [63:0]     rdata,
    output reg  [ID_W-1:0] rid,
    output wire [1:0]      rresp,
    output reg             rlast
);
    assign bresp = 2'b00;
    assign rresp = 2'b00;

    reg [63:0] mem [0:WORDS-1];
    integer seed, lat_min, lat_max, bp, reorder;
    integer i;
    initial begin
        if (!$value$plusargs("axi_seed=%d", seed)) seed = 1;
        if (!$value$plusargs("axi_lat_min=%d", lat_min)) lat_min = 1;
        if (!$value$plusargs("axi_lat_max=%d", lat_max)) lat_max = lat_min;
        if (!$value$plusargs("axi_bp=%d", bp)) bp = 0;
        reorder = $test$plusargs("axi_reorder");
        for (i = 0; i < WORDS; i = i + 1) mem[i] = 64'd0;
    end

    function inside;
        input [63:0] a;
        begin inside = (a >= BASE) && (((a - BASE) >> 3) < WORDS); end
    endfunction

    function [63:0] merge;
        input [63:0] old, data; input [7:0] be;
        integer b;
        begin
            merge = old;
            for (b = 0; b < 8; b = b + 1) if (be[b]) merge[b*8 +: 8] = data[b*8 +: 8];
        end
    endfunction

    function integer lat;
        input integer dummy;
        begin lat = lat_min + (($random(seed) & 32'h7fffffff) % (lat_max - lat_min + 1)); end
    endfunction

    function rnd_ready;
        input integer dummy;
        begin rnd_ready = (($random(seed) & 32'h7fffffff) % 100) >= bp; end
    endfunction

    reg [63:0]     pr_addr [0:QN-1]; reg [ID_W-1:0] pr_id [0:QN-1]; integer pr_t [0:QN-1]; integer pr_n;
    integer        pr_len [0:QN-1]; integer pr_beat [0:QN-1];
    reg [63:0]     ba;
    reg [63:0]     aw_q [0:QN-1]; reg [ID_W-1:0] aw_id [0:QN-1]; integer aw_n;
    reg [63:0]     w_d [0:QN-1]; reg [7:0] w_s [0:QN-1]; integer w_n;
    reg [63:0]     pw_addr [0:QN-1]; reg [63:0] pw_data [0:QN-1]; reg [7:0] pw_strb [0:QN-1];
    reg [ID_W-1:0] pw_id [0:QN-1]; integer pw_t [0:QN-1]; integer pw_n;
    integer j, pick;

    always @(posedge clk) begin
        if (!rst_n) begin
            pr_n = 0; aw_n = 0; w_n = 0; pw_n = 0;
            rvalid <= 1'b0; bvalid <= 1'b0; rlast <= 1'b0;
            arready <= 1'b0; awready <= 1'b0; wready <= 1'b0;
        end else begin
            if (arvalid && arready) begin
                pr_addr[pr_n] = araddr; pr_id[pr_n] = arid; pr_t[pr_n] = lat(0);
                pr_len[pr_n] = arlen; pr_beat[pr_n] = 0; pr_n = pr_n + 1;
            end
            if (awvalid && awready) begin aw_q[aw_n] = awaddr; aw_id[aw_n] = awid; aw_n = aw_n + 1; end
            if (wvalid && wready) begin w_d[w_n] = wdata; w_s[w_n] = wstrb; w_n = w_n + 1; end
            while (aw_n > 0 && w_n > 0) begin
                pw_addr[pw_n] = aw_q[0]; pw_id[pw_n] = aw_id[0];
                pw_data[pw_n] = w_d[0]; pw_strb[pw_n] = w_s[0]; pw_t[pw_n] = lat(0);
                pw_n = pw_n + 1;
                for (j = 1; j < aw_n; j = j + 1) begin aw_q[j-1] = aw_q[j]; aw_id[j-1] = aw_id[j]; end
                for (j = 1; j < w_n; j = j + 1) begin w_d[j-1] = w_d[j]; w_s[j-1] = w_s[j]; end
                aw_n = aw_n - 1; w_n = w_n - 1;
            end
            if (rvalid && rready) rvalid <= 1'b0;
            if (!rvalid || rready) begin
                for (j = 0; j < pr_n; j = j + 1) if (pr_t[j] > 0) pr_t[j] = pr_t[j] - 1;
                pick = -1;
                for (j = 0; j < pr_n; j = j + 1)
                    if (pr_t[j] == 0 && (pick < 0 || (reorder && ($random(seed) & 1)))) pick = j;
                if (pick >= 0) begin
                    ba = pr_addr[pick] + pr_beat[pick] * 8;
                    rvalid <= 1'b1; rid <= pr_id[pick];
                    rdata <= inside(ba) ? mem[(ba - BASE) >> 3] : 64'd0;
                    rlast <= pr_beat[pick] == pr_len[pick];
                    if (pr_beat[pick] < pr_len[pick]) begin
                        pr_beat[pick] = pr_beat[pick] + 1;
                    end else begin
                        for (j = pick + 1; j < pr_n; j = j + 1) begin
                            pr_addr[j-1] = pr_addr[j]; pr_id[j-1] = pr_id[j]; pr_t[j-1] = pr_t[j];
                            pr_len[j-1] = pr_len[j]; pr_beat[j-1] = pr_beat[j];
                        end
                        pr_n = pr_n - 1;
                    end
                end
            end
            if (bvalid && bready) bvalid <= 1'b0;
            if (!bvalid || bready) begin
                for (j = 0; j < pw_n; j = j + 1) if (pw_t[j] > 0) pw_t[j] = pw_t[j] - 1;
                if (pw_n > 0 && pw_t[0] == 0) begin
                    if (inside(pw_addr[0]))
                        mem[(pw_addr[0] - BASE) >> 3] = merge(mem[(pw_addr[0] - BASE) >> 3],
                                                             pw_data[0], pw_strb[0]);
                    bvalid <= 1'b1; bid <= pw_id[0];
                    for (j = 1; j < pw_n; j = j + 1) begin
                        pw_addr[j-1] = pw_addr[j]; pw_data[j-1] = pw_data[j];
                        pw_strb[j-1] = pw_strb[j]; pw_id[j-1] = pw_id[j]; pw_t[j-1] = pw_t[j];
                    end
                    pw_n = pw_n - 1;
                end
            end
            arready <= rnd_ready(0) && pr_n < QN - 2;
            awready <= rnd_ready(0) && aw_n < QN - 2;
            wready  <= rnd_ready(0) && w_n < QN - 2;
        end
    end
endmodule
`default_nettype wire
