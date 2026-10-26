//==============================================================================
// tnca_axi_bridge -- the device's data port, split between on-chip RAM and an
// AXI4 memory (HBM on the U280)
//
// doc/r40_tasks_and_acceptance.md R40-04: real external memory behind the
// same port the core has always used, without changing what that port means.
//
// THE PORT, AND WHAT IT PROMISES (rtl/fpga/tnca_fpga_mem.v, PORTS / LATENCY /
// READ-DURING-WRITE / ATOMICS)
//   valid/ready requests with a tag, byte enables and atomics; a response for
//   every request, carrying the tag and -- for a read or an atomic -- the word
//   as it stood when the request was accepted; responses are never back-
//   pressured. The memories behind this port apply each request's effect in
//   the order they ACCEPT requests; responses may come back in any order
//   (tb/common/tnca_mem_model.v MEM_REORDER, the core-ooo regression).
//
// ROUTING
//   An address below LOCAL_LIMIT goes to the on-chip RAM (`l_*`): the code
//   region and anything else the runtime keeps below the heap, so fetch and
//   data still share one array there, as they always did. An address at or
//   above LOCAL_LIMIT is an offset into the AXI window: AXI address
//   axi_base + (addr - LOCAL_LIMIT). Past axi_size it is answered as the RAM
//   answers past its end -- a read gives zero and a write stores nothing.
//
// ORDER ON AXI
//   AXI orders nothing between its read and write channels, and the port
//   promises acceptance order. So every AXI request in flight is in a small
//   table (OUT_N entries: word address, read or write, tag), and a request
//   that would race one in the table -- same 8-byte word, and either of the
//   two a write -- waits until it has completed. Two reads of one word may
//   overlap. Writes go out on one AXI ID and so complete in issue order; a FIFO
//   of their tags maps each B response back. Reads use the tag as their AXI ID
//   and may complete in any order, which the port allows. Each access is a
//   single 8-byte beat (INCR, length 0): this is the capacity step; bursts and
//   a wider ingress for the transfer engine are separate R40-04 mechanisms,
//   each with its own switch and ablation.
//
// ATOMICS ON AXI
//   AXI4 has no atomic that does what tnca_amo.vh defines, so an atomic is a
//   read-modify-write done here: the table drains, the word is read, the new
//   value computed (amo_new, the same function the RAM and the home slice
//   use), written back unless the operation is a load, and the OLD word
//   returned once the write has completed. Nothing else is accepted
//   meanwhile, so no access can fall between the read and the write.
//
// RESPONSES
//   The RAM's responses come at a fixed latency and cannot wait, so they win
//   the response port; an AXI read or write response, an out-of-range answer
//   or an atomic's result waits for a free cycle (rready/bready low), which
//   AXI permits.
//
// ERRORS AND RESET
//   A SLVERR or DECERR answers its request like any other response (a read
//   with whatever data came back) and sets axi_err, which stays set until
//   reset; the host reads it in R_AXI_STAT after a launch. A reset forgets
//   what was in flight, but the AXI side may not have been reset with us and
//   can still answer it: for QUIET cycles after reset nothing is accepted and
//   every R and B is taken and dropped, so an old answer cannot be mistaken
//   for a new request's. One that arrives later than that, or any R or B
//   that matches nothing in flight, is dropped too and sets axi_stray. The
//   request side cannot be undone the same way -- an AW taken without its W
//   is the interconnect's to clear -- so the platform resets the interconnect
//   with this module (on the U280 both are reset when an image is loaded and
//   at no other time).
//
// COUNTERS
//   Free-running from reset, for the host to difference: AXI reads and writes
//   issued (an atomic counts one of each, or one read if it only loads), and
//   the sum over cycles of the reads in flight -- by Little's law, divided by
//   the reads, their mean latency in cycles from AR to R. They are what says
//   a graph's traffic went to HBM at all, and the latency the memory model is
//   calibrated against (doc/r40_tasks_and_acceptance.md sec 3.4).
//==============================================================================
`default_nettype none

module tnca_axi_bridge #(
    parameter TAG_W       = 5,
    parameter OUT_N       = 8,
    parameter LOCAL_LIMIT = 64'h20000,
    parameter AXI_ID_W    = TAG_W,
    parameter QUIET       = 1024
) (
    input  wire              clk,
    input  wire              rst_n,

    // the core's data port (from tnca_shell)
    input  wire              req_valid,
    output wire              req_ready,
    input  wire [63:0]       req_addr,
    input  wire              req_we,
    input  wire [7:0]        req_be,
    input  wire [63:0]       req_wdata,
    input  wire [TAG_W-1:0]  req_tag,
    input  wire              req_amo,
    input  wire [3:0]        req_aop,
    input  wire [63:0]       req_cmp,
    output reg               rsp_valid,
    output reg  [TAG_W-1:0]  rsp_tag,
    output reg  [63:0]       rsp_rdata,

    // the on-chip RAM's data port (rtl/fpga/tnca_fpga_mem.v)
    output wire              l_req_valid,
    input  wire              l_req_ready,
    output wire [63:0]       l_req_addr,
    output wire              l_req_we,
    output wire [7:0]        l_req_be,
    output wire [63:0]       l_req_wdata,
    output wire [TAG_W-1:0]  l_req_tag,
    output wire              l_req_amo,
    output wire [3:0]        l_req_aop,
    output wire [63:0]       l_req_cmp,
    input  wire              l_rsp_valid,
    input  wire [TAG_W-1:0]  l_rsp_tag,
    input  wire [63:0]       l_rsp_rdata,

    // the AXI window (set by the host before any launch)
    input  wire [63:0]       axi_base,
    input  wire [63:0]       axi_size,
    input  wire              inv,          // a command was served (unused: nothing buffered here)
    output reg               axi_err,      // sticky: a SLVERR or DECERR came back
    output reg               axi_stray,    // sticky: an R or B matched nothing in flight
    output reg  [63:0]       cnt_rd,       // AXI reads issued (AR handshakes)
    output reg  [63:0]       cnt_wr,       // AXI writes issued (AW handshakes)
    output reg  [63:0]       cnt_rd_lat,   // sum over cycles of reads in flight
    output reg  [63:0]       cnt_rbeat,    // R beats taken (here, one per read)

    // AXI4 master, 64-bit data
    output reg               m_awvalid,
    input  wire              m_awready,
    output reg  [63:0]       m_awaddr,
    output wire [AXI_ID_W-1:0] m_awid,
    output wire [7:0]        m_awlen,
    output wire [2:0]        m_awsize,
    output wire [1:0]        m_awburst,
    output reg               m_wvalid,
    input  wire              m_wready,
    output reg  [63:0]       m_wdata,
    output reg  [7:0]        m_wstrb,
    output wire              m_wlast,
    input  wire              m_bvalid,
    output wire              m_bready,
    input  wire [AXI_ID_W-1:0] m_bid,
    input  wire [1:0]        m_bresp,
    output reg               m_arvalid,
    input  wire              m_arready,
    output reg  [63:0]       m_araddr,
    output reg  [AXI_ID_W-1:0] m_arid,
    output wire [7:0]        m_arlen,
    output wire [2:0]        m_arsize,
    output wire [1:0]        m_arburst,
    input  wire              m_rvalid,
    output wire              m_rready,
    input  wire [63:0]       m_rdata,
    input  wire [AXI_ID_W-1:0] m_rid,
    input  wire [1:0]        m_rresp,
    input  wire              m_rlast
);
`include "tnca_amo.vh"
    wire unused_inv = inv;
    assign m_awid    = {AXI_ID_W{1'b0}};
    assign m_awlen   = 8'd0;
    assign m_awsize  = 3'd3;
    assign m_awburst = 2'b01;
    assign m_wlast   = 1'b1;
    assign m_arlen   = 8'd0;
    assign m_arsize  = 3'd3;
    assign m_arburst = 2'b01;

    localparam CNT_W = $clog2(OUT_N + 1);

    //--------------------------------------------------------------------------
    // Routing of the request in front of us
    //--------------------------------------------------------------------------
    wire        is_local = req_addr < LOCAL_LIMIT;
    wire [63:0] off      = req_addr - LOCAL_LIMIT;
    wire        in_range = off < axi_size;
    wire [63:0] word     = {req_addr[63:3], 3'b000};
    wire [63:0] axi_addr = axi_base + {off[63:3], 3'b000};

    //--------------------------------------------------------------------------
    // The table of AXI accesses in flight
    //--------------------------------------------------------------------------
    reg              t_v  [0:OUT_N-1];
    reg              t_w  [0:OUT_N-1];
    reg [63:0]       t_a  [0:OUT_N-1];
    reg [TAG_W-1:0]  t_tag[0:OUT_N-1];
    reg [CNT_W-1:0]  t_n;

    reg hazard;
    reg have_free;
    reg [$clog2(OUT_N)-1:0] free_i;
    integer k;
    always @(*) begin
        hazard = 1'b0;
        have_free = 1'b0;
        free_i = 0;
        for (k = OUT_N - 1; k >= 0; k = k - 1) begin
            if (t_v[k] && t_a[k] == word && (t_w[k] || req_we || req_amo))
                hazard = 1'b1;
            if (!t_v[k]) begin
                have_free = 1'b1;
                free_i = k;
            end
        end
    end

    // write tags, in issue order (one AXI ID: B responses come in that order)
    reg [TAG_W-1:0] wq [0:OUT_N-1];
    reg [$clog2(OUT_N)-1:0] wq_head, wq_tail;

    //--------------------------------------------------------------------------
    // Pending answers that do not come from AXI or the RAM
    //--------------------------------------------------------------------------
    reg              null_v;           // an out-of-range access
    reg [TAG_W-1:0]  null_tag;
    reg              amo_rsp_v;        // an atomic's old word
    reg [TAG_W-1:0]  amo_rsp_tag;
    reg [63:0]       amo_rsp_data;

    //--------------------------------------------------------------------------
    // The atomic state machine
    //--------------------------------------------------------------------------
    localparam A_IDLE = 3'd0, A_DRAIN = 3'd1, A_AR = 3'd2, A_R = 3'd3,
               A_W = 3'd4, A_B = 3'd5;
    reg [2:0]        a_st;
    reg [63:0]       a_addr;
    reg [7:0]        a_be;
    reg [63:0]       a_wdata, a_cmp, a_old;
    reg [3:0]        a_op;
    reg [TAG_W-1:0]  a_tag;
    reg              a_aw_done, a_w_done;

    //--------------------------------------------------------------------------
    // A write is held until both its AW and its W have been taken
    //--------------------------------------------------------------------------
    wire w_hold = m_awvalid | m_wvalid;

    // While an atomic drains, the accesses ahead of it still complete through
    // the response port; from its own read on, R and B are the atomic's.
    wire amo_own = (a_st == A_AR) || (a_st == A_R) || (a_st == A_W) || (a_st == A_B);

    //--------------------------------------------------------------------------
    // After reset: answers to requests we no longer remember
    //--------------------------------------------------------------------------
    localparam Q_W = $clog2(QUIET + 1);
    reg [CNT_W:0] rd_fly;           // reads issued and not yet answered
    reg [Q_W-1:0] quiet;
    wire quiet_now = quiet != 0;

    // Whether an R or a B in front of us answers something in the table.
    // While an atomic owns the channels its read is the only one in flight
    // (the table has drained), under its own tag.
    reg r_match, b_match;
    integer m;
    always @(*) begin
        r_match = 1'b0;
        b_match = 1'b0;
        for (m = 0; m < OUT_N; m = m + 1) begin
            if (t_v[m] && !t_w[m] && t_tag[m] == m_rid[TAG_W-1:0]) r_match = 1'b1;
            if (t_v[m] && t_w[m] && t_tag[m] == wq[wq_head])        b_match = 1'b1;
        end
    end
    // dropped: taken from AXI, never forwarded
    wire drop_r = m_rvalid && (quiet_now || (amo_own ? m_rid[TAG_W-1:0] != a_tag : !r_match));
    wire drop_b = m_bvalid && (quiet_now || (!amo_own && !b_match));

    //--------------------------------------------------------------------------
    // Acceptance
    //--------------------------------------------------------------------------
    wire axi_path_free = (a_st == A_IDLE) && !amo_rsp_v;
    wire can_read  = axi_path_free && !hazard && have_free && !m_arvalid;
    wire can_write = axi_path_free && !hazard && have_free && !w_hold;
    wire can_amo   = axi_path_free && !hazard;
    wire can_null  = !null_v;

    assign req_ready = quiet_now ? 1'b0
                     : is_local ? l_req_ready
                     : !in_range ? can_null
                     : req_amo ? can_amo
                     : req_we ? can_write : can_read;
    wire acc       = req_valid && req_ready;
    wire acc_axi   = acc && !is_local && in_range;

    assign l_req_valid = req_valid && is_local && !quiet_now;
    assign l_req_addr  = req_addr;
    assign l_req_we    = req_we;
    assign l_req_be    = req_be;
    assign l_req_wdata = req_wdata;
    assign l_req_tag   = req_tag;
    assign l_req_amo   = req_amo;
    assign l_req_aop   = req_aop;
    assign l_req_cmp   = req_cmp;

    //--------------------------------------------------------------------------
    // Responses: the RAM first, then whatever is waiting
    //--------------------------------------------------------------------------
    wire take_null = !l_rsp_valid && null_v;
    wire take_amo  = !l_rsp_valid && !null_v && amo_rsp_v;
    wire take_r    = !l_rsp_valid && !null_v && !amo_rsp_v && m_rvalid && !amo_own && !drop_r;
    wire take_b    = !l_rsp_valid && !null_v && !amo_rsp_v && !m_rvalid && m_bvalid && !amo_own
                     && !drop_b;
    assign m_rready = drop_r || (amo_own ? (a_st == A_R) : take_r);
    assign m_bready = drop_b || (amo_own ? (a_st == A_B) : take_b);

    always @(*) begin
        rsp_valid = 1'b0;
        rsp_tag   = {TAG_W{1'b0}};
        rsp_rdata = 64'd0;
        if (l_rsp_valid) begin
            rsp_valid = 1'b1; rsp_tag = l_rsp_tag; rsp_rdata = l_rsp_rdata;
        end else if (take_null) begin
            rsp_valid = 1'b1; rsp_tag = null_tag; rsp_rdata = 64'd0;
        end else if (take_amo) begin
            rsp_valid = 1'b1; rsp_tag = amo_rsp_tag; rsp_rdata = amo_rsp_data;
        end else if (take_r) begin
            rsp_valid = 1'b1; rsp_tag = m_rid[TAG_W-1:0]; rsp_rdata = m_rdata;
        end else if (take_b) begin
            rsp_valid = 1'b1; rsp_tag = wq[wq_head]; rsp_rdata = 64'd0;
        end
    end

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < OUT_N; i = i + 1) t_v[i] <= 1'b0;
            t_n <= 0;
            wq_head <= 0; wq_tail <= 0;
            null_v <= 1'b0; amo_rsp_v <= 1'b0;
            a_st <= A_IDLE; a_aw_done <= 1'b0; a_w_done <= 1'b0;
            m_awvalid <= 1'b0; m_wvalid <= 1'b0; m_arvalid <= 1'b0;
            axi_err <= 1'b0; axi_stray <= 1'b0;
            quiet <= QUIET;
            cnt_rd <= 64'd0; cnt_wr <= 64'd0; cnt_rd_lat <= 64'd0; cnt_rbeat <= 64'd0; rd_fly <= 0;
        end else begin
            // ---- counters ----------------------------------------------------
            if (m_arvalid && m_arready) cnt_rd <= cnt_rd + 64'd1;
            if (m_awvalid && m_awready) cnt_wr <= cnt_wr + 64'd1;
            if (m_rvalid && m_rready && !drop_r) cnt_rbeat <= cnt_rbeat + 64'd1;
            cnt_rd_lat <= cnt_rd_lat + rd_fly;
            rd_fly <= rd_fly + ((m_arvalid && m_arready) ? 1 : 0)
                             - ((m_rvalid && m_rready && !drop_r) ? 1 : 0);

            if (quiet_now) quiet <= quiet - 1'b1;
            if ((drop_r || drop_b) && !quiet_now) axi_stray <= 1'b1;

            // ---- channels completing --------------------------------------
            if (m_arvalid && m_arready) m_arvalid <= 1'b0;
            if (m_awvalid && m_awready) m_awvalid <= 1'b0;
            if (m_wvalid && m_wready)   m_wvalid  <= 1'b0;

            if (take_null) null_v <= 1'b0;
            if (take_amo)  amo_rsp_v <= 1'b0;

            if (m_rvalid && m_rready && !drop_r) begin
                if (m_rresp[1]) axi_err <= 1'b1;
                if (!amo_own) begin
                    for (i = 0; i < OUT_N; i = i + 1)
                        if (t_v[i] && !t_w[i] && t_tag[i] == m_rid[TAG_W-1:0])
                            t_v[i] <= 1'b0;
                end
            end
            if (m_bvalid && m_bready && !drop_b) begin
                if (m_bresp[1]) axi_err <= 1'b1;
                if (!amo_own) begin
                    for (i = 0; i < OUT_N; i = i + 1)
                        if (t_v[i] && t_w[i] && t_tag[i] == wq[wq_head])
                            t_v[i] <= 1'b0;
                    wq_head <= wq_head + 1'b1;
                end
            end

            // ---- a new request ----------------------------------------------
            if (acc && !is_local && !in_range) begin
                null_v   <= 1'b1;
                null_tag <= req_tag;
            end
            if (acc_axi && !req_amo) begin
                t_v[free_i]   <= 1'b1;
                t_w[free_i]   <= req_we;
                t_a[free_i]   <= word;
                t_tag[free_i] <= req_tag;
                if (req_we) begin
                    m_awvalid <= 1'b1; m_awaddr <= axi_addr;
                    m_wvalid  <= 1'b1; m_wdata  <= req_wdata; m_wstrb <= req_be;
                    wq[wq_tail] <= req_tag;
                    wq_tail <= wq_tail + 1'b1;
                end else begin
                    m_arvalid <= 1'b1; m_araddr <= axi_addr; m_arid <= req_tag;
                end
            end
            if (acc_axi && req_amo) begin
                a_st    <= A_DRAIN;
                a_addr  <= axi_addr;
                a_be    <= req_be;
                a_wdata <= req_wdata;
                a_cmp   <= req_cmp;
                a_op    <= req_aop;
                a_tag   <= req_tag;
            end

            // ---- the atomic -------------------------------------------------
            case (a_st)
            A_DRAIN: begin
                // nothing of ours may be in flight: the read and the write
                // must be the only two accesses to that word, back to back
                if (!m_arvalid && !w_hold && t_n == 0) begin
                    m_arvalid <= 1'b1; m_araddr <= a_addr; m_arid <= a_tag;
                    a_st <= A_AR;
                end
            end
            A_AR: if (m_arvalid && m_arready) a_st <= A_R;
            A_R: if (m_rvalid && !drop_r) begin
                a_old <= m_rdata;
                if (a_op == `TNCA_AOP_LD) begin
                    amo_rsp_v <= 1'b1; amo_rsp_tag <= a_tag; amo_rsp_data <= m_rdata;
                    a_st <= A_IDLE;
                end else begin
                    m_awvalid <= 1'b1; m_awaddr <= a_addr;
                    m_wvalid  <= 1'b1; m_wstrb  <= a_be;
                    m_wdata   <= amo_new(a_op, a_be, m_rdata, a_wdata, a_cmp);
                    a_st <= A_W;
                end
            end
            A_W: if (!(m_awvalid && !m_awready) && !(m_wvalid && !m_wready)) a_st <= A_B;
            A_B: if (m_bvalid && !drop_b) begin
                amo_rsp_v <= 1'b1; amo_rsp_tag <= a_tag; amo_rsp_data <= a_old;
                a_st <= A_IDLE;
            end
            default: ;
            endcase

            // ---- occupancy of the table -------------------------------------
            t_n <= t_n + ((acc_axi && !req_amo) ? 1 : 0)
                       - ((m_rvalid && m_rready && !amo_own && !drop_r) ? 1 : 0)
                       - ((m_bvalid && m_bready && !amo_own && !drop_b) ? 1 : 0);
        end
    end
endmodule
`default_nettype wire
