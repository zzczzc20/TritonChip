//==============================================================================
// tnca_operand_collector -- gather an instruction's operands from a banked RF
//
// doc/phase3_tasks_and_acceptance.md R3-04. tnca_vrf_banked accepts one read
// per bank per cycle and says which requests it took; everything a request
// needs after that -- retrying the ones it did not take, holding the operands
// that did arrive, deciding when an instruction is ready, and being fair about
// whose turn it is -- lives here.
//
// An entry is one instruction's worth of work: up to SRCS source registers,
// each with its own "asked for it" and "got it" bit. The entry becomes ready
// when every source it declared has arrived, and not before. FLOW-01's "未收齐
// 不发射" is that condition and nothing else; there is no partial issue.
//
// FAIRNESS
//
//   One round-robin pointer over the entries, advanced past whoever was
//   served. That is the frozen policy FLOW-02 refers to: within any window of
//   ENTRIES consecutive grants, an entry with an outstanding request is served
//   at least once. It is a bound the design owes, so it is stated here rather
//   than left to be inferred from the code.
//
// REQUESTS ARE RETRIED, NOT LOST
//
//   A request the RF did not accept stays `asked=0` and is presented again.
//   `n_wait` counts the cycles an entry spent with an unserved request, which
//   is the number FLOW-02 wants -- "计数说明真实等待而非丢请求". A dropped
//   request would show up as a shorter wait, which is the wrong direction to
//   be wrong in.
//
// SQUASH AND EPOCH
//
//   `flush` clears entries that have not yet issued. It deliberately does NOT
//   clear `outstanding`: a read already accepted by the RF will produce a
//   response whether or not anyone still wants it, and the response has to be
//   absorbed rather than delivered to whatever occupies that slot next.
//   FLOW-05 asks for exactly that distinction, and FLOW-03's "已发事务可以完成"
//   is the same property from the other side.
//==============================================================================
`include "tnca_defs.vh"

module tnca_operand_collector #(
    parameter ENTRIES   = 4,
    parameter SRCS      = 3,
    parameter VLEN      = 16,
    parameter ELEN      = 32,
    parameter RCTX_W    = 1,
    parameter ENT_W     = 2,      // $clog2(ENTRIES)
    parameter NUM_PORTS = 4       // read ports on the RF
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- allocate an entry ------------------------------------------------
    input  wire                          alloc_valid,
    input  wire [ENT_W-1:0]              alloc_entry,
    input  wire [SRCS-1:0]               alloc_srcs,      // which are used
    input  wire [SRCS*8-1:0]             alloc_idx,
    input  wire [RCTX_W-1:0]             alloc_rctx,
    output wire [ENTRIES-1:0]            busy,

    // ---- to the register file --------------------------------------------
    output wire [NUM_PORTS-1:0]          req_valid,
    output wire [NUM_PORTS*RCTX_W-1:0]   req_rctx,
    output wire [NUM_PORTS*8-1:0]        req_idx,
    input  wire [NUM_PORTS-1:0]          req_accept,

    input  wire [NUM_PORTS-1:0]          rsp_valid,
    input  wire [NUM_PORTS*VLEN*ELEN-1:0] rsp_data,

    // ---- an entry whose operands are all present -------------------------
    output wire [ENTRIES-1:0]            ready,
    input  wire [ENTRIES-1:0]            consume,

    // ---- squash / fault / epoch ------------------------------------------
    input  wire                          flush,

    // ---- counters (R3-06 will read these) --------------------------------
    output reg  [31:0]                   n_wait,
    output reg  [31:0]                   n_grant,
    output reg  [31:0]                   n_conflict,
    output reg  [31:0]                   n_orphan
);

    localparam WIDTH = VLEN * ELEN;

    reg [ENTRIES-1:0]        e_busy;
    reg [SRCS-1:0]           e_need  [0:ENTRIES-1];
    reg [SRCS-1:0]           e_asked [0:ENTRIES-1];
    reg [SRCS-1:0]           e_have  [0:ENTRIES-1];
    reg [RCTX_W-1:0]         e_rctx  [0:ENTRIES-1];
    reg [SRCS*8-1:0]         e_idx   [0:ENTRIES-1];
    reg [SRCS*WIDTH-1:0]     e_data  [0:ENTRIES-1];

    assign busy = e_busy;

    // An accepted read carries its destination with it: which entry, which
    // source slot. FLOW-03/FLOW-04 both come down to this being reserved at
    // acceptance rather than looked up at response time.
    reg [ENT_W-1:0]  port_ent  [0:NUM_PORTS-1];
    reg [SRCS-1:0]   port_slot [0:NUM_PORTS-1];   // one-hot
    reg [NUM_PORTS-1:0] port_live;
    reg [NUM_PORTS-1:0] port_orphan;   // issued, but its entry was flushed

    reg [ENT_W-1:0] rr;     // the round-robin pointer

    // ---- pick up to NUM_PORTS outstanding sources ------------------------
    reg [NUM_PORTS-1:0]        rq_v;
    reg [NUM_PORTS*RCTX_W-1:0] rq_rctx;
    reg [NUM_PORTS*8-1:0]      rq_idx;
    reg [NUM_PORTS*ENT_W-1:0]  rq_ent;
    reg [NUM_PORTS*SRCS-1:0]   rq_slot;

    integer k, e, sidx, p;
    reg found;
    always @* begin
        rq_v    = {NUM_PORTS{1'b0}};
        rq_rctx = {NUM_PORTS*RCTX_W{1'b0}};
        rq_idx  = {NUM_PORTS*8{1'b0}};
        rq_ent  = {NUM_PORTS*ENT_W{1'b0}};
        rq_slot = {NUM_PORTS*SRCS{1'b0}};
        p = 0;
        // Walk the entries starting at rr, so the entry after the last one
        // served gets first refusal this cycle.
        for (k = 0; k < ENTRIES; k = k + 1) begin
            e = (rr + k) % ENTRIES;
            for (sidx = 0; sidx < SRCS; sidx = sidx + 1) begin
                if (p < NUM_PORTS && e_busy[e] && e_need[e][sidx]
                        && !e_asked[e][sidx]) begin
                    rq_v[p]                       = 1'b1;
                    rq_rctx[p*RCTX_W +: RCTX_W]   = e_rctx[e];
                    rq_idx[p*8 +: 8]              = e_idx[e][sidx*8 +: 8];
                    rq_ent[p*ENT_W +: ENT_W]      = e[ENT_W-1:0];
                    rq_slot[p*SRCS +: SRCS]       = (1 << sidx);
                    p = p + 1;
                end
            end
        end
    end

    assign req_valid = rq_v;
    assign req_rctx  = rq_rctx;
    assign req_idx   = rq_idx;

    // ---- ready ------------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < ENTRIES; gi = gi + 1) begin : g_ready
            assign ready[gi] = e_busy[gi] &&
                               (e_have[gi] == e_need[gi]);
        end
    endgenerate

    // ---- state ------------------------------------------------------------
    integer i, j, s2;
    reg [ENT_W-1:0] dst_e;
    reg [SRCS-1:0]  dst_s;
    reg             any_wait;
    // Counters are accumulated into blocking temporaries and assigned once.
    // `n_grant <= n_grant + 1` inside a loop over ports is three
    // non-blocking assignments to the same target in one evaluation, and
    // only the last survives: three grants in a cycle counted as one. A
    // counter that undercounts grants overcounts nothing and hides
    // contention, which is the opposite of what FLOW-02 asks the counters
    // to show.
    reg [31:0]      grants_c, conflicts_c, orphans_c;
    // Per-entry masks accumulated the same way and for the same reason.
    // Three ports serving one entry in one cycle are three non-blocking
    // assignments to e_asked[that entry], and only the last one survives:
    // two of the three sources would look unrequested and be asked for
    // again, which is how a three-operand read cost six grants.
    reg [SRCS-1:0]  asked_c [0:ENTRIES-1];
    reg [SRCS-1:0]  have_c  [0:ENTRIES-1];
    reg [ENTRIES-1:0] touch_asked, touch_have;
    always @(posedge clk) begin
        if (!rst_n) begin
            e_busy     <= {ENTRIES{1'b0}};
            port_live  <= {NUM_PORTS{1'b0}};
            port_orphan<= {NUM_PORTS{1'b0}};
            rr         <= {ENT_W{1'b0}};
            n_wait     <= 32'd0;
            n_grant    <= 32'd0;
            n_conflict <= 32'd0;
            n_orphan   <= 32'd0;
            for (i = 0; i < ENTRIES; i = i + 1) begin
                e_need[i]  <= {SRCS{1'b0}};
                e_asked[i] <= {SRCS{1'b0}};
                e_have[i]  <= {SRCS{1'b0}};
            end
        end else begin
            grants_c    = 32'd0;
            conflicts_c = 32'd0;
            orphans_c   = 32'd0;
            touch_asked = {ENTRIES{1'b0}};
            touch_have  = {ENTRIES{1'b0}};
            for (i = 0; i < ENTRIES; i = i + 1) begin
                asked_c[i] = e_asked[i];
                have_c[i]  = e_have[i];
            end
            // -- responses land first, so a response and a new request for
            //    the same slot in one cycle do not fight over it
            for (j = 0; j < NUM_PORTS; j = j + 1) begin
                if (rsp_valid[j] && port_live[j]) begin
                    dst_e = port_ent[j];
                    dst_s = port_slot[j];
                    if (port_orphan[j]) begin
                        // Its entry is gone. The response is absorbed and
                        // counted, never written into whatever now holds
                        // that slot.
                        orphans_c = orphans_c + 1;
                    end else begin
                        for (s2 = 0; s2 < SRCS; s2 = s2 + 1)
                            if (dst_s[s2]) begin
                                have_c[dst_e][s2] = 1'b1;
                                touch_have[dst_e] = 1'b1;
                                e_data[dst_e][s2*WIDTH +: WIDTH] <=
                                    rsp_data[j*WIDTH +: WIDTH];
                            end
                    end
                    port_live[j]   <= 1'b0;
                    port_orphan[j] <= 1'b0;
                end
            end

            // -- requests: accepted ones are marked asked and reserve a
            //    destination; refused ones stay outstanding and are counted
            any_wait = 1'b0;
            for (j = 0; j < NUM_PORTS; j = j + 1) begin
                if (rq_v[j]) begin
                    if (req_accept[j]) begin
                        asked_c[rq_ent[j*ENT_W +: ENT_W]] =
                            asked_c[rq_ent[j*ENT_W +: ENT_W]]
                            | rq_slot[j*SRCS +: SRCS];
                        touch_asked[rq_ent[j*ENT_W +: ENT_W]] = 1'b1;
                        port_ent[j]   <= rq_ent[j*ENT_W +: ENT_W];
                        port_slot[j]  <= rq_slot[j*SRCS +: SRCS];
                        port_live[j]  <= 1'b1;
                        port_orphan[j]<= 1'b0;
                        grants_c      = grants_c + 1;
                        rr            <= (rq_ent[j*ENT_W +: ENT_W] + 1)
                                         % ENTRIES;
                    end else begin
                        conflicts_c = conflicts_c + 1;
                        any_wait    = 1'b1;
                    end
                end
            end
            if (any_wait)
                n_wait <= n_wait + 1;
            for (i = 0; i < ENTRIES; i = i + 1) begin
                if (touch_asked[i]) e_asked[i] <= asked_c[i];
                if (touch_have[i])  e_have[i]  <= have_c[i];
            end
            n_grant    <= n_grant + grants_c;
            n_conflict <= n_conflict + conflicts_c;
            n_orphan   <= n_orphan + orphans_c;

            // -- allocate
            if (alloc_valid && !e_busy[alloc_entry] && !flush) begin
                e_busy[alloc_entry]  <= 1'b1;
                e_need[alloc_entry]  <= alloc_srcs;
                e_asked[alloc_entry] <= {SRCS{1'b0}};
                e_have[alloc_entry]  <= {SRCS{1'b0}};
                e_rctx[alloc_entry]  <= alloc_rctx;
                e_idx[alloc_entry]   <= alloc_idx;
            end

            // -- consume: the instruction took its operands
            for (i = 0; i < ENTRIES; i = i + 1)
                if (consume[i] && ready[i])
                    e_busy[i] <= 1'b0;

            // -- flush: entries go, reads already issued do not
            if (flush) begin
                e_busy <= {ENTRIES{1'b0}};
                for (i = 0; i < ENTRIES; i = i + 1) begin
                    e_need[i]  <= {SRCS{1'b0}};
                    e_asked[i] <= {SRCS{1'b0}};
                    e_have[i]  <= {SRCS{1'b0}};
                end
                // Every read the RF has already accepted will still answer.
                // Marking it an orphan is what makes "no silent overwrite"
                // true rather than hoped for.
                for (j = 0; j < NUM_PORTS; j = j + 1)
                    if (port_live[j])
                        port_orphan[j] <= 1'b1;
            end
        end
    end

    // ---- the collected operands, for whoever consumes the entry ----------
    wire [ENTRIES*SRCS*WIDTH-1:0] operands;
    genvar ge;
    generate
        for (ge = 0; ge < ENTRIES; ge = ge + 1) begin : g_ops
            assign operands[ge*SRCS*WIDTH +: SRCS*WIDTH] = e_data[ge];
        end
    endgenerate

`ifndef SYNTHESIS
    // FLOW-04: request/response conservation. A response may only arrive on
    // a port that has one outstanding.
    integer ai;
    always @(posedge clk) begin
        if (rst_n) begin
            for (ai = 0; ai < NUM_PORTS; ai = ai + 1)
                if (rsp_valid[ai] && !port_live[ai]) begin
                    $display("COLLECT-ASSERT: response on port %0d with nothing outstanding", ai);
                    $finish;
                end
        end
    end
`endif

endmodule
