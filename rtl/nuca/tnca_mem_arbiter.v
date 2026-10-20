//==============================================================================
// tnca_mem_arbiter -- round-robin multiplexer from N slice memory ports onto
// one line-granular memory port
//
// Each slice's tag is only unique within that slice, so the arbiter carries the
// slice index alongside it and uses it to steer the response back. The memory
// model returns both untouched.
//
// STAGE-3 SIMPLIFICATION: memory controllers sit below the slices on a private
// path rather than on the mesh edge. Arch overview sec 10.3 wants every byte
// that crosses a link counted, and DRAM traffic on the mesh changes those
// numbers; putting the controllers on the mesh is Stage-4 work, together with
// the per-link bandwidth model. Nothing about the slice or the cache protocol
// depends on which it is.
//==============================================================================

module tnca_mem_arbiter #(
    parameter N         = 4,
    parameter N_W       = 2,
    parameter ADDR_W    = 64,
    parameter LINE_BITS = 512,
    parameter TAG_W     = 3
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- slice side (flattened) ---------------------------------------------
    input  wire [N-1:0]             s_req_valid,
    output wire [N-1:0]             s_req_ready,
    input  wire [N*ADDR_W-1:0]      s_req_addr,
    input  wire [N-1:0]             s_req_we,
    input  wire [N*LINE_BITS-1:0]   s_req_wdata,
    input  wire [N*TAG_W-1:0]       s_req_tag,

    output wire [N-1:0]             s_rsp_valid,
    output wire [N*TAG_W-1:0]       s_rsp_tag,
    output wire [N*LINE_BITS-1:0]   s_rsp_rdata,

    // ---- memory side --------------------------------------------------------
    output wire                     m_req_valid,
    input  wire                     m_req_ready,
    output wire [ADDR_W-1:0]        m_req_addr,
    output wire                     m_req_we,
    output wire [LINE_BITS-1:0]     m_req_wdata,
    output wire [TAG_W-1:0]         m_req_tag,
    output wire [N_W-1:0]           m_req_src,

    input  wire                     m_rsp_valid,
    input  wire [TAG_W-1:0]         m_rsp_tag,
    input  wire [N_W-1:0]           m_rsp_src,
    input  wire [LINE_BITS-1:0]     m_rsp_rdata
);

    reg [N_W-1:0] rr, held_pick;
    reg held;
    integer ai;

    reg [N_W-1:0] pick;
    reg           pick_ok;
    always @(*) begin
        pick    = {N_W{1'b0}};
        pick_ok = 1'b0;
        for (ai = N-1; ai >= 0; ai = ai - 1) begin : g_pick
            if (s_req_valid[(rr + ai) % N]) begin
                pick    = (rr + ai) % N;
                pick_ok = 1'b1;
            end
        end
        // Once visible under backpressure, the selected requester owns the
        // output until its handshake. Its own request obeys ready/valid.
        if(held) begin pick=held_pick;pick_ok=s_req_valid[held_pick];end
    end

    assign m_req_valid = pick_ok;
    assign m_req_addr  = s_req_addr [pick*ADDR_W    +: ADDR_W];
    assign m_req_we    = s_req_we   [pick];
    assign m_req_wdata = s_req_wdata[pick*LINE_BITS +: LINE_BITS];
    assign m_req_tag   = s_req_tag  [pick*TAG_W     +: TAG_W];
    assign m_req_src   = pick;

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : g_rdy
            assign s_req_ready[gi] = pick_ok && (pick == gi) && m_req_ready;
            assign s_rsp_valid[gi] = m_rsp_valid && (m_rsp_src == gi);
            assign s_rsp_tag[gi*TAG_W +: TAG_W]         = m_rsp_tag;
            assign s_rsp_rdata[gi*LINE_BITS +: LINE_BITS] = m_rsp_rdata;
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin rr <= 0;held<=0;held_pick<=0;end
        else begin
            if(m_req_valid && !m_req_ready && !held) begin held<=1;held_pick<=pick;end
            if(m_req_valid && m_req_ready) begin
                held<=0;
                rr <= (pick == N-1) ? {N_W{1'b0}} : (pick + 1'b1);
            end
        end
    end

endmodule
