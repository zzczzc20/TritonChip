//==============================================================================
// tnca_mem_mux -- two memory clients onto one core memory port
//
// The LSU and the tensor transfer engine both issue word requests. They are
// merged here with a fixed priority and distinguished by the top tag bit, so a
// response reaches the right client without the memory system keeping any
// per-client state.
//
// The transfer engine has priority. It is the resource-releasing path: a
// transfer that stalls holds an RF window, and a held window blocks the
// producer that would refill it. Arch overview sec 2.4 is about exactly this
// shape of dependency -- throttle new work, never the paths that free
// resources.
//==============================================================================

module tnca_mem_mux #(
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter CTAG_W    = 4,     // per-client tag
    parameter TAG_W     = 5      // CTAG_W + 1 owner bit
) (
    // ---- client 0: LSU ------------------------------------------------------
    input  wire                     a_req_valid,
    output wire                     a_req_ready,
    input  wire [63:0]              a_req_addr,
    input  wire                     a_req_we,
    input  wire [MEM_BYTES-1:0]     a_req_be,
    input  wire [MEM_DW-1:0]        a_req_wdata,
    input  wire [CTAG_W-1:0]        a_req_tag,
    input  wire [1:0]               a_req_hint,
    // Only the LSU issues atomics; the transfer engine moves tiles and has no
    // read-modify-write to do, so client B has no atomic fields at all.
    input  wire                     a_req_amo,
    input  wire [3:0]               a_req_aop,
    input  wire [MEM_DW-1:0]        a_req_cmp,
    output wire                     a_rsp_valid,
    output wire [CTAG_W-1:0]        a_rsp_tag,
    output wire [MEM_DW-1:0]        a_rsp_rdata,

    // ---- client 1: tensor transfer engine -----------------------------------
    input  wire                     b_req_valid,
    output wire                     b_req_ready,
    input  wire [63:0]              b_req_addr,
    input  wire                     b_req_we,
    input  wire [MEM_BYTES-1:0]     b_req_be,
    input  wire [MEM_DW-1:0]        b_req_wdata,
    input  wire [CTAG_W-1:0]        b_req_tag,
    output wire                     b_rsp_valid,
    output wire [CTAG_W-1:0]        b_rsp_tag,
    output wire [MEM_DW-1:0]        b_rsp_rdata,

    // ---- core memory port ---------------------------------------------------
    output wire                     m_req_valid,
    input  wire                     m_req_ready,
    output wire [63:0]              m_req_addr,
    output wire                     m_req_we,
    output wire [MEM_BYTES-1:0]     m_req_be,
    output wire [MEM_DW-1:0]        m_req_wdata,
    output wire [TAG_W-1:0]         m_req_tag,
    output wire [1:0]               m_req_hint,
    output wire                     m_req_amo,
    output wire [3:0]               m_req_aop,
    output wire [MEM_DW-1:0]        m_req_cmp,
    input  wire                     m_rsp_valid,
    input  wire [TAG_W-1:0]         m_rsp_tag,
    input  wire [MEM_DW-1:0]        m_rsp_rdata
);

    wire pick_b = b_req_valid;

    assign m_req_valid = a_req_valid | b_req_valid;
    assign m_req_addr  = pick_b ? b_req_addr  : a_req_addr;
    assign m_req_we    = pick_b ? b_req_we    : a_req_we;
    assign m_req_be    = pick_b ? b_req_be    : a_req_be;
    assign m_req_wdata = pick_b ? b_req_wdata : a_req_wdata;
    assign m_req_tag   = pick_b ? {1'b1, b_req_tag} : {1'b0, a_req_tag};
    assign m_req_hint  = pick_b ? 2'd0 : a_req_hint;
    assign m_req_amo   = !pick_b && a_req_amo;
    assign m_req_aop   = pick_b ? 4'd0 : a_req_aop;
    assign m_req_cmp   = pick_b ? {MEM_DW{1'b0}} : a_req_cmp;

    // R41 T2: b is the priority client, so its ready is the port's ready.
    // b_req_valid && b_req_ready is unchanged (pick_b IS b_req_valid); the
    // engine's request valid -- a wide OR over its states' requests -- is no
    // longer in front of its own handshake (u_txf -> xbuf family).
    assign b_req_ready = m_req_ready;
    assign a_req_ready = !pick_b && m_req_ready;

    assign a_rsp_valid = m_rsp_valid && !m_rsp_tag[TAG_W-1];
    assign a_rsp_tag   = m_rsp_tag[CTAG_W-1:0];
    assign a_rsp_rdata = m_rsp_rdata;

    assign b_rsp_valid = m_rsp_valid &&  m_rsp_tag[TAG_W-1];
    assign b_rsp_tag   = m_rsp_tag[CTAG_W-1:0];
    assign b_rsp_rdata = m_rsp_rdata;

endmodule
