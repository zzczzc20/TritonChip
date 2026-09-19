//==============================================================================
// tnca_slice_ni -- network interface on a NUCA slice
//
// Delivers request flits into the slice and turns slice responses back into
// response flits addressed at the original requester.
//
// The requester's tile id travels in the request payload and is stored in the
// slice's target entry, so a response knows where to go without the network
// keeping any per-transaction state. The slice never has to know the topology.
//==============================================================================
`include "tnca_noc_pkg.vh"

module tnca_slice_ni #(
    parameter XW    = 2,
    parameter YW    = 2,
    parameter TAG_W = 5,
    parameter SRC_W = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- slice request port -------------------------------------------------
    output wire                     req_valid,
    input  wire                     req_ready,
    output wire [63:0]              req_addr,
    output wire                     req_we,
    output wire [7:0]               req_be,
    output wire [63:0]              req_wdata,
    output wire [SRC_W-1:0]         req_src,
    output wire [TAG_W-1:0]         req_tag,
    output wire                     req_amo,
    output wire [3:0]               req_aop,
    output wire [63:0]              req_cmp,

    // ---- slice response port ------------------------------------------------
    input  wire                     rsp_valid,
    output wire                     rsp_ready,
    input  wire [SRC_W-1:0]         rsp_src,
    input  wire [TAG_W-1:0]         rsp_tag,
    input  wire [63:0]              rsp_rdata,

    // ---- mesh local port ----------------------------------------------------
    output wire                                 net_in_valid_req,
    input  wire                                 net_in_ready_req,
    output wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0] net_in_flit_req,

    output wire                                 net_in_valid_rsp,
    input  wire                                 net_in_ready_rsp,
    output wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0] net_in_flit_rsp,

    input  wire                                 net_out_valid_req,
    output wire                                 net_out_ready_req,
    input  wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0] net_out_flit_req,

    input  wire                                 net_out_valid_rsp,
    output wire                                 net_out_ready_rsp,
    input  wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0] net_out_flit_rsp
);

    localparam PW = `TNCA_NOC_PAYLOAD_W;

    // ---- inbound requests ---------------------------------------------------
    assign req_valid         = net_out_valid_req;
    assign net_out_ready_req = req_ready;
    assign req_addr          = net_out_flit_req[`TNCA_REQ_ADDR];
    assign req_we            = net_out_flit_req[`TNCA_REQ_WE];
    assign req_be            = net_out_flit_req[`TNCA_REQ_BE];
    assign req_wdata         = net_out_flit_req[`TNCA_REQ_WDATA];
    assign req_src           = net_out_flit_req[`TNCA_REQ_SRC];
    assign req_amo           = net_out_flit_req[`TNCA_REQ_AMO];
    assign req_aop           = net_out_flit_req[`TNCA_REQ_AOP];
    assign req_cmp           = net_out_flit_req[`TNCA_REQ_CMP];
    // The requester's tag is 3 bits on the wire; the slice's tag field is
    // wider, so zero-extend rather than chain a range onto a range.
    wire [4:0] req_tag_raw = net_out_flit_req[`TNCA_REQ_TAG];
    assign req_tag           = {{(TAG_W-5){1'b0}}, req_tag_raw};

    // ---- outbound responses -------------------------------------------------
    wire [PW-1:0] rsp_payload;
    assign rsp_payload[`TNCA_RSP_RDATA] = rsp_rdata;
    assign rsp_payload[`TNCA_RSP_TAG]   = rsp_tag[4:0];
    assign rsp_payload[PW-1:69]         = {(PW-69){1'b0}};

    // The requester's tile id decodes to mesh coordinates here, which is the
    // only place in the design that knows the topology.
    wire [XW-1:0] dst_x = rsp_src[0];
    wire [YW-1:0] dst_y = rsp_src[1];

    assign net_in_valid_rsp = rsp_valid;
    assign net_in_flit_rsp  = {dst_y, dst_x, rsp_payload};
    assign rsp_ready        = net_in_ready_rsp;

    // A slice never injects a request and never receives one.
    assign net_in_valid_req  = 1'b0;
    assign net_in_flit_req   = {(XW+YW+PW){1'b0}};
    assign net_out_ready_rsp = 1'b1;

    wire _unused = &{1'b0, net_out_valid_rsp, net_out_flit_rsp, clk, rst_n, 1'b0};

endmodule
