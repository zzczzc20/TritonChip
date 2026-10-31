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
    parameter DATA_W = 64,
    parameter BYTE_W = DATA_W/8,
    parameter PAYLOAD_W = 216 + DATA_W-64 + BYTE_W-8,
    parameter NX    = 2,
    parameter NY    = 2,
    parameter XW    = 2,
    parameter YW    = 2,
    parameter TAG_W = 5,
    parameter WTAG_W = 5,           // the tag on the wire (R41 G4: the core's port tag)
    parameter SRC_W = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- slice request port -------------------------------------------------
    output wire                     req_valid,
    input  wire                     req_ready,
    output wire [63:0]              req_addr,
    output wire                     req_we,
    output wire [BYTE_W-1:0]        req_be,
    output wire [DATA_W-1:0]        req_wdata,
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
    input  wire [DATA_W-1:0]        rsp_rdata,

    // ---- mesh local port ----------------------------------------------------
    output wire                                 net_in_valid_req,
    input  wire                                 net_in_ready_req,
    output wire [XW+YW+PAYLOAD_W-1:0] net_in_flit_req,

    output wire                                 net_in_valid_rsp,
    input  wire                                 net_in_ready_rsp,
    output wire [XW+YW+PAYLOAD_W-1:0] net_in_flit_rsp,

    input  wire                                 net_out_valid_req,
    output wire                                 net_out_ready_req,
    input  wire [XW+YW+PAYLOAD_W-1:0] net_out_flit_req,

    input  wire                                 net_out_valid_rsp,
    output wire                                 net_out_ready_rsp,
    input  wire [XW+YW+PAYLOAD_W-1:0] net_out_flit_rsp
);

    localparam PW = PAYLOAD_W;
    // Preserve every legacy bit position at DATA_W=64. Comparison operands
    // remain 64-bit scalar values; wide data/byte enables expand together.
    localparam BE_L=65, DATA_L=BE_L+BYTE_W, SRC_L=DATA_L+DATA_W;
    localparam TAG_L=SRC_L+4, AMO_L=TAG_L+WTAG_W, AOP_L=AMO_L+1, CMP_L=AOP_L+4;

    // ---- inbound requests ---------------------------------------------------
    assign req_valid         = net_out_valid_req;
    assign net_out_ready_req = req_ready;
    assign req_addr          = net_out_flit_req[`TNCA_REQ_ADDR];
    assign req_we            = net_out_flit_req[`TNCA_REQ_WE];
    assign req_be            = net_out_flit_req[BE_L+:BYTE_W];
    assign req_wdata         = net_out_flit_req[DATA_L+:DATA_W];
    assign req_src           = net_out_flit_req[SRC_L+:4];
    assign req_amo           = net_out_flit_req[AMO_L];
    assign req_aop           = net_out_flit_req[AOP_L+:4];
    assign req_cmp           = net_out_flit_req[CMP_L+:64];
    // The requester's tag is 3 bits on the wire; the slice's tag field is
    // wider, so zero-extend rather than chain a range onto a range.
    wire [WTAG_W-1:0] req_tag_raw = net_out_flit_req[TAG_L+:WTAG_W];
    assign req_tag           = req_tag_raw;   // zero-extended to the slice's tag

    // ---- outbound responses -------------------------------------------------
    wire [PW-1:0] rsp_payload;
    assign rsp_payload[0+:DATA_W] = rsp_rdata;
    assign rsp_payload[DATA_W+:WTAG_W] = rsp_tag[WTAG_W-1:0];
    assign rsp_payload[PW-1:DATA_W+WTAG_W] = {(PW-DATA_W-WTAG_W){1'b0}};

    // The requester's tile id decodes to mesh coordinates here, which is the
    // only place in the design that knows the topology.
    wire [XW-1:0] dst_x = rsp_src % NX;
    wire [YW-1:0] dst_y = rsp_src / NX;

    assign net_in_valid_rsp = rsp_valid;
    assign net_in_flit_rsp  = {dst_y, dst_x, rsp_payload};
    assign rsp_ready        = net_in_ready_rsp;

    // A slice never injects a request and never receives one.
    assign net_in_valid_req  = 1'b0;
    assign net_in_flit_req   = {(XW+YW+PW){1'b0}};
    assign net_out_ready_rsp = 1'b1;

    wire _unused = &{1'b0, net_out_valid_rsp, net_out_flit_rsp, clk, rst_n, 1'b0};

`ifndef SYNTHESIS
    initial if ((DATA_W!=64 && DATA_W!=128 && DATA_W!=256 && DATA_W!=512)
                || BYTE_W!=DATA_W/8 || WTAG_W<5 || TAG_W<WTAG_W || PAYLOAD_W!=216+DATA_W-64+BYTE_W-8+WTAG_W-5)
        $fatal(1,"invalid NI data/mask/payload geometry");
`endif

endmodule
