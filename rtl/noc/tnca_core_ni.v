//==============================================================================
// tnca_core_ni -- network interface on a core's memory port
//
// Turns the core's tagged word-granular memory requests into request flits
// addressed to the home slice, and returns response flits to the core.
//
// HOME SELECTION (arch overview sec 10.1, 10.2)
//
// home = H(physical line address), a pure function of the address and nothing
// else. Two different views of the same physical line must never compute
// different homes, which is why this takes no descriptor, no TensorID and no
// hint as input -- those are performance metadata and cannot be allowed to
// change where a line lives (sec 7.5).
//
// The hash interleaves at line granularity and folds two address bits in, so a
// stride that is a multiple of the slice count does not pile every access onto
// one slice. Page colouring and program-to-cluster affinity (sec 10.2) are
// placement policy and belong to the runtime; they change which physical pages
// a program gets, not this function.
//
// WHY THIS CANNOT DEADLOCK (sec 10.4, 10.5)
//
// XY routing removes cyclic channel dependencies inside a virtual network, but
// sec 10.4 is explicit that endpoint protocol can still deadlock. The rule that
// closes it here: the core's response path is unconditionally ready. The LSU
// reserved a beat-table slot before issuing, so the landing site for the
// response already exists and accepting it needs no further allocation
// (sec 10.5, rules 1 and 2). The response network therefore always drains,
// which lets the requests occupying the slices retire.
//==============================================================================
`include "tnca_noc_pkg.vh"

module tnca_core_ni #(
    parameter XW      = 2,
    parameter YW      = 2,
    parameter NX      = 2,
    parameter NY      = 2,
    parameter TAG_W   = 3,
    parameter SRC_W   = 4,
    parameter MY_ID   = 0,
    parameter LINE_SH = 6            // log2(line bytes)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- core memory port ---------------------------------------------------
    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [63:0]              req_addr,
    input  wire                     req_we,
    input  wire [7:0]               req_be,
    input  wire [63:0]              req_wdata,
    input  wire [TAG_W-1:0]         req_tag,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [63:0]              rsp_rdata,

    // ---- mesh local port ----------------------------------------------------
    output wire                                     net_in_valid_req,
    input  wire                                     net_in_ready_req,
    output wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0]     net_in_flit_req,

    output wire                                     net_in_valid_rsp,
    input  wire                                     net_in_ready_rsp,
    output wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0]     net_in_flit_rsp,

    input  wire                                     net_out_valid_req,
    output wire                                     net_out_ready_req,
    input  wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0]     net_out_flit_req,

    input  wire                                     net_out_valid_rsp,
    output wire                                     net_out_ready_rsp,
    input  wire [XW+YW+`TNCA_NOC_PAYLOAD_W-1:0]     net_out_flit_rsp
);

    localparam PW = `TNCA_NOC_PAYLOAD_W;

    // ---- home slice ---------------------------------------------------------
    wire [63:0] line = req_addr >> LINE_SH;
    wire [1:0]  home = line[1:0] ^ line[3:2];
    wire [XW-1:0] home_x = home[0];
    wire [YW-1:0] home_y = home[1];

    // ---- request injection --------------------------------------------------
    wire [PW-1:0] req_payload;
    assign req_payload[`TNCA_REQ_ADDR]  = req_addr;
    assign req_payload[`TNCA_REQ_WE]    = req_we;
    assign req_payload[`TNCA_REQ_BE]    = req_be;
    assign req_payload[`TNCA_REQ_WDATA] = req_wdata;
    assign req_payload[`TNCA_REQ_SRC]   = MY_ID[3:0];
    assign req_payload[`TNCA_REQ_TAG]   = req_tag;

    assign net_in_valid_req = req_valid;
    assign net_in_flit_req  = {home_y, home_x, req_payload};
    assign req_ready        = net_in_ready_req;

    // A core never injects on the response network.
    assign net_in_valid_rsp = 1'b0;
    assign net_in_flit_rsp  = {(XW+YW+PW){1'b0}};

    // A core is not a destination for requests.
    assign net_out_ready_req = 1'b1;

    // ---- response delivery --------------------------------------------------
    // Unconditionally ready: this is the endpoint rule the network's freedom
    // from deadlock rests on.
    assign net_out_ready_rsp = 1'b1;

    always @(posedge clk) begin
        if (!rst_n) begin
            rsp_valid <= 1'b0;
        end else begin
            rsp_valid <= net_out_valid_rsp;
            rsp_tag   <= net_out_flit_rsp[`TNCA_RSP_TAG];
            rsp_rdata <= net_out_flit_rsp[`TNCA_RSP_RDATA];
        end
    end

    wire _unused = &{1'b0, net_out_valid_req, net_out_flit_req, 1'b0};

endmodule
