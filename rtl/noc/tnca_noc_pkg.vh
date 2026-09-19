//==============================================================================
// tnca_noc_pkg.vh -- packet layout on the mesh
//
// One flit per packet. The header is the top of the flit so the router can
// extract it without knowing anything about the payload:
//
//     flit = { dst_y[YW-1:0], dst_x[XW-1:0], payload[PAYLOAD_W-1:0] }
//
// Two payload formats share the width. They travel on different virtual
// networks and are never confused: VN_REQ carries requests, VN_RSP responses.
//==============================================================================
`ifndef TNCA_NOC_PKG_VH
`define TNCA_NOC_PKG_VH

`define TNCA_NOC_PAYLOAD_W 144
`define TNCA_NOC_NV        2
`define TNCA_VN_REQ        0
`define TNCA_VN_RSP        1

// ---- request payload --------------------------------------------------------
`define TNCA_REQ_ADDR   63:0      // byte address, word aligned
`define TNCA_REQ_WE     64
`define TNCA_REQ_BE     72:65
`define TNCA_REQ_WDATA  136:73
`define TNCA_REQ_SRC    140:137   // requesting tile
`define TNCA_REQ_TAG    143:141   // requester's transaction tag

// ---- response payload -------------------------------------------------------
`define TNCA_RSP_RDATA  63:0
`define TNCA_RSP_TAG    66:64

`endif
