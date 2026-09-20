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

`define TNCA_NOC_PAYLOAD_W 216
`define TNCA_NOC_NV        2
`define TNCA_VN_REQ        0
`define TNCA_VN_RSP        1

// ---- request payload --------------------------------------------------------
`define TNCA_REQ_ADDR   63:0      // byte address, word aligned
`define TNCA_REQ_WE     64
`define TNCA_REQ_BE     72:65
`define TNCA_REQ_WDATA  136:73
`define TNCA_REQ_SRC    140:137   // requesting tile
`define TNCA_REQ_TAG    145:141   // requester's transaction tag
`define TNCA_REQ_AMO    146       // atomic read-modify-write (ISA sec 9.2)
`define TNCA_REQ_AOP    150:147   // TNCA_AOP_*
`define TNCA_REQ_CMP    214:151   // atom.cas compare operand

// The tag is five bits because a tile now has two independent memory clients:
// the LSU and the tensor transfer engine. Bit 4 says which, and the low four
// are that client's own tag. A response therefore reaches the right engine
// without the network keeping any per-transaction state.
//
// WHY THE FLIT CARRIES A SECOND 64-BIT OPERAND
//
// atom.cas compares one value and swaps in another, so an atomic request has
// two data operands where every other request has one. With one flit per
// packet -- the Stage-3 simplification -- the flit has to be as wide as the
// widest request, so the compare operand is a field rather than a second beat.
// A real implementation serialises it; link width accounting is a Stage-4
// item (sec 10.3) and this is one of the numbers it would look at.

// ---- response payload -------------------------------------------------------
`define TNCA_RSP_RDATA  63:0
`define TNCA_RSP_TAG    68:64

`endif
