// tnca_uarch_defaults.vh -- what a device is elaborated with when nothing says otherwise
//
// A profile's parameters reach the device as one generated header of
// `define TNCA_IF_<NAME> / `define TNCA_UARCH_<NAME> lines (python/tnca/profiles.py,
// uarch_header), compiled before the platform and the device. A name the
// header leaves out takes the value here, which is the RTL's own default:
// a feature that is not mentioned is off.
//
// Two kinds, because a platform has to know one and must not need the other:
//
//   TNCA_IF_*     shape the device's ports. The platform reads these too
//                 (cmd_role_mask is NUM_RCTX wide, done_fault_rctx RCTX_W,
//                 the memory tag LSU_TAG_W). A new one is a contract change.
//   TNCA_UARCH_*  everything inside. The platform never names them, so a new
//                 one is a line here, a parameter in the device, and an entry
//                 in the profile -- nothing in any platform.
//
// Every definition is guarded, so including this more than once is harmless.

`ifndef TNCA_IF_NUM_RCTX
`define TNCA_IF_NUM_RCTX 2
`endif
`ifndef TNCA_IF_RCTX_W
`define TNCA_IF_RCTX_W 1
`endif
`ifndef TNCA_IF_LSU_TAG_W
`define TNCA_IF_LSU_TAG_W 5
`endif

`ifndef TNCA_UARCH_VLEN
`define TNCA_UARCH_VLEN 16
`endif
`ifndef TNCA_UARCH_ELEN
`define TNCA_UARCH_ELEN 32
`endif
`ifndef TNCA_UARCH_VREGS_PER_RCTX
`define TNCA_UARCH_VREGS_PER_RCTX 64
`endif
// Derived unless a profile states them: the physical file is every
// context's registers, and its index is as wide as that needs.
`ifndef TNCA_UARCH_PHYS_VREGS
`define TNCA_UARCH_PHYS_VREGS (NUM_RCTX * VREGS_PER_RCTX)
`endif
`ifndef TNCA_UARCH_PHYS_W
`define TNCA_UARCH_PHYS_W $clog2(PHYS_VREGS)
`endif
`ifndef TNCA_UARCH_HAS_SFU
`define TNCA_UARCH_HAS_SFU 1
`endif
`ifndef TNCA_UARCH_HAS_ATOMICS
`define TNCA_UARCH_HAS_ATOMICS 1
`endif
`ifndef TNCA_UARCH_HAS_SYS_ATOMICS
`define TNCA_UARCH_HAS_SYS_ATOMICS 0
`endif
`ifndef TNCA_UARCH_USE_BANKED_VRF
`define TNCA_UARCH_USE_BANKED_VRF 0
`endif
`ifndef TNCA_UARCH_VRF_BYPASS
`define TNCA_UARCH_VRF_BYPASS 0
`endif
`ifndef TNCA_UARCH_FE_NOBUBBLE
`define TNCA_UARCH_FE_NOBUBBLE 0
`endif
`ifndef TNCA_UARCH_VRF_PREFETCH
`define TNCA_UARCH_VRF_PREFETCH 0
`endif
`ifndef TNCA_UARCH_RCTX_SWITCH
`define TNCA_UARCH_RCTX_SWITCH 0
`endif
`ifndef TNCA_UARCH_LSU_INFLIGHT
`define TNCA_UARCH_LSU_INFLIGHT 1
`endif
`ifndef TNCA_UARCH_RSLICE
`define TNCA_UARCH_RSLICE 0
`endif
`ifndef TNCA_UARCH_RPAIR
`define TNCA_UARCH_RPAIR 0
`endif
`ifndef TNCA_UARCH_MTX_ABUF
`define TNCA_UARCH_MTX_ABUF 0
`endif
`ifndef TNCA_UARCH_FPU_OVERLAP
`define TNCA_UARCH_FPU_OVERLAP 0
`endif
`ifndef TNCA_UARCH_HAS_TENSOR_XPOSE
`define TNCA_UARCH_HAS_TENSOR_XPOSE 0
`endif

`ifndef TNCA_UARCH_BAR_N
`define TNCA_UARCH_BAR_N 32
`endif
`ifndef TNCA_UARCH_BAR_W
`define TNCA_UARCH_BAR_W 5
`endif
`ifndef TNCA_UARCH_BAR_EPOCH_W
`define TNCA_UARCH_BAR_EPOCH_W 48
`endif
`ifndef TNCA_UARCH_FMA_IMPL
`define TNCA_UARCH_FMA_IMPL 0
`endif
`ifndef TNCA_UARCH_MAC_DP
`define TNCA_UARCH_MAC_DP 0
`endif
`ifndef TNCA_UARCH_MAC_CUTS
`define TNCA_UARCH_MAC_CUTS 29
`endif
