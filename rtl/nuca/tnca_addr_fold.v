//==============================================================================
// tnca_addr_fold -- a 64-bit physical address narrowed to W bits (R41 G2)
//
// fold: the low W-1 bits, and as the top bit whether any bit at or above
//       W-1 was set. An address below 2^(W-1) is unchanged; one at or above it
//       stays at or above 2^(W-1), so a home with W-bit addresses still sends
//       it to memory out of every aperture the image maps (and it faults there)
//       instead of aliasing a real line.
// sat:  a size: unchanged below 2^W, all ones otherwise (it still covers every
//       W-bit line above its base).
// W = 64 passes both through.
//==============================================================================
`default_nettype none
module tnca_addr_fold #(parameter W = 64) (
    input  wire [63:0]  addr,
    input  wire [63:0]  size,
    output wire [W-1:0] addr_w,
    output wire [W-1:0] size_w
);
    generate if (W >= 64) begin : g_full
        assign addr_w = addr[W-1:0];
        assign size_w = size[W-1:0];
    end else begin : g_narrow
        assign addr_w = {|addr[63:W-1], addr[W-2:0]};
        assign size_w = (|size[63:W]) ? {W{1'b1}} : size[W-1:0];
    end endgenerate
endmodule
`default_nettype wire
