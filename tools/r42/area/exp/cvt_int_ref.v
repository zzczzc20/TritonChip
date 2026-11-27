// cvt_int_ref -- the vector FPU's int <-> binary32 conversion (tnca_vector_fpu.v function cvt_int_f, copied verbatim) as a
// module, to measure one lane's cost (R42 area). The register wrapper cvt_int_top is the OOC top.
`include "tnca_defs.vh"
module cvt_int_ref (input [31:0] x, input [7:0] fn, input [2:0] rm, output [31:0] y);
    function is_nan32; input [31:0] v;
        is_nan32 = (v[30:23] == 8'hff) && (v[22:0] != 23'd0);
    endfunction
    function is_zero32; input [31:0] v;
        is_zero32 = (v[30:0] == 31'd0);
    endfunction

    // Ordered "less than" on binary32 via the monotone integer mapping.
    function [31:0] cvt_int_f;
        input [31:0] x;
        input [7:0]  fn;
        input [2:0]  rm;
        reg [7:0]  cexp;
        reg [23:0] cman;
        reg [63:0] cmag;
        reg [7:0]  csh;
        reg        cneg, cuns, rnd_inc;
        reg [63:0] cfrac_mask, cfrac, chalf;
        reg [31:0] iv, imag, inorm;
        reg [7:0]  ish, ie;
        reg [24:0] iman;
        reg        ig, ir, isk;
        integer    kk;
        begin
            cvt_int_f = 32'd0;
            if (fn == `TNCA_CVT_S32_F32 || fn == `TNCA_CVT_U32_F32) begin
                // integer -> binary32
                iv   = x;
                cneg = (fn == `TNCA_CVT_S32_F32) && iv[31];
                imag = cneg ? (~iv + 32'd1) : iv;
                if (imag == 32'd0) begin
                    cvt_int_f = 32'd0;
                end else begin
                    ish = 0;
                    for (kk = 0; kk < 32; kk = kk + 1)
                        if (imag[kk]) ish = kk[7:0];       // index of the MSB
                    ie = 127 + ish;
                    // left-align the magnitude so bit 31 is the MSB
                    inorm = imag << (31 - ish);
                    iman  = {1'b0, inorm[31:8]};           // 24 significant bits
                    ig    = inorm[7];
                    isk   = |inorm[6:0];
                    ir    = ig & (isk | iman[0]);
                    iman  = iman + {24'd0, ir};
                    if (iman[24]) begin
                        iman = iman >> 1;
                        ie   = ie + 1;
                    end
                    cvt_int_f = {cneg, ie, iman[22:0]};
                end
            end else begin
                // binary32 -> integer (ISA sec 8.4): round by rm,
                // saturate out-of-range values to the integer extremes,
                // NaN -> 0. An unsigned result below zero saturates to 0.
                cexp = x[30:23];
                cman = {1'b1, x[22:0]};
                cneg = x[31];
                cuns = (fn == `TNCA_CVT_F32_U32);
                cfrac = 64'd0;
                if (is_nan32(x)) begin
                    cvt_int_f = 32'd0;
                end else if (cexp < 8'd127) begin
                    // |x| < 1: only the rounding mode can make it +-1
                    case (rm)
                    `TNCA_RND_RNE: rnd_inc = (cexp == 8'd126) && (x[22:0] != 23'd0); // |x| > 0.5
                    `TNCA_RND_RMI: rnd_inc = cneg  && (x[30:0] != 31'd0);
                    `TNCA_RND_RPI: rnd_inc = !cneg && (x[30:0] != 31'd0);
                    default:       rnd_inc = 1'b0;                          // toward zero
                    endcase
                    cvt_int_f = !rnd_inc ? 32'd0 :
                             (cneg ? (cuns ? 32'd0 : 32'hffffffff) : 32'd1);
                end else if (cexp >= (cuns ? 8'd159 : 8'd158)) begin
                    // |x| >= 2^31 (signed) or 2^32 (unsigned), and infinities
                    cvt_int_f = cuns ? (cneg ? 32'd0 : 32'hffffffff)
                                  : (cneg ? 32'h80000000 : 32'h7fffffff);
                end else begin
                    // 1 <= |x| < 2^32: |x| = {cman, 8'b0} / 2^csh with
                    // csh = 158 - cexp in [0, 31], never negative or wrapped.
                    csh   = 8'd158 - cexp;
                    cmag  = {32'd0, cman, 8'd0} >> csh[4:0];
                    cfrac_mask = (64'd1 << csh[4:0]) - 64'd1;
                    cfrac = {32'd0, cman, 8'd0} & cfrac_mask;
                    chalf = (csh[4:0] == 5'd0) ? 64'd0 : (64'd1 << (csh[4:0] - 5'd1));
                    case (rm)
                    `TNCA_RND_RNE: rnd_inc = (csh[4:0] != 5'd0) &&
                                             ((cfrac > chalf) || ((cfrac == chalf) && cmag[0]));
                    `TNCA_RND_RMI: rnd_inc = cneg  && (cfrac != 64'd0);
                    `TNCA_RND_RPI: rnd_inc = !cneg && (cfrac != 64'd0);
                    default:       rnd_inc = 1'b0;
                    endcase
                    cmag = cmag + {63'd0, rnd_inc};
                    if (!cuns) begin
                        // cmag <= 2^31: -2^31 is exact, +2^31 saturates
                        cvt_int_f = cneg ? (~cmag[31:0] + 32'd1) :
                                 (cmag > 64'd2147483647) ? 32'h7fffffff : cmag[31:0];
                    end else begin
                        cvt_int_f = cneg ? 32'd0 :
                                 (cmag > 64'd4294967295) ? 32'hffffffff : cmag[31:0];
                    end
                end
            end
        end
    endfunction
    assign y = cvt_int_f(x, fn, rm);
endmodule
module cvt_int_top (input clk, input [31:0] x, input [7:0] fn, input [2:0] rm, output reg [31:0] y_q);
    reg [31:0] wx; reg [7:0] wf; reg [2:0] wr;
    always @(posedge clk) begin wx <= x; wf <= fn; wr <= rm; end
    wire [31:0] y;
    cvt_int_ref u (.x(wx), .fn(wf), .rm(wr), .y(y));
    always @(posedge clk) y_q <= y;
endmodule
// the same wrapper around the R42 area implementation tnca_cvt_int
module cvt_new_top (input clk, input [31:0] x, input [7:0] fn, input [2:0] rm, output reg [31:0] y_q);
    reg [31:0] wx; reg [7:0] wf; reg [2:0] wr;
    always @(posedge clk) begin wx <= x; wf <= fn; wr <= rm; end
    wire [31:0] y;
    tnca_cvt_int u (.x(wx), .fn(wf), .rm(wr), .y(y));
    always @(posedge clk) y_q <= y;
endmodule
