//==============================================================================
// tnca_vector_fpu -- VLEN-lane binary32 unit
//
// Covers ISA major 0x11 (vector FP), 0x12 (convert), the FP subset of 0x15
// (reduction) and 0x16 (SFU).
//
// Every arithmetic result goes through tnca_fp32_fma, so there is exactly one
// rounding implementation in the core:
//     vf.add a,b  ->  fma(a, 1.0, b)          (a*1.0 is exact)
//     vf.sub a,b  ->  fma(a, 1.0, -b)
//     vf.mul a,b  ->  fma(a, b, -) mul_only
//     vf.fma      ->  fma(a, b, c)
//
// LATENCY
// -------
// Every operation in this unit produces its result FP_LAT cycles after the
// operands are presented, reductions excepted. The latency is uniform on
// purpose: mixing one- and three-cycle results inside a single functional unit
// would push the difference into the core's writeback scheduler for no benefit,
// and the non-FMA paths (min/max, compare, convert) are cheap enough to delay.
//
// Reductions
// ----------
// vf.reduce.* is a LOG2(VLEN)-cycle sequential operation that folds the vector
// through lanes 0..VLEN/2-1 of the existing FMA array, one tree level per
// cycle. Building the tree out of 15 dedicated adders would be about a quarter
// more FP area for an operation that is not on the GEMM inner loop; the arch
// overview's own advice (sec 6.2) is to size the arithmetic after the register
// file, not before it.
//
// The fold is a balanced binary tree over lane index, which is what makes
// vf.reduce.add bit-reproducible as ISA sec 8.7 requires. Masked-off lanes are
// replaced by the operation's identity before the first level.
//==============================================================================
`include "tnca_defs.vh"

module tnca_vector_fpu #(
    parameter VLEN   = 16,
    parameter ELEN   = 32,
    parameter LW     = 4,            // log2(VLEN)
    parameter FP_LAT = 2,            // must match tnca_fp32_fma's stage count
    parameter SFU_LAT = 6,           // must match rtl/gen/tnca_sfu_*_xls.v
    // Follows the core's HAS_SFU. A build with the capability bit clear does
    // not instantiate the slice, so it does not pay ~60 k gates for logic its
    // decoder rejects.
    parameter HAS_SFU = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [7:0]               major,
    input  wire [7:0]               func,
    input  wire [9:0]               mod,
    input  wire [VLEN-1:0]          mask,
    input  wire [VLEN*ELEN-1:0]     va,
    input  wire [VLEN*ELEN-1:0]     vb,
    input  wire [VLEN*ELEN-1:0]     vc,

    // multi-cycle control (reductions only)
    output wire                     is_multicycle,
    input  wire                     start,         // issue pulse, 1 cycle
    output wire                     busy,
    output wire                     done,

    output reg  [VLEN*ELEN-1:0]     vres,
    output reg  [63:0]              sres,
    output reg  [VLEN-1:0]          pres,

    // Multicycle result, valid in the cycle `done` is asserted. It is a
    // separate port on purpose: vres/sres are muxed by the *current*
    // instruction's opcode, and by the time a reduction or an SFU pass
    // finishes the opcode inputs belong to whatever else has reached the
    // execute stage.
    output wire [63:0]              mc_sres,
    output wire [VLEN*ELEN-1:0]     mc_vres
);

    localparam [31:0] FP_ONE   = 32'h3f800000;
    localparam [31:0] FP_PZERO = 32'h00000000;
    localparam [31:0] FP_NINF  = 32'hff800000;
    localparam [31:0] FP_PINF  = 32'h7f800000;

    integer i;
    genvar  g;

    wire [ELEN-1:0] a [0:VLEN-1];
    wire [ELEN-1:0] b [0:VLEN-1];
    wire [ELEN-1:0] c [0:VLEN-1];
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_slice
            assign a[g] = va[g*ELEN +: ELEN];
            assign b[g] = vb[g*ELEN +: ELEN];
            assign c[g] = vc[g*ELEN +: ELEN];
        end
    endgenerate

    // Two multicycle families share this unit's completion port: FP
    // reductions, which fold through the FMA array, and the SFU, which has a
    // datapath of its own. They can never overlap -- one instruction is in
    // the execute stage at a time, and the core will not issue a second
    // multicycle op into a busy unit -- so they share `busy`, `done` and the
    // result port and differ only in which sequencer runs.
    //
    // Only FP reductions belong here. Integer reductions share major 0x15 but
    // run on tnca_vector_alu's tree; claiming them here would start the
    // sequencer and later retire a write to a stale destination.
    wire is_red_mc = (major == `TNCA_MAJ_RED) && (func >= `TNCA_RD_FADD);
    wire is_sfu_mc = (HAS_SFU != 0) && (major == `TNCA_MAJ_SFU);
    assign is_multicycle = is_red_mc | is_sfu_mc;

    // --------------------------------------------------------------------------
    // Reduction sequencer state. Level `lvl` folds 2^(LW-lvl) values into half.
    // --------------------------------------------------------------------------
    reg [ELEN-1:0] acc [0:VLEN-1];
    reg [LW-1:0]   lvl;
    reg [7:0]      red_func;
    reg [9:0]      red_mod;
    reg            red_busy, red_done;

    // Keyed on the reduction alone: an SFU pass must not hijack the FMA
    // operands, and `is_multicycle` now covers both families.
    wire red_active = red_busy | (start & is_red_mc);

    // During a fold the instruction inputs belong to a later instruction, so
    // the rounding mode has to come from the latched copy.
    wire ftz = red_active ? red_mod[8] : mod[8];

    // Identity for the masked-off lanes.
    reg [ELEN-1:0] ident;
    always @(*) begin
        case (func)
        `TNCA_RD_FADD: ident = FP_PZERO;
        `TNCA_RD_FMAX: ident = FP_NINF;
        `TNCA_RD_FMIN: ident = FP_PINF;
        default:       ident = FP_PZERO;
        endcase
    end

    // --------------------------------------------------------------------------
    // FMA operand selection: lane ops, or a reduction tree level.
    // --------------------------------------------------------------------------

    reg [ELEN-1:0] fa [0:VLEN-1];
    reg [ELEN-1:0] fb [0:VLEN-1];
    reg [ELEN-1:0] fc [0:VLEN-1];
    reg            f_sub, f_mul;

    always @(*) begin
        f_sub = 1'b0;
        f_mul = 1'b0;
        case (major)
        `TNCA_MAJ_VFP: begin
            case (func)
            `TNCA_VF_ADD: f_sub = 1'b0;
            `TNCA_VF_SUB: f_sub = 1'b1;
            `TNCA_VF_MUL: f_mul = 1'b1;
            `TNCA_VF_FMS: f_sub = 1'b1;
            default:      ;
            endcase
        end
        default: ;
        endcase

        for (i = 0; i < VLEN; i = i + 1) begin
            // default: lane arithmetic
            case (func)
            `TNCA_VF_ADD, `TNCA_VF_SUB: begin
                fa[i] = a[i]; fb[i] = FP_ONE; fc[i] = b[i];
            end
            `TNCA_VF_MUL: begin
                fa[i] = a[i]; fb[i] = b[i];   fc[i] = FP_PZERO;
            end
            default: begin   // vf.fma / vf.fms
                fa[i] = a[i]; fb[i] = b[i];   fc[i] = c[i];
            end
            endcase
        end

        // A reduction level overrides lanes 0..VLEN/2-1 with tree operands.
        if (red_active) begin
            for (i = 0; i < VLEN; i = i + 1) begin
                fa[i] = acc[2*i % VLEN];
                fb[i] = FP_ONE;
                fc[i] = acc[(2*i + 1) % VLEN];
            end
            f_sub = 1'b0;
            f_mul = 1'b0;
        end
    end

    wire [ELEN-1:0] fma_y [0:VLEN-1];
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_fma
            tnca_fp32_fma u_fma (
                .clk(clk), .rst_n(rst_n),
                .a(fa[g]), .b(fb[g]), .c(fc[g]),
                .sub_c(f_sub), .mul_only(f_mul), .ftz(ftz),
                .y(fma_y[g])
            );
        end
    endgenerate

    // --------------------------------------------------------------------------
    // min / max, compare, neg / abs, convert
    // --------------------------------------------------------------------------
    function is_nan32; input [31:0] v;
        is_nan32 = (v[30:23] == 8'hff) && (v[22:0] != 23'd0);
    endfunction
    function is_zero32; input [31:0] v;
        is_zero32 = (v[30:0] == 31'd0);
    endfunction

    // Ordered "less than" on binary32 via the monotone integer mapping.
    function lt32; input [31:0] x; input [31:0] y;
        reg [31:0] mx, my;
        begin
            mx = x[31] ? (32'h80000000 - {1'b0, x[30:0]}) : (32'h80000000 + {1'b0, x[30:0]});
            my = y[31] ? (32'h80000000 - {1'b0, y[30:0]}) : (32'h80000000 + {1'b0, y[30:0]});
            lt32 = (mx < my);
        end
    endfunction

    reg [ELEN-1:0] mm  [0:VLEN-1];
    reg            cmpr[0:VLEN-1];
    reg [ELEN-1:0] cvt [0:VLEN-1];
    reg [ELEN-1:0] na  [0:VLEN-1];

    reg x_nan, y_nan, both_zero, alt, is_max;
    reg unord, eqv, ltv;

    // FP -> int, with the rounding mode in mod[2:0]
    reg signed [63:0] fixv;
    reg [7:0]  cexp;
    reg [23:0] cman;
    reg [63:0] cmag;
    reg [7:0]  csh;
    reg        cneg, rnd_inc;
    reg [63:0] cfrac_mask;
    reg [63:0] cfrac;
    reg [63:0] chalf;

    // int -> FP, round to nearest even: fold through the FMA is overkill, so
    // normalise directly.
    reg [31:0] iv;
    reg [31:0] imag;
    reg [7:0]  ish;
    reg [7:0]  ie;
    reg [31:0] inorm;
    reg [24:0] iman;
    reg        ig, ir, isk;
    integer    kk;

    always @(*) begin
        is_max = (func == `TNCA_VF_MAX);
        for (i = 0; i < VLEN; i = i + 1) begin
            // ---- min / max (ISA sec 8.3) ----------------------------------
            x_nan = is_nan32(a[i]);
            y_nan = is_nan32(b[i]);
            both_zero = is_zero32(a[i]) && is_zero32(b[i]);
            ltv = lt32(a[i], b[i]);
            alt = is_max ? ~ltv : ltv;     // pick a[i] ?

            if (mod[1:0] == `TNCA_FMM_PROP && (x_nan || y_nan))
                mm[i] = 32'h7fc00000;
            else if (x_nan && y_nan)
                mm[i] = 32'h7fc00000;
            else if (x_nan)
                mm[i] = b[i];
            else if (y_nan)
                mm[i] = a[i];
            else if (both_zero && (mod[1:0] != `TNCA_FMM_FAST))
                // min(+0,-0) = -0 ; max(+0,-0) = +0
                mm[i] = is_max ? ((a[i][31] == 1'b0) ? a[i] : b[i])
                               : ((a[i][31] == 1'b1) ? a[i] : b[i]);
            else
                mm[i] = alt ? a[i] : b[i];

            // ---- compare (ISA sec 4, FP column) ---------------------------
            unord = x_nan | y_nan;
            eqv   = (!unord) && ((a[i] == b[i]) || (is_zero32(a[i]) && is_zero32(b[i])));
            case (mod[3:0])
            `TNCA_CC_EQ: cmpr[i] = eqv;
            `TNCA_CC_NE: cmpr[i] = unord | ~eqv;
            `TNCA_CC_LT: cmpr[i] = (!unord) && ltv && !eqv;
            `TNCA_CC_LE: cmpr[i] = (!unord) && (ltv || eqv);
            `TNCA_CC_GT: cmpr[i] = (!unord) && !ltv && !eqv;
            `TNCA_CC_GE: cmpr[i] = (!unord) && (!ltv || eqv);
            `TNCA_CC_O:  cmpr[i] = ~unord;
            `TNCA_CC_UO: cmpr[i] =  unord;
            default:     cmpr[i] = 1'b0;
            endcase

            // ---- neg / abs -------------------------------------------------
            case (func)
            `TNCA_VF_NEG: na[i] = {~a[i][31], a[i][30:0]};
            `TNCA_VF_ABS: na[i] = { 1'b0,     a[i][30:0]};
            default:      na[i] = a[i];
            endcase

            // ---- convert ---------------------------------------------------
            if (func == `TNCA_CVT_S32_F32 || func == `TNCA_CVT_U32_F32) begin
                // integer -> binary32
                iv   = a[i];
                cneg = (func == `TNCA_CVT_S32_F32) && iv[31];
                imag = cneg ? (~iv + 32'd1) : iv;
                if (imag == 32'd0) begin
                    cvt[i] = 32'd0;
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
                    cvt[i] = {cneg, ie, iman[22:0]};
                end
            end else begin
                // binary32 -> integer, saturating; NaN -> 0
                cexp = a[i][30:23];
                cman = {1'b1, a[i][22:0]};
                cneg = a[i][31];
                if (is_nan32(a[i])) begin
                    cvt[i] = 32'd0;
                end else if (cexp < 8'd127) begin
                    // |x| < 1 : only the rounding mode can make it +-1
                    cfrac = 64'd0;
                    case (mod[2:0])
                    `TNCA_RND_RNE: cvt[i] = ((cexp == 8'd126) && (a[i][22:0] != 23'd0))
                                            ? (cneg ? 32'hffffffff : 32'd1)
                                            : ((cexp == 8'd126) ? 32'd0    // 0.5 -> 0 (even)
                                                                : 32'd0);
                    `TNCA_RND_RMI: cvt[i] = (cneg && (a[i][30:0] != 31'd0))
                                            ? 32'hffffffff : 32'd0;
                    `TNCA_RND_RPI: cvt[i] = (!cneg && (a[i][30:0] != 31'd0))
                                            ? 32'd1 : 32'd0;
                    default:       cvt[i] = 32'd0;         // toward zero
                    endcase
                end else if (cexp >= 8'd158) begin
                    // |x| >= 2^31 : saturate
                    cvt[i] = (func == `TNCA_CVT_F32_S32)
                             ? (cneg ? 32'h80000000 : 32'h7fffffff)
                             : (cneg ? 32'd0        : 32'hffffffff);
                end else begin
                    csh   = 8'd150 - cexp;                 // right shift of the 24-bit mantissa
                    cmag  = {40'd0, cman} >> csh;
                    cfrac_mask = (64'd1 << csh) - 64'd1;
                    cfrac = {40'd0, cman} & cfrac_mask;
                    chalf = 64'd1 << (csh - 1);
                    case (mod[2:0])
                    `TNCA_RND_RNE: rnd_inc = (cfrac > chalf) ||
                                             ((cfrac == chalf) && cmag[0]);
                    `TNCA_RND_RMI: rnd_inc = cneg  && (cfrac != 64'd0);
                    `TNCA_RND_RPI: rnd_inc = !cneg && (cfrac != 64'd0);
                    default:       rnd_inc = 1'b0;
                    endcase
                    cmag = cmag + {63'd0, rnd_inc};
                    fixv = cneg ? -$signed(cmag) : $signed(cmag);
                    if (func == `TNCA_CVT_F32_S32) begin
                        cvt[i] = (fixv > 64'sd2147483647)  ? 32'h7fffffff :
                                 (fixv < -64'sd2147483648) ? 32'h80000000 :
                                 fixv[31:0];
                    end else begin
                        cvt[i] = cneg ? 32'd0 :
                                 (cmag > 64'd4294967295) ? 32'hffffffff : cmag[31:0];
                    end
                end
            end
        end
    end

    // --------------------------------------------------------------------------
    // Reduction sequencer
    // --------------------------------------------------------------------------
    reg [ELEN-1:0] red_step [0:VLEN-1];
    always @(*) begin
        for (i = 0; i < VLEN; i = i + 1) begin
            case (red_func)
            `TNCA_RD_FADD: red_step[i] = fma_y[i];
            `TNCA_RD_FMAX: red_step[i] = (is_nan32(acc[2*i % VLEN]))   ? acc[(2*i+1) % VLEN] :
                                         (is_nan32(acc[(2*i+1)%VLEN])) ? acc[2*i % VLEN]     :
                                         lt32(acc[2*i % VLEN], acc[(2*i+1) % VLEN])
                                            ? acc[(2*i+1) % VLEN] : acc[2*i % VLEN];
            `TNCA_RD_FMIN: red_step[i] = (is_nan32(acc[2*i % VLEN]))   ? acc[(2*i+1) % VLEN] :
                                         (is_nan32(acc[(2*i+1)%VLEN])) ? acc[2*i % VLEN]     :
                                         lt32(acc[2*i % VLEN], acc[(2*i+1) % VLEN])
                                            ? acc[2*i % VLEN] : acc[(2*i+1) % VLEN];
            default:       red_step[i] = acc[2*i % VLEN];
            endcase
        end
    end

    // The FMA array now has FP_LAT cycles of latency, so a tree level cannot be
    // captured in the cycle it is presented. Each level takes FP_LAT+1 cycles:
    // present the operand pairs, wait for them to fall out, capture. The
    // operands are stable throughout because `acc` only changes at the capture.
    reg [1:0] rwait;

    always @(posedge clk) begin
        if (!rst_n) begin
            red_busy <= 1'b0;
            red_done <= 1'b0;
            lvl      <= {LW{1'b0}};
            rwait    <= 2'd0;
        end else begin
            red_done <= 1'b0;
            if (!red_busy) begin
                if (start && is_red_mc) begin
                    for (i = 0; i < VLEN; i = i + 1)
                        acc[i] <= mask[i] ? a[i] : ident;
                    red_func <= func;
                    red_mod  <= mod;
                    lvl      <= {LW{1'b0}};
                    rwait    <= 2'd0;
                    red_busy <= 1'b1;
                end
            end else if (rwait != FP_LAT[1:0]) begin
                rwait <= rwait + 2'd1;
            end else begin
                for (i = 0; i < VLEN/2; i = i + 1)
                    acc[i] <= red_step[i];
                rwait <= 2'd0;
                if (lvl == LW-1) begin
                    red_busy <= 1'b0;
                    red_done <= 1'b1;
                end
                lvl <= lvl + 1'b1;
            end
        end
    end

    // --------------------------------------------------------------------------
    // The SFU. One slice shared by all lanes; see rtl/core/tnca_sfu.v.
    // --------------------------------------------------------------------------
    wire                 sfu_busy, sfu_done;
    wire [VLEN*ELEN-1:0] sfu_vres;

    generate
        if (HAS_SFU != 0) begin : g_sfu
            tnca_sfu #(.VLEN(VLEN), .ELEN(ELEN), .LAT(SFU_LAT)) u_sfu (
                .clk(clk), .rst_n(rst_n),
                .start(start & is_sfu_mc), .func(func), .va(va),
                .busy(sfu_busy), .done(sfu_done), .vres(sfu_vres)
            );
        end else begin : g_no_sfu
            assign sfu_busy = 1'b0;
            assign sfu_done = 1'b0;
            assign sfu_vres = {(VLEN*ELEN){1'b0}};
        end
    endgenerate

    assign busy = red_busy | sfu_busy;
    assign done = red_done | sfu_done;

    // --------------------------------------------------------------------------
    // Output selection
    //
    // The FMA array's results already arrive FP_LAT cycles late. Everything
    // else is combinational, so the *selected* non-FMA result is delayed by the
    // same amount -- delaying one chosen 512-bit value rather than all three
    // candidates, or the three 512-bit inputs.
    //
    // The opcode is delayed alongside it: by the time a result emerges, the
    // opcode inputs belong to whatever else has since reached the execute stage.
    // --------------------------------------------------------------------------
    reg [VLEN*ELEN-1:0] nf_v;
    reg [VLEN-1:0]      nf_p;

    always @(*) begin
        nf_v = {(VLEN*ELEN){1'b0}};
        nf_p = {VLEN{1'b0}};
        case (major)
        `TNCA_MAJ_VFP: begin
            for (i = 0; i < VLEN; i = i + 1) begin
                case (func)
                `TNCA_VF_MIN, `TNCA_VF_MAX: nf_v[i*ELEN +: ELEN] = mm[i];
                `TNCA_VF_NEG, `TNCA_VF_ABS: nf_v[i*ELEN +: ELEN] = na[i];
                default:                    nf_v[i*ELEN +: ELEN] = {ELEN{1'b0}};
                endcase
                nf_p[i] = mask[i] & cmpr[i];
            end
        end
        `TNCA_MAJ_CVT: begin
            for (i = 0; i < VLEN; i = i + 1)
                nf_v[i*ELEN +: ELEN] = cvt[i];
        end
        default: ;
        endcase
    end

    reg [VLEN*ELEN-1:0] dly_v   [0:FP_LAT-1];
    reg [VLEN-1:0]      dly_p   [0:FP_LAT-1];
    reg [7:0]           dly_maj [0:FP_LAT-1];
    reg [7:0]           dly_fn  [0:FP_LAT-1];

    integer d;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (d = 0; d < FP_LAT; d = d + 1) begin
                dly_v[d]   <= {(VLEN*ELEN){1'b0}};
                dly_p[d]   <= {VLEN{1'b0}};
                dly_maj[d] <= 8'd0;
                dly_fn[d]  <= 8'd0;
            end
        end else begin
            dly_v[0]   <= nf_v;
            dly_p[0]   <= nf_p;
            dly_maj[0] <= major;
            dly_fn[0]  <= func;
            for (d = 1; d < FP_LAT; d = d + 1) begin
                dly_v[d]   <= dly_v[d-1];
                dly_p[d]   <= dly_p[d-1];
                dly_maj[d] <= dly_maj[d-1];
                dly_fn[d]  <= dly_fn[d-1];
            end
        end
    end

    wire [7:0] out_maj = dly_maj[FP_LAT-1];
    wire [7:0] out_fn  = dly_fn[FP_LAT-1];

    // Which of the two sources carries this result: the FMA array for the
    // arithmetic opcodes, the delayed combinational path for everything else.
    wire out_from_fma = (out_maj == `TNCA_MAJ_VFP) &&
                        ((out_fn == `TNCA_VF_ADD) || (out_fn == `TNCA_VF_SUB) ||
                         (out_fn == `TNCA_VF_MUL) || (out_fn == `TNCA_VF_FMA) ||
                         (out_fn == `TNCA_VF_FMS));

    always @(*) begin
        vres = {(VLEN*ELEN){1'b0}};
        sres = 64'd0;
        pres = dly_p[FP_LAT-1];
        if (out_from_fma) begin
            for (i = 0; i < VLEN; i = i + 1)
                vres[i*ELEN +: ELEN] = fma_y[i];
        end else begin
            vres = dly_v[FP_LAT-1];
        end
    end

    // A reduction broadcasts lane 0; an SFU pass returns its own vector. The
    // two are never done in the same cycle.
    assign mc_sres = {{(64-ELEN){1'b0}}, acc[0]};
    generate
        for (g = 0; g < VLEN; g = g + 1) begin : g_redbc
            assign mc_vres[g*ELEN +: ELEN] =
                sfu_done ? sfu_vres[g*ELEN +: ELEN] : acc[0];
        end
    endgenerate

endmodule
