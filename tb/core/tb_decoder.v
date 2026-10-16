//==============================================================================
// tb_decoder -- checks that tnca_decoder accepts exactly the encodings
// doc/isa/TNCA-ISA-v0.1.md defines, and rejects everything else.
//
// The reserved-bit rule (ISA sec 2.3) is only worth anything if it is enforced,
// so the vector file is mostly mutations of valid instructions: one reserved
// field set, one predicate on an unpredicated family, one out-of-range register
// index. Each must raise ILLEGAL_INSTRUCTION rather than being ignored.
//
// Vector file: "<insn64> <expect_illegal> <label>" per line.
//==============================================================================
`timescale 1ns/1ps

module tb_decoder;

    reg  [63:0] insn;
    wire        illegal;
    wire [2:0]  fu, dstc;
    wire [7:0]  dsti, major, func;
    wire [9:0]  mod_bits;

    tnca_decoder u_dec (
        .insn(insn),
        .o_illegal(illegal), .o_fu(fu),
        .o_major(major), .o_func(func), .o_mod(mod_bits),
        .o_dst_class(dstc), .o_dst_idx(dsti),
        .o_sa_en(), .o_sa_idx(), .o_sb_en(), .o_sb_idx(),
        .o_sc_en(), .o_sc_idx(),
        .o_va_en(), .o_va_idx(), .o_vb_en(), .o_vb_idx(),
        .o_vc_en(), .o_vc_idx(),
        .o_pa_en(), .o_pa_idx(), .o_pb_en(), .o_pb_idx(),
        .o_pm_en(), .o_pm_idx(), .o_pm_neg(), .o_mask_dst(),
        .o_imm(),
        .o_is_branch(), .o_br_cond(), .o_br_on_zero(),
        .o_br_indirect(), .o_br_link(), .o_br_off(),
        .o_is_exit(), .o_is_trap(), .o_is_yield(),
        .o_mem_en(), .o_mem_store(), .o_mem_vector(), .o_mem_indexed(),
        .o_mem_size(), .o_mem_signed(), .o_mem_hint()
    );

    integer fh, r, n, fails, expect;
    reg [1023:0] vecfile;
    reg [255:0]  label;

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        fh = $fopen(vecfile, "r");
        if (fh == 0) begin $display("FATAL: cannot open %0s", vecfile); $finish; end

        n = 0; fails = 0; r = 3;
        while (r == 3) begin
            r = $fscanf(fh, "%h %d %s", insn, expect, label);
            if (r == 3) begin
                #1;
                n = n + 1;
                if (illegal !== expect[0]) begin
                    fails = fails + 1;
                    if (fails <= 25)
                        $display("FAIL insn=%h want_illegal=%0d got=%0d  (%0s)",
                                 insn, expect, illegal, label);
                end
                // A rejected instruction must present no side effect.
                if (illegal && (fu !== 3'd0 || dstc !== 3'd0)) begin
                    fails = fails + 1;
                    $display("FAIL insn=%h rejected but fu=%0d dstc=%0d (%0s)",
                             insn, fu, dstc, label);
                end
                #1;
            end
        end
        $fclose(fh);
        // A speculative fetch past an exit can contain unspecified bits.
        // Illegal still reaches the trap pipeline, but raw X opcode bits
        // must not select FPU/multiply resources in another RCTX's pipeline.
        insn = 64'hxxxxxxxxxxxxxxxx;
        #1;
        if (illegal !== 1'b1 || fu !== 3'd0 || dstc !== 3'd0 ||
            major !== 8'd0 || func !== 8'd0 || mod_bits !== 10'd0) begin
            fails = fails + 1;
            $display("FAIL unspecified illegal opcode leaked execution classification");
        end
        insn = 64'hff00000000000000;
        #1;
        if (illegal !== 1'b1 || major !== 8'd0 || func !== 8'd0 || mod_bits !== 10'd0) begin
            fails = fails + 1;
            $display("FAIL known illegal opcode did not retain inert classification");
        end
        $display("tb_decoder: %0d vectors, %0d failures", n, fails);
        if (fails == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

endmodule
