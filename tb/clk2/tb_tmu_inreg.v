// tb_tmu_inreg -- tnca_tmu with IN_REG=1 against IN_REG=0 (REGISTER_WRITES 0 and 1).
// Three slices issue random register-write sequences (valid and fields held until ready, a
// random gap between writes). The register sets of the slices are disjoint, so the final
// state does not depend on the arbitration order: slice 0 POLICY/PARAMS, slice 1 GEAR/TILE,
// slice 2 staging BASE/SIZE/DESC with an occasional CLEAR. Checks: every write acknowledged
// exactly once in both, final outputs and STAT_REG equal; reports the cycle price.
`timescale 1ns/1ps
module tb_tmu_inreg;
    parameter NOPS = 300;
    localparam N = 3, SRC_W = 4, TENSORS = 8, DEAD = 16, ID_W = 16;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer errs = 0, cyc = 0;

    reg [11:0] op_off  [0:N-1][0:NOPS-1];
    reg [63:0] op_data [0:N-1][0:NOPS-1];
    integer s_i, o_i;
    initial begin
        for (s_i = 0; s_i < N; s_i = s_i + 1)
            for (o_i = 0; o_i < NOPS; o_i = o_i + 1) begin
                op_data[s_i][o_i] = {$urandom, $urandom};
                case (s_i)
                  0: op_off[s_i][o_i] = (o_i % 2) ? 12'h008 : 12'h000;
                  1: op_off[s_i][o_i] = (o_i % 3 == 0) ? 12'h040 : 12'h010;
                  default: begin
                      case (o_i % 4)
                        0: op_off[s_i][o_i] = 12'h100;
                        1: op_off[s_i][o_i] = 12'h108;
                        2: op_off[s_i][o_i] = 12'h110;
                        default: op_off[s_i][o_i] = (o_i % 16 == 15) ? 12'h018 : 12'h100;
                      endcase
                      if (o_i % 4 == 2) op_data[s_i][o_i][39:36] = $urandom % 4;
                      op_data[s_i][o_i][63:48] = 0;
                  end
                endcase
            end
    end

    genvar rw, v, s;
    generate
      for (rw = 0; rw <= 1; rw = rw + 1) begin : g_rw
        for (v = 0; v <= 1; v = v + 1) begin : g_v     // v = IN_REG
            reg  [N-1:0]       mwv;
            wire [N-1:0]       mwr;
            reg  [N*12-1:0]    mwoff;
            reg  [N*64-1:0]    mwdata;
            reg  [N*SRC_W-1:0] mwsrc;
            wire [N*64-1:0]    mrdata;
            wire [7:0] policy; wire [3:0] b_bits, gear_init; wire [5:0] d_lsb;
            wire [15:0] gw, gu, gl; wire oe;
            wire [TENSORS-1:0] tv, tby;
            wire [TENSORS*64-1:0] tb_, ts_;
            wire [TENSORS*65-1:0] te_;
            wire [TENSORS*6-1:0] tl_;
            wire [DEAD-1:0] dv; wire [DEAD*ID_W-1:0] did;
            tnca_tmu #(.N_SLICE(N),.SRC_W(SRC_W),.N_SRC(N),.TENSORS(TENSORS),.T_W(3),
                       .REGISTER_WRITES(rw),.IN_REG(v)) dut (
                .clk(clk),.rst_n(rst_n),
                .mw_valid(mwv),.mw_ready(mwr),.mw_off(mwoff),.mw_data(mwdata),.mw_src(mwsrc),
                .mr_off({12'h020,12'h020,12'h020}),.mr_data(mrdata),
                .tl_valid({N{1'b0}}),.tl_ready(),.tl_addr({N*64{1'b0}}),.tl_drop({N{1'b0}}),
                .policy(policy),.b_bits(b_bits),.d_lsb(d_lsb),.gear_init(gear_init),
                .gear_window(gw),.gear_ub(gu),.gear_lb(gl),.owner_epoch(oe),
                .t_valid(tv),.t_base(tb_),.t_size(ts_),.t_end(te_),.t_bypass(tby),.t_lsb(tl_),
                .dead_valid(dv),.dead_id(did));
            integer idx [0:N-1];
            integer gap [0:N-1];
            integer acks [0:N-1];
            integer done_cyc;
            integer k;
            initial begin
                mwv = 0; mwoff = 0; mwdata = 0; mwsrc = 0; done_cyc = -1;
                for (k = 0; k < N; k = k + 1) begin idx[k] = 0; gap[k] = 2; acks[k] = 0; end
            end
            always @(posedge clk) if (rst_n) begin
                for (k = 0; k < N; k = k + 1) begin
                    if (mwv[k] && mwr[k]) begin
                        mwv[k] = 1'b0; acks[k] = acks[k] + 1; idx[k] = idx[k] + 1;
                        gap[k] = $urandom % 5;
                    end else if (!mwv[k] && idx[k] < NOPS) begin
                        if (gap[k] == 0) begin
                            mwv[k] = 1'b1;
                            mwoff [k*12 +: 12] = op_off[k][idx[k]];
                            mwdata[k*64 +: 64] = op_data[k][idx[k]];
                            mwsrc [k*SRC_W +: SRC_W] = k;
                        end else gap[k] = gap[k] - 1;
                    end
                end
                if (done_cyc < 0 && acks[0] == NOPS && acks[1] == NOPS && acks[2] == NOPS) done_cyc = cyc;
            end
        end
        reg finished;
        initial begin
            finished = 0;
            wait (g_v[0].done_cyc >= 0 && g_v[1].done_cyc >= 0);
            repeat (20) @(posedge clk);
            if (g_v[0].policy !== g_v[1].policy || g_v[0].b_bits !== g_v[1].b_bits ||
                g_v[0].gear_init !== g_v[1].gear_init || g_v[0].d_lsb !== g_v[1].d_lsb ||
                g_v[0].gw !== g_v[1].gw || g_v[0].gu !== g_v[1].gu || g_v[0].gl !== g_v[1].gl ||
                g_v[0].oe !== g_v[1].oe || g_v[0].tv !== g_v[1].tv || g_v[0].tby !== g_v[1].tby ||
                g_v[0].tb_ !== g_v[1].tb_ || g_v[0].ts_ !== g_v[1].ts_ || g_v[0].te_ !== g_v[1].te_ ||
                g_v[0].tl_ !== g_v[1].tl_ || g_v[0].mrdata !== g_v[1].mrdata) begin
                errs = errs + 1;
                $display("MISMATCH rw=%0d policy %h/%h gw %h/%h tv %b/%b mr %h/%h", rw,
                    g_v[0].policy, g_v[1].policy, g_v[0].gw, g_v[1].gw, g_v[0].tv, g_v[1].tv,
                    g_v[0].mrdata[63:0], g_v[1].mrdata[63:0]);
            end
            $display("rw=%0d: %0d writes per slice acknowledged by cycle %0d (IN_REG=0) and %0d (IN_REG=1)", rw, NOPS,
                g_v[0].done_cyc, g_v[1].done_cyc);
            $display("   final state: policy %h b_bits %h gear %h/%h/%h tensors_valid %b stat_reg %0d", g_v[1].policy,
                g_v[1].b_bits, g_v[1].gw, g_v[1].gu, g_v[1].gl, g_v[1].tv, g_v[1].mrdata[63:0]);
            finished = 1;
        end
      end
    endgenerate

    initial begin repeat (4) @(posedge clk); rst_n = 1; end
    always @(posedge clk) cyc = cyc + 1;

    wire [1:0] fin = {g_rw[1].finished, g_rw[0].finished};
    initial begin
        wait (fin == 2'b11);
        if (errs == 0) $display("PASS tb_tmu_inreg"); else $display("FAIL tb_tmu_inreg errs=%0d", errs);
        $finish;
    end
    initial begin #50000000; $display("FAIL tb_tmu_inreg timeout"); $finish; end
endmodule
