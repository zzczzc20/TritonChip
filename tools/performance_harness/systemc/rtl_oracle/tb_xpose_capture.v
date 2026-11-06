// Observation only (R41 X8). Compile as a second top beside tb_tnca_core;
// never synthesize. Records the transfer engine's transposed-write sequencing
// (tnca_tensor.v S_XRUN / S_XDRAIN / S_XWRITE) for xpose_oracle.py: one line
// per cycle while a transposed command is in flight.
//   C <cycle> <esz> <R> <C> <state> <req> <rsp> <wr> <done>
`timescale 1ns/1ps
module tnca_xpose_capture;
    integer file = 0, cycle = 0;
    reg [4095:0] path;
    initial begin
        if (!$value$plusargs("xpose_capture=%s", path)) $fatal(1, "missing +xpose_capture");
        file = $fopen(path, "w");
        if (!file) $fatal(1, "cannot create capture");
        $fdisplay(file, "TNCA_XPOSE_CAPTURE 1");
    end
    always @(posedge tb_tnca_core.clk) begin
        cycle = cycle + 1;
        if (tb_tnca_core.rst_n && tb_tnca_core.dut.u_txf.t_xpose &&
            (tb_tnca_core.dut.u_txf.st != 5'd0 || tb_tnca_core.dut.u_txf.done))
            $fdisplay(file, "C %0d %0d %0d %0d %0d %0d %0d %0d %0d", cycle,
                tb_tnca_core.dut.u_txf.t_esz, tb_tnca_core.dut.u_txf.t_b3,
                tb_tnca_core.dut.u_txf.t_b4, tb_tnca_core.dut.u_txf.st,
                tb_tnca_core.dut.u_txf.mem_req_valid && tb_tnca_core.dut.u_txf.mem_req_ready,
                tb_tnca_core.dut.u_txf.mem_rsp_valid,
                tb_tnca_core.dut.u_txf.vw_en && tb_tnca_core.dut.u_txf.vw_ready,
                tb_tnca_core.dut.u_txf.done);
    end
    final if (file) begin $fdisplay(file, "END %0d", cycle); $fclose(file); end
endmodule
