// Observation only. Compile as a second top with tb_nuca_slice; never synthesize.
// This records the accepted NUCA boundary, not the core issue boundary.
`timescale 1ns/1ps
module tnca_tlm_capture;
    integer file=0, cycle=0, sequence_id=0;
    reg [4095:0] path;
    initial begin
        if (!$value$plusargs("tlm_capture=%s",path)) $fatal(1,"missing +tlm_capture");
        file=$fopen(path,"w");
        if(!file)$fatal(1,"cannot create TLM capture");
        $fdisplay(file,"TNCA_NUCA_CAPTURE 1");
    end
    always @(posedge tb_nuca_slice.clk) begin
        cycle=cycle+1;
        if(tb_nuca_slice.rst_n)begin
            if(tb_nuca_slice.req_valid && tb_nuca_slice.req_ready)begin
                sequence_id=sequence_id+1;
                $fdisplay(file,"Q %0d %0d %0d %0d %0d %016h %016h %02h",
                    sequence_id,cycle,tb_nuca_slice.req_src,tb_nuca_slice.req_tag,
                    tb_nuca_slice.req_we,tb_nuca_slice.req_addr,tb_nuca_slice.req_wdata,tb_nuca_slice.req_be);
            end
            if(tb_nuca_slice.rsp_valid && tb_nuca_slice.rsp_ready)
                $fdisplay(file,"P %0d %0d %0d %016h",cycle,tb_nuca_slice.rsp_src,
                    tb_nuca_slice.rsp_tag,tb_nuca_slice.rsp_rdata);
        end
    end
    final begin
        if(file)begin $fdisplay(file,"END %0d %0d",sequence_id,cycle);$fclose(file);end
    end
endmodule
