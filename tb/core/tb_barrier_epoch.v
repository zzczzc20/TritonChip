`timescale 1ns/1ps
// Public-port regression: the 65536th generation must wait for its command.
module tb_barrier_epoch;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, arm=0, done=0, sleep=0, init=0;
`ifdef EPOCH_EXHAUST
    localparam GENERATIONS=15;
`else
    localparam GENERATIONS=65537;
`endif
    reg [63:0] token=0;
    wire [63:0] next_token;
    wire complete, error, fault;
    wire [1:0] wake, wake_error;
    tnca_barrier #(.BAR_N(2), .BAR_W(1)
`ifdef EPOCH_EXHAUST
        , .EPOCH_W(4)
`endif
    ) dut (
        .clk(clk), .rst_n(rst_n), .flush(1'b0),
        .init_en(init), .init_bar(1'b0),
        .arm_en(arm), .arm_bar(1'b0), .arm_arrivals(8'd0), .arm_operations(8'd1),
        .arm_token(next_token), .arrive_en(1'b0), .arrive_bar(1'b0),
        .mtx_free_en(1'b0), .mtx_free_bar(1'b0),
        .mtx_done_en(1'b0), .mtx_done_bar(1'b0),
        .txf_done_en(done), .txf_done_err(1'b0), .txf_done_bar(1'b0),
        .inval_en(1'b0), .inval_bar(1'b0),
        .test_token(token), .test_done(complete), .test_error(error),
        .sleep_en(sleep), .sleep_rctx(1'b0), .sleep_token(token),
        .wake(wake), .wake_err(wake_error), .fault(fault));
    integer i;
    initial begin
        repeat(2) @(negedge clk);
        rst_n=1;
        #1; if (!complete) $fatal(1,"quiescent generation not complete");
        for(i=1;i<=GENERATIONS;i=i+1) begin
            @(negedge clk); arm=1;
            #1; token=next_token;
            if(token != ({32'd0,i[31:0]}<<16)) $fatal(1,"generation truncated at %0d",i);
            @(negedge clk); arm=0;
            #1; if(complete || error || fault) $fatal(1,"premature completion at %0d",i);
            sleep=(i>=65536);
            done=1;
            @(negedge clk); done=0; sleep=0;
            @(negedge clk);
            #1; if(!complete || error || fault) $fatal(1,"completion missing at %0d",i);
            if(i>=65536 && (!wake[0] || wake_error[0])) $fatal(1,"waiter not released at %0d",i);
        end
`ifdef EPOCH_EXHAUST
        @(negedge clk); arm=1;
        @(negedge clk); arm=0;
        #1; if(!fault || !error) $fatal(1,"epoch exhaustion must fault");
        init=1;
        @(negedge clk); init=0; token=0;
        #1; if(error || !complete || next_token!=(64'd1<<16))
            $fatal(1,"quiescent init must restore generation one");
        $display("PASS barrier exhaustion faults before wrap; init recovers");
`else
        token=64'd65535<<16;
        #1; if(!complete) $fatal(1,"late completed token must remain complete");
        $display("PASS barrier generations 0..65537: no early release at 16-bit boundary");
`endif
        $finish;
    end
endmodule
