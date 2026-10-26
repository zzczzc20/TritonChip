// Compact contiguous-vector LSU: one shared planner, finite response landing
// slots, one registered memory request and one vector RF completion port.
// A surrounding dispatcher must retain the narrow LSU for unsupported work.
`include "tnca_defs.vh"
`default_nettype none
module tnca_lsu_line #(
    parameter SLOTS=8,
    parameter TAG_W=3,
    parameter RCTX_W=2,
    parameter [63:0] TMU_BASE=64'h7fff0000
)(
    input wire clk, rst_n,
    input wire req_valid,
    output wire req_eligible, req_ready, quiescent,
    input wire req_store, req_vector, req_indexed, req_amo,
    input wire [2:0] req_size,
    input wire req_signed,
    input wire [1:0] req_hint,
    input wire [63:0] req_base, req_offset,
    input wire [511:0] req_vdata,
    input wire [15:0] req_mask,
    input wire [RCTX_W-1:0] req_rctx,
    input wire [2:0] req_dst_class,
    input wire [7:0] req_dst_idx,
    output wire cpl_valid,
    input wire cpl_ack,
    output wire [RCTX_W-1:0] cpl_rctx,
    output wire [2:0] cpl_dst_class,
    output wire [7:0] cpl_dst_idx,
    output reg [511:0] cpl_vdata,
    output wire [63:0] cpl_sdata,
    output wire [15:0] cpl_lane_en,
    output wire mem_req_valid,
    input wire mem_req_ready,
    output reg [63:0] mem_req_addr,
    output reg mem_req_we,
    output reg [63:0] mem_req_be,
    output reg [511:0] mem_req_wdata,
    output reg [TAG_W-1:0] mem_req_tag,
    output reg [1:0] mem_req_hint,
    input wire mem_rsp_valid,
    input wire [TAG_W-1:0] mem_rsp_tag,
    input wire [511:0] mem_rsp_rdata,
    output reg [31:0] perf_accepted, perf_completed, perf_request_stall,
    output reg [31:0] perf_slot_stall, perf_peak
);
    wire [63:0] address=req_base+req_offset;
    wire [7:0] span=8'd16<<req_size;
    assign req_eligible=req_vector && !req_indexed && !req_amo && req_size<=2
                        && ({2'b0,address[5:0]}+span<=8'd64)
                        && address[63:33]==0 && address[63:12]!=TMU_BASE[63:12];
    reg [SLOTS-1:0] valid,done,issued;
    reg q_valid;
    reg free_found;
    reg [TAG_W-1:0] free_tag;
    always @* begin
        free_found=0;free_tag=0;
        for(integer i=SLOTS-1;i>=0;i=i-1)
            if(!valid[i]) begin free_found=1;free_tag=i;end
    end
    assign req_ready=free_found && (!q_valid || mem_req_ready);
    wire admit=req_valid && req_eligible && req_ready;
    assign quiescent=valid==0 && !q_valid;
    assign mem_req_valid=q_valid;

    // One packing/mask network shared by every outstanding command.
    reg [511:0] packed_data;
    reg [63:0] packed_mask;
    always @* begin
        packed_data=0;packed_mask=0;
        for(integer i=0;i<16;i=i+1) begin
            case(req_size)
                0: begin packed_data[i*8+:8]=req_vdata[i*32+:8];packed_mask[i]=req_mask[i];end
                1: begin packed_data[i*16+:16]=req_vdata[i*32+:16];packed_mask[i*2+:2]={2{req_mask[i]}};end
                default: begin packed_data[i*32+:32]=req_vdata[i*32+:32];packed_mask[i*4+:4]={4{req_mask[i]}};end
            endcase
        end
    end
    reg [511:0] landing[0:SLOTS-1];
    reg [5:0] offset[0:SLOTS-1];
    reg [1:0] size[0:SLOTS-1];
    reg signed_load[0:SLOTS-1],store[0:SLOTS-1];
    reg [15:0] mask[0:SLOTS-1];
    reg [RCTX_W-1:0] rctx[0:SLOTS-1];
    reg [2:0] dst_class[0:SLOTS-1];
    reg [7:0] dst_idx[0:SLOTS-1];
    reg [TAG_W-1:0] rr,pick,held_pick;
    reg found,held;
    integer candidate;
    always @* begin
        found=0;pick=0;candidate=0;
        for(integer n=0;n<SLOTS;n=n+1) begin
            candidate=(rr+n)%SLOTS;
            if(!found && done[candidate]) begin found=1;pick=candidate;end
        end
        if(held) begin found=1;pick=held_pick;end
    end
    assign cpl_valid=found;
    wire retire=cpl_valid && cpl_ack;
    assign cpl_rctx=rctx[pick];
    assign cpl_dst_class=store[pick] ? `TNCA_DST_NONE : dst_class[pick];
    assign cpl_dst_idx=dst_idx[pick];
    assign cpl_lane_en=store[pick] ? 16'd0 : mask[pick];
    assign cpl_sdata={{32{signed_load[pick] && cpl_vdata[31]}},cpl_vdata[31:0]};
    reg [511:0] shifted;
    reg [31:0] value;
    always @* begin
        shifted=landing[pick]>>(offset[pick]*8);
        cpl_vdata=0;value=0;
        for(integer i=0;i<16;i=i+1) begin
            case(size[pick])
                0: value=signed_load[pick] ? {{24{shifted[i*8+7]}},shifted[i*8+:8]} : {24'd0,shifted[i*8+:8]};
                1: value=signed_load[pick] ? {{16{shifted[i*16+15]}},shifted[i*16+:16]} : {16'd0,shifted[i*16+:16]};
                default: value=shifted[i*32+:32];
            endcase
            if(!store[pick] && mask[pick][i]) cpl_vdata[i*32+:32]=value;
        end
    end
    reg [TAG_W:0] count;
    always @(posedge clk) begin
        if(!rst_n) begin
            valid<=0;done<=0;issued<=0;q_valid<=0;rr<=0;held<=0;held_pick<=0;count<=0;
            perf_accepted<=0;perf_completed<=0;perf_request_stall<=0;perf_slot_stall<=0;perf_peak<=0;
            mem_req_addr<=0;mem_req_we<=0;mem_req_be<=0;mem_req_wdata<=0;mem_req_tag<=0;mem_req_hint<=0;
        end else begin
            if(q_valid && mem_req_ready) begin q_valid<=0;issued[mem_req_tag]<=1;end
            if(admit) begin
                valid[free_tag]<=1;done[free_tag]<=req_mask==0;issued[free_tag]<=0;
                offset[free_tag]<=address[5:0];size[free_tag]<=req_size[1:0];
                signed_load[free_tag]<=req_signed;store[free_tag]<=req_store;
                mask[free_tag]<=req_mask;rctx[free_tag]<=req_rctx;
                dst_class[free_tag]<=req_dst_class;dst_idx[free_tag]<=req_dst_idx;
                if(req_mask!=0) begin
                    q_valid<=1;mem_req_addr<={address[63:6],6'd0};mem_req_we<=req_store;
                    mem_req_be<=packed_mask<<address[5:0];
                    mem_req_wdata<=req_store ? packed_data<<(address[5:0]*8) : 512'd0;
                    mem_req_tag<=free_tag;mem_req_hint<=req_hint;
                end
                perf_accepted<=perf_accepted+1;
            end
            if(mem_rsp_valid) begin done[mem_rsp_tag]<=1;landing[mem_rsp_tag]<=mem_rsp_rdata;end
            if(retire) begin
                valid[pick]<=0;done[pick]<=0;issued[pick]<=0;rr<=pick+1'b1;held<=0;
                perf_completed<=perf_completed+1;
            end else if(found && !held) begin held<=1;held_pick<=pick;end
            case({admit,retire})
                2'b10: count<=count+1'b1;
                2'b01: count<=count-1'b1;
                default: count<=count;
            endcase
            if(count+admit-retire>perf_peak) perf_peak<=count+admit-retire;
            if(q_valid && !mem_req_ready) perf_request_stall<=perf_request_stall+1;
            if(req_valid && req_eligible && !free_found) perf_slot_stall<=perf_slot_stall+1;
        end
    end
`ifndef SYNTHESIS
    initial if((SLOTS!=2 && SLOTS!=4 && SLOTS!=8) || SLOTS!=(1<<TAG_W))
        $fatal(1,"line LSU slots must match finite tag space (2|4|8)");
    always @(posedge clk) if(rst_n) begin
        if(mem_rsp_valid && (!valid[mem_rsp_tag] || !issued[mem_rsp_tag] || done[mem_rsp_tag])) $fatal(1,"line LSU invalid/duplicate response");
        if(admit && valid[free_tag]) $fatal(1,"line LSU slot reuse");
        if(retire && !valid[pick]) $fatal(1,"line LSU invalid completion");
    end
`endif
endmodule
`default_nettype wire
