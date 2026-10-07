//==============================================================================
// tnca_fpga_mem -- the device's memory, as RAM rather than as a model
//
// doc/phase3_6_tasks_and_acceptance.md R36-MEM-01 and R36-MEM-02.
//
// The two memories this repository has run the core against are behavioural:
// tb/common/tnca_mem_model.v keeps the whole array in flip-flops and
// tb/common/tnca_mem_ref.v keeps it as a byte array. Both are correct and
// neither is hardware. This is the third implementation behind the same port
// list, and the first one that a synthesizer has an opinion about.
//
// R36-MEM-01 asks for the accounting below, item by item, so that "the memory
// works" is a claim with contents. R36-MEM-02 asks that a second implementation
// complete a real regression through the same boundary; tnca_mem_ref.v is that
// one and this is a third, which is why the list is written out rather than
// assumed.
//
// ---------------------------------------------------------------------------
// CAPACITY
//   WORDS x 64 bits. The default, 65536 words, is 512 KiB, and that number is
//   not a choice made here: it is the size of the machine the profile
//   describes. tb/device/tnca_device.v declares `parameter WORDS = 65536` for
//   exactly this profile, and `Session`'s heap limit defaults to 0x80000,
//   which is that many words to the byte. A board a quarter of the size is not
//   the same machine: an artifact laid out for the profile puts its heap at
//   0x20000, past the end of a 128 KiB memory, and its first upload is refused
//   -- which is how this default came to be what it is.
//
//   The parameter is also the range check: an address at or past WORDS reads
//   as zero and writes nothing, exactly as both models do.
//
//   ADDR_W must satisfy WORDS == 2**ADDR_W. The read port is indexed by a
//   truncated address, so a WORDS that is not a power of two would make the
//   truncation wrong in a way the range check cannot see -- an address of
//   WORDS would fold back onto word zero and be answered from it. The two
//   parameters are one parameter written twice, and this is the paragraph that
//   says so.
//
// PORTS
//   One fetch port, read only, addressed in bytes and answered in words. One
//   data port, read and write, with the valid/ready handshake and the tag and
//   byte enables of tb/common/tnca_mem_model.v.
//
//   Both ports reach the SAME array, and that is a decision rather than an
//   accident. Splitting code from data into two arrays would make instruction
//   storage simpler to infer and would silently change a property the core
//   currently has: in both models a store to an address is visible to a
//   subsequent fetch of it. tnca_mem_ref.v:45-48 records that the contract
//   does not promise this -- but it also does not forbid it, and a device that
//   quietly lost it would be a different machine from the one every existing
//   test was written against. So the fetch port reads the array the data port
//   writes, and the cost is one dual-port block instead of two single-port
//   ones.
//
// LATENCY
//   Fetch: one cycle, registered, no backpressure -- the address presented in
//   cycle N returns its word in cycle N+1, which is the stage tnca_core's
//   fetch buffer is built around.
//
//   Data: two cycles from an accepted request to a valid response. Not one,
//   because the read value has to be registered before it can be handed back,
//   and not variable, because a variable one would make every cycle count the
//   core reports a function of traffic rather than of the program. Both models
//   are two cycles at their default LATENCY, so a core that runs against this
//   one and against those reports the numbers it always did.
//
//   Throughput is one request per cycle except while an atomic is executing,
//   which is the one operation that cannot finish in one.
//
// MASK
//   Eight byte enables, one per byte of the 64-bit word. A write changes only
//   the bytes whose enable is set, and an enable of zero is a write that
//   stores nothing. This is what the byte array in tnca_mem_ref.v exists to
//   check: there, a wrong enable is a wrong byte, not the same wrong word
//   twice.
//
// READ-DURING-WRITE
//   READ FIRST. When a write and a read of the same address land on the data
//   port in one cycle, the read returns the OLD word and the new one is stored
//   behind it. This is the whole reason the normal path can be one request per
//   cycle: a write does not have to be split into a read and a write to find
//   out what it overwrote, because the port hands the old value over as a
//   matter of course.
//
//   It is also the behaviour the core already depends on. Both models capture
//   the read value AT ACCEPT and answer with it (tnca_mem_ref.v:182-188 says
//   so in as many words): a later write to the same address must not change an
//   answer that was already determined.
//
//   The coding style below is the one that infers it: the read and the write
//   are separate clocked blocks on the same edge, so neither observes the
//   other's effect and the read returns what stood there before. See the note
//   beside the two blocks -- the version that put them in one block read more
//   clearly and was not recognised as a memory at all.
//
// INITIALISATION
//   THERE IS NONE. This is the one place where this module deliberately does
//   NOT match the models, and the difference is not cosmetic.
//
//   Both models zero their array in an `initial` block. That is a simulation
//   affordance: it costs nothing, and it means a kernel that reads a location
//   it never wrote gets zero rather than an X. Real RAM does not do this. A
//   block RAM comes out of configuration holding whatever it holds, and a
//   bring-up flow that assumed otherwise would work in simulation and produce
//   garbage on the board for exactly the kernels that read before they write.
//
//   The synthesizable set forbids `initial` anyway -- tools/r36/rtl_tests.py
//   treats one as a finding -- so the rule and the tool agree here by
//   construction. What replaces it is a responsibility, and it belongs to the
//   host: memory the host intends the kernel to read must be uploaded before
//   the kernel is launched. The upload path is the payload window in
//   rtl/fpga/tnca_mmio.v, which is the same path a kernel's results come back
//   through, so the flow that has to be right for correctness is the flow that
//   is exercised on every transfer rather than only on the first one.
//
//   R36-HAND-02 asks that a storage or interface gap not be dressed up as a
//   board-level task. This is the gap, stated here rather than discovered
//   later: a kernel whose output depends on uninitialised memory is correct in
//   simulation and undefined on hardware, and no amount of board work changes
//   that. It is handled by initialising from the host, which is what a real
//   accelerator does.
//
// ATOMICS
//   Handled here, from rtl/common/tnca_amo.vh -- the same three functions the
//   four-tile home slice uses, because two definitions that disagreed would
//   mean one of the two test suites was checking the wrong machine.
//
//   An atomic needs the word read, modified and written back, which is two
//   accesses to a port that does one. So an atomic is the single case where
//   the data port stalls: `req_ready` goes low for the cycle the writeback
//   needs, and one request in two is accepted while an atomic is in flight. A
//   request accepted then would have been a second access to an array whose
//   port was still busy, and the failure mode of allowing it is a lost write
//   that only shows up under contention.
//
//   The response carries the OLD word, which is what both models return and
//   what the requester extracts its result from. `TNCA_AOP_LD` reads and does
//   not write back.
//
// REPLACEABILITY
//   R36-MEM-01 asks what may be swapped without disturbing anything else. The
//   port list above is the boundary and it is the whole boundary: capacity is
//   WORDS, latency is fixed at two and is not promised to anything that would
//   notice if a future implementation made it four, the tag width is TAG_W,
//   and nothing outside this module sees the array, the address decode or the
//   read-during-write mode.
//
//   What is NOT swappable without saying so is the read-during-write mode.
//   Write-first would change what a store's response carries and would break
//   the accept-time capture both models document. A future implementation may
//   change it, but it must change this paragraph in the same commit.
//==============================================================================
`default_nettype none

module tnca_fpga_mem #(
    // WORDS must be 2**ADDR_W. See CAPACITY above.
    parameter WORDS     = 65536,
    parameter ADDR_W    = 16,
    parameter MEM_DW    = 64,
    parameter MEM_BYTES = 8,
    parameter TAG_W     = 5
) (
    input  wire                     clk,
    input  wire                     rst_n,

    //--------------------------------------------------------------------------
    // Fetch port: byte address in, word out, one cycle, no handshake
    //--------------------------------------------------------------------------
    input  wire [63:0]              imem_addr,
    output reg  [MEM_DW-1:0]        imem_data,

    //--------------------------------------------------------------------------
    // Data port
    //--------------------------------------------------------------------------
    input  wire                     req_valid,
    output wire                     req_ready,
    input  wire [63:0]              req_addr,
    input  wire                     req_we,
    input  wire [MEM_BYTES-1:0]     req_be,
    input  wire [MEM_DW-1:0]        req_wdata,
    input  wire [TAG_W-1:0]         req_tag,
    input  wire                     req_amo,
    input  wire [3:0]               req_aop,
    input  wire [MEM_DW-1:0]        req_cmp,

    output reg                      rsp_valid,
    output reg  [TAG_W-1:0]         rsp_tag,
    output reg  [MEM_DW-1:0]        rsp_rdata
);

`include "tnca_amo.vh"

    // MEM_WORDS is `WORDS`, widened once, and both range checks below measure
    // against it.
    //
    // `WORDS` is a parameter and a parameter's width is the width of the value
    // it was given -- 32 bits for a value this size. `WORDS[63:0]` asks for bits
    // that do not exist, and a tool is free to answer with X. A comparison
    // against X is X, and `if (x)` is false, so the range check would have
    // taken the out-of-range branch at every address: the fetch port answering
    // zero for the whole array, the data port storing nothing, and the refusal
    // looking exactly like a correctly-configured boundary tested one word too
    // low.
    //
    // That is the failure this comment records. It was found at the other end
    // of the design -- rtl/fpga/tnca_mmio.v has the same select in its payload
    // window, and there every access was refused, which is what a range check
    // that answers X looks like from outside. Fixing it there sent me here.
    localparam [63:0] MEM_WORDS = WORDS;

    // Declared before anything reads it. iverilog 13.0 refuses a use that
    // precedes a declaration and 12.0 is the version this repository pins,
    // but a file that only elaborates under one of the two is a file that
    // stops being testable the day the pin moves.
    integer t_j;

    //==========================================================================
    // The array
    //
    // Written in exactly one always block and read in two. That is what a
    // dual-port block RAM is, and keeping the write in a single block is what
    // lets a synthesizer recognise it -- two writers would be a register file
    // holding a megabit in flip-flops.
    //==========================================================================
    reg [MEM_DW-1:0] mem [0:WORDS-1];

    //==========================================================================
    // The data port's cycle, worked out combinationally
    //
    // `amo_wb` is high for the one cycle between an atomic being accepted and
    // its writeback, and it is the only thing that makes the port busy.
    //==========================================================================
    reg amo_wb;

    reg [ADDR_W-1:0]    amo_addr_q;
    reg [MEM_BYTES-1:0] amo_be_q;
    reg [MEM_DW-1:0]    amo_wdata_q;
    reg [3:0]           amo_aop_q;
    reg [MEM_DW-1:0]    amo_cmp_q;
    reg [TAG_W-1:0]     amo_tag_q;
    reg                 amo_range_q;

    assign req_ready = !amo_wb;

    // The address is 64-bit and the array is not, so the range check is made
    // on the full width before anything is truncated. A check made after
    // truncation is a check that an address of 0x1_0000_0000 passes.
    wire req_range = (req_addr[63:3] < MEM_WORDS);

    wire accept = req_valid && req_ready;

    // The data port's read register, READ FIRST. Declared here, before the
    // arithmetic that reads it, because a declaration that follows its own use
    // is a file that stops elaborating when the toolchain pin moves.
    reg [MEM_DW-1:0] dout_q;

    //--------------------------------------------------------------------------
    // The atomic's arithmetic, from the shared header
    //--------------------------------------------------------------------------
    // Combinational from `dout_q`, which holds the old word and is stable for
    // the whole writeback cycle. The result is stored under the same byte
    // enables the request carried, which is how a 4-byte atomic in a 64-bit
    // word lands in the half it was addressed at.
    wire [MEM_DW-1:0] amo_res = amo_new(amo_aop_q, amo_be_q, dout_q,
                                        amo_wdata_q, amo_cmp_q);

    wire amo_do_write = amo_range_q && (amo_aop_q != `TNCA_AOP_LD);

    //--------------------------------------------------------------------------
    // What the port is doing this cycle. Wires, not registers: a version of
    // this that latched them would be a second driver on the same names, and
    // the two would disagree for exactly one cycle -- the one after an accept.
    //--------------------------------------------------------------------------
    wire [ADDR_W-1:0]    b_raddr = amo_wb ? amo_addr_q : req_addr[ADDR_W+2:3];
    wire                 b_we    = amo_wb ? amo_do_write
                                          : (accept && req_we && !req_amo
                                             && req_range);
    wire [ADDR_W-1:0]    b_waddr = amo_wb ? amo_addr_q : req_addr[ADDR_W+2:3];
    wire [MEM_DW-1:0]    b_wdata = amo_wb ? amo_res : req_wdata;
    wire [MEM_BYTES-1:0] b_be    = amo_wb ? amo_be_q : req_be;

    //==========================================================================
    // The fetch port
    //
    // Registered and unconditional, with no ready to withhold: the core
    // presents an address and expects the word beside it on the next edge.
    // An address past the end reads zero rather than aliasing onto a word
    // that exists.
    //==========================================================================
    always @(posedge clk) begin
        if (imem_addr[63:3] < MEM_WORDS)
            imem_data <= mem[imem_addr[ADDR_W+2:3]];
        else
            imem_data <= {MEM_DW{1'b0}};
    end

    //==========================================================================
    // The array, and the response pipeline
    //==========================================================================
    reg              v1;
    reg              range1;
    reg [TAG_W-1:0]  tag1;

    //--------------------------------------------------------------------------
    // The array's two access blocks
    //
    // ONE BLOCK READS, ANOTHER WRITES, and they are kept apart on purpose.
    // Both are clocked on the same edge and neither can observe the other's
    // effect within the cycle, so the read still returns the OLD word -- the
    // READ-FIRST contract is unchanged. What changes is whether a synthesizer
    // recognises a memory at all.
    //
    // It was written as one block first, read and write adjacent, which is the
    // spelling that makes the read-before-write ordering visible to a reader.
    // yosys replaced the array with a list of registers and said so:
    //
    //     Warning: Replacing memory \mem with list of registers.
    //     See rtl/fpga/tnca_fpga_mem.v:309
    //
    // A read and a write to the same array in the same block is not a form the
    // frontend maps to a memory cell, and 65536 x 64 bits of registers is four
    // million flip-flops. `proc` then did not finish in fifteen minutes, which
    // would have been a gate that never returns rather than a gate that fails.
    // Splitting them is not a workaround for a tool: separate read and write
    // ports is what a block RAM is, and the ordering argument above is the one
    // that genuinely holds.
    //--------------------------------------------------------------------------
    always @(posedge clk)
        dout_q <= mem[b_raddr];

    always @(posedge clk)
        if (b_we)
            for (t_j = 0; t_j < MEM_BYTES; t_j = t_j + 1)
                if (b_be[t_j])
                    mem[b_waddr][t_j*8 +: 8] <= b_wdata[t_j*8 +: 8];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1        <= 1'b0;
            range1    <= 1'b0;
            tag1      <= {TAG_W{1'b0}};
            rsp_valid <= 1'b0;
            rsp_tag   <= {TAG_W{1'b0}};
            rsp_rdata <= {MEM_DW{1'b0}};
            amo_wb    <= 1'b0;
            amo_addr_q  <= {ADDR_W{1'b0}};
            amo_be_q    <= {MEM_BYTES{1'b0}};
            amo_wdata_q <= {MEM_DW{1'b0}};
            amo_aop_q   <= 4'd0;
            amo_cmp_q   <= {MEM_DW{1'b0}};
            amo_tag_q   <= {TAG_W{1'b0}};
            amo_range_q <= 1'b0;
        end else begin
            rsp_valid <= 1'b0;

            // ---- stage one: accept -----------------------------------------
            //
            // An atomic does not enter the ordinary pipeline. Its response is
            // built a cycle later, from the value the read is about to bring
            // back, and its slot in stage two is left for it -- which is why
            // `v1` is cleared here for an atomic rather than set.
            v1     <= accept && !req_amo;
            range1 <= req_range;
            tag1   <= req_tag;

            // ---- the atomic: latch, then write back -------------------------
            if (req_amo && accept) begin
                amo_addr_q  <= req_addr[ADDR_W+2:3];
                amo_be_q    <= req_be;
                amo_wdata_q <= req_wdata;
                amo_aop_q   <= req_aop;
                amo_cmp_q   <= req_cmp;
                amo_tag_q   <= req_tag;
                amo_range_q <= req_range;
                amo_wb      <= 1'b1;
            end else if (amo_wb) begin
                // `accept` is low this cycle, because `req_ready` is, so the
                // two branches cannot both be taken and this one always
                // clears what the other one set.
                amo_wb <= 1'b0;
            end

            // ---- stage two: respond -----------------------------------------
            //
            // Two sources and never both. An atomic accepted last cycle set
            // `v1` to zero on its way past, so the ordinary branch is silent
            // for exactly the cycle the atomic branch speaks, and responses
            // leave in the order the requests arrived.
            if (amo_wb) begin
                rsp_valid <= 1'b1;
                rsp_tag   <= amo_tag_q;
                // The OLD word: what the read brought back before the
                // writeback above stored over it, and what the requester
                // picks its result out of.
                rsp_rdata <= amo_range_q ? dout_q : {MEM_DW{1'b0}};
            end else if (v1) begin
                rsp_valid <= 1'b1;
                rsp_tag   <= tag1;
                // An address past the end answers zero, as both models do,
                // rather than a word that happens to sit at the truncated
                // index.
                rsp_rdata <= range1 ? dout_q : {MEM_DW{1'b0}};
            end
        end
    end

endmodule

`default_nettype wire
