//==============================================================================
// tnca_mmio -- tnca_shell's three ports, behind a flat register file
//
// doc/phase3_6_tasks_and_acceptance.md R36-RTL-02, R36-IO-01 and R36-IO-02.
//
// WHY THIS FILE EXISTS
//
// tnca_shell owns the timing contract and says so: the transport that carries
// commands is "a file mailbox today, an MMIO doorbell on a board", and it is a
// port rather than an assumption. This is the other side of that port -- the
// medium, and only the medium.
//
// The protocol does not change. spec/r2_transport_v1.json is a request/response
// mailbox with a monotonic sequence number: the host writes a descriptor and
// rings, the device reads it, acts, and writes a completion. Two text files
// become two register groups. tb/device/tnca_device.v does the same mapping in
// the opposite direction, file to port, and R2-10 is the clause that allows it.
//
// WHAT IS AND IS NOT INSIDE
//
//   inside:  the descriptor registers, the doorbell, the completion registers,
//            the payload window, the opcode decode and the range guards.
//
//   outside: everything the shell already does -- launch sequencing, the
//            deadline, the fault latch, the completion record and the
//            host/core arbitration. This module asks the shell; it does not
//            second-guess it. In particular the payload window issues one
//            `host_*` access and waits, and it is the shell that decides
//            whether the host is allowed to be asking at all.
//
// THE REGISTER MAP
//
// Byte offsets, 64-bit registers, little-endian as a bus would see them.
//
//   Mailbox, request side (the host writes these, then the doorbell)
//     0x00 REQ_SEQ       the sequence this request carries
//     0x08 REQ_OP        UPLOAD 1, DOWNLOAD 2, LAUNCH 3, QUIESCE 4,
//                        SHUTDOWN 5, PING 6, PERF_READ 7
//     0x10 REQ_A0        UPLOAD/DOWNLOAD: device address. LAUNCH: pc.
//                        PERF_READ: counter index (spec/r38/counters.json).
//     0x18 REQ_A1        UPLOAD/DOWNLOAD: word count. LAUNCH: arg base.
//     0x20 REQ_A2        LAUNCH: role mask.
//     0x28 REQ_A3        LAUNCH: pid_x | pid_y << 16
//     0x30 REQ_A4        LAUNCH: pid_z | npid_x << 16
//     0x38 REQ_A5        LAUNCH: npid_y | npid_z << 16
//     0x40 REQ_DEADLINE  cycles this command may take before the shell calls
//                        it TIMEOUT. Zero means no deadline.
//     0x48 DOORBELL      write anything to ring. Values written are ignored.
//
//   Mailbox, response side (the device writes these, the host reads them)
//     0x50 RSP_SEQ       the sequence this completion answers
//     0x58 RSP_STATUS    OK 0, FAULT 1, TIMEOUT 2, REFUSED 3
//     0x60 RSP_R0        FAULT: fault code. PERF_READ: low 32 bits of the
//                        counter. Otherwise: cycles.
//     0x68 RSP_R1        FAULT: fault pc. PERF_READ: high 32 bits of the
//                        counter. Otherwise: instructions retired.
//
//   Payload window -- one 64-bit word of device memory per access. This is the
//   medium the UPLOAD and DOWNLOAD payloads travel through, exactly as up.hex
//   and down.hex are theirs in simulation: a channel beside the mailbox, not
//   part of it.
//     0x70 HOST_ADDR     the device address of the next access. It advances by
//                        eight when an access completes, so a bulk transfer is
//                        a write to this register followed by a run of data.
//     0x78 HOST_DATA     a write here is the word to store; a read here is the
//                        word last fetched.
//     0x80 HOST_BE       the byte enables of the next access. Reset is 0xFF.
//     0x88 HOST_CTRL     write-only. bit 0 = write-enable, bit 1 = START.
//     0x90 HOST_STATUS   bit 0 = BUSY, bit 1 = VALID (a fetched word is in
//                        HOST_DATA), bit 2 = REFUSED (the address was outside
//                        device memory or not word-aligned; nothing was
//                        issued).
//
//   Identity
//     0x98 MAGIC         reads 0x0001_0000_544E_4341 -- version 1.0, "TNCA".
//                        A host that reads this knows it found a device and
//                        not an unprogrammed address space.
//     0xA0 SERVED        the highest sequence the device has acted on. PING
//                        reports it, and it is what the `seq > served` rule is
//                        applied against.
//
// RANGE GUARDS, AND WHERE THEY COME FROM
//
// The UPLOAD and DOWNLOAD guards are the ones in tb/device/tnca_device.v, kept
// deliberately identical: an address that is negative, not word-aligned, a
// count of zero, or a range that runs past the end of memory is REFUSED (3),
// and refused means nothing was touched. R36-IO-01 asks for exactly that --
// illegal input with no device-side effect.
//
// One of those guards was a real bug in the simulation binding and the shape of
// the fix is preserved here. tb/device/tnca_device.v:187-191 records it: the
// fields were `integer`, which is 32 bits and signed, so a host address of
// 0xFFFFFFF0 arrived as -16 and the range check read -16/8 + 1 = -1, which is
// not greater than WORDS -- an address far outside memory passing the guard
// that exists to catch it. Everything here is 64-bit unsigned and the bound is
// tested by comparison, not by sign.
//
// WHAT A BOARD CANNOT HAVE, AND WHAT THIS DOES INSTEAD
//
// SHUTDOWN ends the simulation process. A board has no process to end, so the
// opcode is acknowledged with OK and does nothing else: a device that cannot
// be shut down by a register write is a device a host cannot accidentally kill.
// PING returns `served`, which is what the simulation binding returns.
//
// PERF_READ (transport 1.1, R38 S1-02) reads one device counter through the
// shell's combinational `perf_sel`/`perf_rdata` port, split into two 32-bit
// halves exactly as tb/device/tnca_device.v answers it, so the runtime's
// one decoder serves both. It is served in MS_IDLE like every request that
// is not a launch, which is to say only while no command is running: a read
// never lands inside a launch's measurement window. An index past the last
// counter is REFUSED, not answered with zero -- a zero would be a reading.
//
// The `seq > served` rule is kept. In simulation it exists because the device
// polls a file and re-reads the request it has already acted on; a doorbell is
// edge-triggered and cannot do that. It is kept anyway because it is the
// protocol, not a property of the file, and a host that rings twice for one
// sequence gets one completion -- which is the answer R36-IO-03 wants.
//
// There is no $fopen, no $readmemh, no plusarg and no delay in this file.
// tools/r36/rtl_tests.py asks the same of every file in the synthesizable set.
//
// RESET
//
// The shell resets on `negedge rst_n`, so this does too. A board must drive
// that from a synchroniser rather than straight off a pin or a PLL lock; that
// synchroniser belongs to the platform wrapper and not here, and R36-RST-01
// asks for the responsibility to be written down rather than assumed.
//==============================================================================
`default_nettype none

module tnca_mmio #(
    // The register file is addressed in bytes and holds 64-bit registers.
    parameter ADDR_W = 8,
    // Device memory in 64-bit words. The UPLOAD and DOWNLOAD range guards
    // measure against this, and it must be the number the memory actually
    // has: a guard that is optimistic is a guard that passes the address
    // which walks off the end.
    parameter WORDS = 65536,
    // Forwarded to the shell through the command port.
    parameter NUM_RCTX = 2,
    parameter RCTX_W   = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,

    //--------------------------------------------------------------------------
    // Host register interface
    //
    // A write takes effect on the edge it is offered, so `reg_wdata` need only
    // be stable in that cycle. A read is combinational from `reg_addr`, and
    // `reg_rdata` is meaningful in the same cycle.
    //--------------------------------------------------------------------------
    input  wire [ADDR_W-1:0]        reg_addr,
    input  wire [63:0]              reg_wdata,
    input  wire                     reg_we,
    input  wire                     reg_re,
    output reg  [63:0]              reg_rdata,

    //--------------------------------------------------------------------------
    // To the shell: command port
    //--------------------------------------------------------------------------
    output reg                      cmd_valid,
    input  wire                     cmd_ready,
    output reg  [3:0]               cmd_op,
    output reg  [63:0]              cmd_pc,
    output reg  [63:0]              cmd_arg_base,
    output reg  [NUM_RCTX-1:0]      cmd_role_mask,
    output reg  [3:0]               cmd_slice,       // R39-05/12: role-mask bits 28, 31:29
    output reg  [31:0]              cmd_pid_x,
    output reg  [31:0]              cmd_pid_y,
    output reg  [31:0]              cmd_pid_z,
    output reg  [31:0]              cmd_npid_x,
    output reg  [31:0]              cmd_npid_y,
    output reg  [31:0]              cmd_npid_z,
    output reg  [63:0]              cmd_deadline,

    //--------------------------------------------------------------------------
    // From the shell: completion port
    //--------------------------------------------------------------------------
    input  wire                     done_valid,
    output reg                      done_ready,
    input  wire [3:0]               done_status,
    input  wire [63:0]              done_cycles,
    input  wire [63:0]              done_retired,
    input  wire [7:0]               done_fault_code,
    input  wire [63:0]              done_fault_pc,
    input  wire [RCTX_W-1:0]        done_fault_rctx,
    input  wire [23:0]              done_fault_detail,

    //--------------------------------------------------------------------------
    // To the shell: host access to device memory
    //
    // The five request outputs are wires and not registers, and that is a
    // correctness decision rather than a style one. See the note beside the
    // assignments below.
    //--------------------------------------------------------------------------
    output wire                     host_req_valid,
    input  wire                     host_req_ready,
    output wire                     host_req_we,
    output wire [63:0]              host_req_addr,
    output wire [7:0]               host_req_be,
    output wire [63:0]              host_req_wdata,
    input  wire                     host_rsp_valid,
    input  wire [63:0]              host_rsp_rdata,

    //--------------------------------------------------------------------------
    // To the shell: the counter read port (PERF_READ)
    //--------------------------------------------------------------------------
    output wire [4:0]               perf_sel,
    input  wire [63:0]              perf_rdata
);

`include "tnca_perf_count.vh"

    //==========================================================================
    // The opcodes and statuses, from transport.py and r2_transport_v1.json
    //==========================================================================
    localparam [3:0] OP_UPLOAD   = 4'd1;
    localparam [3:0] OP_DOWNLOAD = 4'd2;
    localparam [3:0] OP_LAUNCH   = 4'd3;
    localparam [3:0] OP_QUIESCE  = 4'd4;
    localparam [3:0] OP_SHUTDOWN = 4'd5;
    localparam [3:0] OP_PING     = 4'd6;
    localparam [3:0] OP_PERF_READ = 4'd7;

    localparam [3:0] RSP_OK      = 4'd0;
    localparam [3:0] RSP_FAULT   = 4'd1;
    localparam [3:0] RSP_TIMEOUT = 4'd2;
    localparam [3:0] RSP_REFUSED = 4'd3;

    // The shell's own command encoding. Not the same numbers as the mailbox
    // opcodes above and not meant to be: the mailbox is the wire, this is the
    // port, and conflating them is how one of the two gets changed alone.
    localparam [3:0] CMD_QUIESCE = 4'd2;

    //==========================================================================
    // Register offsets
    //==========================================================================
    localparam [ADDR_W-1:0] R_REQ_SEQ    = 8'h00;
    localparam [ADDR_W-1:0] R_REQ_OP     = 8'h08;
    localparam [ADDR_W-1:0] R_REQ_A0     = 8'h10;
    localparam [ADDR_W-1:0] R_REQ_A1     = 8'h18;
    localparam [ADDR_W-1:0] R_REQ_A2     = 8'h20;
    localparam [ADDR_W-1:0] R_REQ_A3     = 8'h28;
    localparam [ADDR_W-1:0] R_REQ_A4     = 8'h30;
    localparam [ADDR_W-1:0] R_REQ_A5     = 8'h38;
    localparam [ADDR_W-1:0] R_REQ_DEAD   = 8'h40;
    localparam [ADDR_W-1:0] R_DOORBELL   = 8'h48;
    localparam [ADDR_W-1:0] R_RSP_SEQ    = 8'h50;
    localparam [ADDR_W-1:0] R_RSP_STATUS = 8'h58;
    localparam [ADDR_W-1:0] R_RSP_R0     = 8'h60;
    localparam [ADDR_W-1:0] R_RSP_R1     = 8'h68;
    localparam [ADDR_W-1:0] R_HOST_ADDR  = 8'h70;
    localparam [ADDR_W-1:0] R_HOST_DATA  = 8'h78;
    localparam [ADDR_W-1:0] R_HOST_BE    = 8'h80;
    localparam [ADDR_W-1:0] R_HOST_CTRL  = 8'h88;
    localparam [ADDR_W-1:0] R_HOST_STAT  = 8'h90;
    localparam [ADDR_W-1:0] R_MAGIC      = 8'h98;
    localparam [ADDR_W-1:0] R_SERVED     = 8'hA0;

    // Version 1.0, then "TNCA" as ASCII in the low half.
    localparam [63:0] MAGIC = {16'h0001, 16'h0000, 32'h544E_4341};

    // MEM_WORDS is `WORDS`, widened once, and every range check below measures
    // against it.
    //
    // `WORDS` is a parameter and a parameter's width is the width of the value
    // it was given -- 32 bits for 16384. `WORDS[63:0]` therefore asks for bits
    // that do not exist, and a tool is free to answer with X. The comparison it
    // feeds then evaluates to X, `host_addr_ok` is unknown, and `if
    // (host_addr_ok)` takes the else branch: every payload access refused, at
    // every address, including zero.
    //
    // That is exactly what happened, and it is worth the paragraph because of
    // what it looked like from outside. The mailbox worked, the identity
    // register read back correctly, and the window refused everything -- which
    // reads as a boundary condition set too tight, not as an out-of-range
    // select. Widening it in one named place is what makes the next reader able
    // to see it.
    localparam [63:0] MEM_WORDS = WORDS;

    //==========================================================================
    // Mailbox registers
    //==========================================================================
    reg [63:0] req_seq, req_op, req_a0, req_a1, req_a2, req_a3, req_a4,
               req_a5, req_deadline;
    reg [63:0] rsp_seq, rsp_r0, rsp_r1;
    reg [3:0]  rsp_status;

    // The highest sequence acted on, and PING's answer. 64-bit to match the
    // field: a 32-bit counter would wrap in a long session and start refusing
    // sequences it had already served.
    reg [63:0] served;

    //==========================================================================
    // Payload window registers
    //==========================================================================
    reg [63:0] host_addr;
    reg [63:0] host_data;
    reg [7:0]  host_be;
    reg        host_status_valid;    // a fetched word is in host_data
    reg        host_status_refused;  // the last START was refused, not issued
    reg        host_busy;            // an access is in flight

    //==========================================================================
    // The mailbox sequencer
    //
    // MS_IDLE waits for the doorbell. A request whose sequence has already
    // been served is not served again and produces no completion, which is
    // the `seq > served` rule and the reason a duplicate ring is harmless.
    //
    // MS_ISSUE offers the command and holds it until the shell is idle enough
    // to take it; the shell's `cmd_ready` is low while a launch is live and
    // while a host access is in flight, so this state is also where the two
    // ports are kept from overlapping.
    //
    // MS_RUN waits for the completion and holds `done_ready` for exactly one
    // cycle. `done_valid` is a level, not a pulse -- the shell holds it until
    // it is taken -- so a completion cannot be missed by arriving early.
    //==========================================================================
    localparam [1:0] MS_IDLE  = 2'd0;
    localparam [1:0] MS_ISSUE = 2'd1;
    localparam [1:0] MS_RUN   = 2'd2;

    reg [1:0] ms;
    reg       doorbell_q;    // a doorbell write happened and is not yet taken

    //==========================================================================
    // The payload window sequencer
    //
    // One access at a time, and the address advances only when one completes.
    // An address outside memory is refused before anything is issued, so a
    // refusal cannot leave a request in flight that the shell will answer
    // later and this module will no longer be expecting.
    //==========================================================================
    localparam [1:0] PS_IDLE = 2'd0;
    localparam [1:0] PS_REQ  = 2'd1;
    localparam [1:0] PS_WAIT = 2'd2;

    reg [1:0]  ps;
    reg        ps_we_q;                 // write-enable latched at START
    reg [63:0] ps_addr_q;               // address latched at START
    reg [63:0] ps_wdata_q;              // write data latched at START
    reg [7:0]  ps_be_q;                 // byte enables latched at START

    // The request port is driven combinationally from the state and the
    // latched descriptor, NOT advanced by assignments inside PS_REQ.
    //
    // It was written the second way first, and it deadlocked on the first
    // access. `host_req_ready` is a combinational output of the shell
    // (rtl/tnca_shell.v:324) and does not depend on `host_req_valid`, so on the
    // first cycle in PS_REQ the ready was already high while the valid a
    // registered assignment was about to raise was still low. The sequencer
    // read that ready as its own handshake completing, dropped the valid it had
    // never raised, and moved to PS_WAIT to wait for a response to a request
    // that was never made.
    //
    // Driving valid combinationally means the cycle the state becomes PS_REQ is
    // the cycle the shell is asked, and the ready it answers with in that same
    // cycle is about that request. The descriptor fields are latched at START
    // rather than rewritten in PS_REQ for the same reason: a field that arrives
    // a cycle after its own valid is a field the shell samples from the
    // previous access.
    assign host_req_valid = (ps == PS_REQ);
    assign host_req_we    = ps_we_q;
    assign host_req_addr  = ps_addr_q;
    assign host_req_be    = ps_be_q;
    assign host_req_wdata = ps_wdata_q;

    // The guard. `host_addr` is 64-bit unsigned so there is no sign to be
    // caught out by; `(!host_addr[2:0])` is the alignment test and the
    // comparison against the end is done on the word index.
    wire host_addr_aligned = (host_addr[2:0] == 3'd0);
    wire host_addr_in_range = (host_addr[63:3] < MEM_WORDS);
    wire host_addr_ok = host_addr_aligned && host_addr_in_range;

    // The UPLOAD and DOWNLOAD guards, word for word from tnca_device.v.
    // `a0 % 8 == 0`, `a1 > 0`, and `a0/8 + a1 <= WORDS`, with a0 unsigned so
    // that a host address with the top bit set cannot arrive negative.
    wire [63:0] req_a0_words = req_a0 >> 3;
    wire req_guard_ok = (req_a0[2:0] == 3'd0) && (req_a1 != 64'd0) &&
                        ((req_a0_words + req_a1) <= MEM_WORDS);

    // The counter a PERF_READ names. Combinational from the descriptor, and
    // read by the shell combinationally: the doorbell is served in one
    // cycle of MS_IDLE, with the descriptor already written.
    assign perf_sel = req_a0[4:0];
    wire perf_index_ok = (req_a0 < PERF_NUM_COUNTERS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // The descriptor is NOT reset here. It is written by the host and
            // therefore lives with the other host-written registers, in the
            // block further down that has `reg_we` for a guard. Clearing it
            // here as well made two always blocks drive one reg -- which
            // Verilog calls a multiple driver no matter how the guards are
            // arranged, which simulation showed nothing of, and which yosys
            // reports 648 times over `req_*`, `host_addr` and `host_be`.
            rsp_seq <= 64'd0;  rsp_status <= RSP_OK;
            rsp_r0 <= 64'd0;   rsp_r1 <= 64'd0;
            served <= 64'd0;
            ms <= MS_IDLE;
            doorbell_q <= 1'b0;
            cmd_valid <= 1'b0;
            cmd_op <= 4'd0;
            cmd_pc <= 64'd0;   cmd_arg_base <= 64'd0;
            cmd_role_mask <= {NUM_RCTX{1'b0}};
            cmd_slice     <= 4'd0;
            cmd_pid_x <= 32'd0;  cmd_pid_y <= 32'd0;  cmd_pid_z <= 32'd0;
            cmd_npid_x <= 32'd1; cmd_npid_y <= 32'd1; cmd_npid_z <= 32'd1;
            cmd_deadline <= 64'd0;
            done_ready <= 1'b0;
        end else begin
            // A doorbell write is a one-cycle pulse into this flag. Holding
            // the doorbell register high would ring once per cycle for as
            // long as the host left it set.
            if (reg_we && reg_addr == R_DOORBELL)
                doorbell_q <= 1'b1;

            done_ready <= 1'b0;

            case (ms)
            MS_IDLE: begin
                // The flag is cleared whether the ring is served or dropped.
                //
                // Clearing it on the served path alone leaves it stuck when a
                // duplicate sequence arrives, and a stuck flag does not wait
                // for the next ring -- it fires the moment `REQ_SEQ` is
                // written, several writes before the host has finished
                // filling in the descriptor it belongs to. The request that
                // gets served then is made of the previous request's fields
                // and this one's sequence.
                if (doorbell_q)
                    doorbell_q <= 1'b0;
                // A doorbell for a sequence already acted on is dropped,
                // not answered. Answering it would be a second completion
                // for one request, which is the thing the rule exists to
                // prevent.
                if (doorbell_q && (req_seq > served)) begin
                    served <= req_seq;

                    // Everything that is not a launch completes here: it has
                    // an answer already and nothing for the core to do.
                    case (req_op[3:0])
                    OP_LAUNCH: begin
                        cmd_op        <= 4'd1;
                        cmd_pc        <= req_a0;
                        cmd_arg_base  <= req_a1;
                        cmd_role_mask <= req_a2[NUM_RCTX-1:0];
                        // {pair, slice, log2 slices}: bits 28, 31, 30:29.
                        cmd_slice     <= {req_a2[28], req_a2[31:29]};
                        cmd_pid_x     <= req_a3[15:0];
                        cmd_pid_y     <= req_a3[31:16];
                        cmd_pid_z     <= req_a4[15:0];
                        cmd_npid_x    <= req_a4[31:16];
                        cmd_npid_y    <= req_a5[15:0];
                        cmd_npid_z    <= req_a5[31:16];
                        cmd_deadline  <= req_deadline;
                        cmd_valid     <= 1'b1;
                        ms            <= MS_ISSUE;
                    end
                    OP_QUIESCE: begin
                        cmd_op        <= CMD_QUIESCE;
                        cmd_pc        <= 64'd0;
                        cmd_arg_base  <= 64'd0;
                        cmd_role_mask <= {NUM_RCTX{1'b0}};
                        cmd_slice     <= 4'd0;
            cmd_slice     <= 4'd0;
                        cmd_pid_x     <= 32'd0;
                        cmd_pid_y     <= 32'd0;
                        cmd_pid_z     <= 32'd0;
                        cmd_npid_x    <= 32'd1;
                        cmd_npid_y    <= 32'd1;
                        cmd_npid_z    <= 32'd1;
                        cmd_deadline  <= req_deadline;
                        cmd_valid     <= 1'b1;
                        ms            <= MS_ISSUE;
                    end
                    OP_UPLOAD, OP_DOWNLOAD: begin
                        // The payload travelled through the window; what is
                        // left is the guard and the record. A refused range
                        // touches nothing, which is R36-IO-01's requirement.
                        rsp_seq    <= req_seq;
                        rsp_status <= req_guard_ok ? RSP_OK : RSP_REFUSED;
                        rsp_r0     <= 64'd0;
                        rsp_r1     <= 64'd0;
                    end
                    OP_SHUTDOWN: begin
                        // Acknowledged and nothing else. See the header: a
                        // board has no process to end, and a device a register
                        // write can kill is a device a host can kill by
                        // accident.
                        rsp_seq    <= req_seq;
                        rsp_status <= RSP_OK;
                        rsp_r0     <= 64'd0;
                        rsp_r1     <= 64'd0;
                    end
                    OP_PING: begin
                        rsp_seq    <= req_seq;
                        rsp_status <= RSP_OK;
                        // The ping is a request the device acted on, so it
                        // is counted in the number it reports. That is what
                        // tb/device/tnca_device.v:341 does -- a blocking
                        // `served = seq`, read again at :388 -- and a ping
                        // that answered with its own predecessor would be
                        // off by one against the simulation binding.
                        rsp_r0     <= req_seq;
                        rsp_r1     <= 64'd0;
                    end
                    OP_PERF_READ: begin
                        rsp_seq    <= req_seq;
                        rsp_status <= perf_index_ok ? RSP_OK : RSP_REFUSED;
                        rsp_r0     <= perf_index_ok ? {32'd0, perf_rdata[31:0]}  : 64'd0;
                        rsp_r1     <= perf_index_ok ? {32'd0, perf_rdata[63:32]} : 64'd0;
                    end
                    default: begin
                        // An opcode the device does not have is refused
                        // rather than ignored, so a host can tell a typo
                        // from a device that stopped answering.
                        rsp_seq    <= req_seq;
                        rsp_status <= RSP_REFUSED;
                        rsp_r0     <= 64'd0;
                        rsp_r1     <= 64'd0;
                    end
                    endcase
                end
            end

            MS_ISSUE: begin
                if (cmd_ready) begin
                    cmd_valid <= 1'b0;
                    ms        <= MS_RUN;
                end
            end

            MS_RUN: begin
                if (done_valid) begin
                    done_ready <= 1'b1;
                    rsp_seq    <= req_seq;
                    rsp_status <= done_status;
                    // The completion record is what the shell decided, not a
                    // second opinion. FAULT carries the code and the pc where
                    // tnca_device.v put them; everything else carries cycles
                    // and retired instructions in the same two fields.
                    if (done_status == RSP_FAULT) begin
                        rsp_r0 <= {56'd0, done_fault_code};
                        rsp_r1 <= done_fault_pc;
                    end else if (done_status == RSP_TIMEOUT) begin
                        rsp_r0 <= done_cycles;
                        rsp_r1 <= 64'd0;
                    end else begin
                        rsp_r0 <= done_cycles;
                        rsp_r1 <= done_retired;
                    end
                    ms <= MS_IDLE;
                end
            end

            default: ms <= MS_IDLE;
            endcase
        end
    end

    //==========================================================================
    // The payload window
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // `host_addr` and `host_be` are host-written, so their reset is
            // with the rest of the host-written registers below. `host_data`
            // is written by this window alone and is reset here.
            host_data <= 64'd0;
            host_status_valid   <= 1'b0;
            host_status_refused <= 1'b0;
            host_busy           <= 1'b0;
            ps        <= PS_IDLE;
            ps_we_q   <= 1'b0;
            ps_addr_q <= 64'd0;
            ps_wdata_q <= 64'd0;
            ps_be_q    <= 8'hFF;
        end else begin
            if (reg_we && reg_addr == R_HOST_DATA)
                host_data <= reg_wdata;

            case (ps)
            PS_IDLE: begin
                host_busy <= 1'b0;
                if (reg_we && reg_addr == R_HOST_CTRL && reg_wdata[1]) begin
                    // VALID is cleared at the start of an access, not when it
                    // completes: a host that reads HOST_DATA between the two
                    // must not be handed the previous word as if it were this
                    // one.
                    host_status_valid <= 1'b0;
                    if (host_addr_ok) begin
                        host_status_refused <= 1'b0;
                        ps_we_q    <= reg_wdata[0];
                        ps_addr_q  <= host_addr;
                        // Latched here so that every field of the request is
                        // already in place on the cycle `host_req_valid` goes
                        // high, which is the next one. A field taken straight
                        // from `host_data` would be a field the host is still
                        // free to overwrite while the access is in flight.
                        ps_wdata_q <= host_data;
                        ps_be_q    <= host_be;
                        ps         <= PS_REQ;
                        host_busy  <= 1'b1;
                    end else begin
                        host_status_refused <= 1'b1;
                    end
                end
            end

            PS_REQ: begin
                // Nothing to drive: the port is combinational from this state
                // and the descriptor latched at START. The only thing left is
                // to notice the handshake and drop out of the state, which
                // lowers `host_req_valid` in the same cycle.
                if (host_req_ready)
                    ps <= PS_WAIT;
            end

            PS_WAIT: begin
                // The shell answers both a read and a write: its host path
                // waits on the same memory response either way
                // (rtl/tnca_shell.v:365), so one state serves both and a
                // write cannot leave the window busy forever.
                // No address is advanced here. The window used to step
                // `host_addr` by eight on completion, and a host that writes
                // R_HOST_ADDR before every access -- which is what
                // tb/fpga/tb_tnca_fpga.v does and what
                // python/tnca/backends/u50.py does -- never read it. It was
                // a second driver of a host-written register for a feature
                // nobody used.
                if (host_rsp_valid) begin
                    host_data <= host_rsp_rdata;
                    host_status_valid <= 1'b1;
                    host_busy <= 1'b0;
                    ps <= PS_IDLE;
                end
            end

            default: ps <= PS_IDLE;
            endcase
        end
    end

    //==========================================================================
    // The read mux
    //
    // Combinational, and complete: every address decodes to something, so a
    // read of a hole returns zero rather than a latch. HOST_STATUS is the one
    // exception that is assembled rather than stored, because BUSY belongs to
    // the payload sequencer and not to a register.
    //==========================================================================
    always @* begin
        case (reg_addr)
        R_REQ_SEQ:    reg_rdata = req_seq;
        R_REQ_OP:     reg_rdata = req_op;
        R_REQ_A0:     reg_rdata = req_a0;
        R_REQ_A1:     reg_rdata = req_a1;
        R_REQ_A2:     reg_rdata = req_a2;
        R_REQ_A3:     reg_rdata = req_a3;
        R_REQ_A4:     reg_rdata = req_a4;
        R_REQ_A5:     reg_rdata = req_a5;
        R_REQ_DEAD:   reg_rdata = req_deadline;
        R_RSP_SEQ:    reg_rdata = rsp_seq;
        R_RSP_STATUS: reg_rdata = {60'd0, rsp_status};
        R_RSP_R0:     reg_rdata = rsp_r0;
        R_RSP_R1:     reg_rdata = rsp_r1;
        R_HOST_ADDR:  reg_rdata = host_addr;
        R_HOST_DATA:  reg_rdata = host_data;
        R_HOST_BE:    reg_rdata = {56'd0, host_be};
        R_HOST_STAT:  reg_rdata = {61'd0, host_status_refused,
                                   host_status_valid, host_busy};
        R_MAGIC:      reg_rdata = MAGIC;
        R_SERVED:     reg_rdata = served;
        // DOORBELL and HOST_CTRL are write-only. Reading one returns zero
        // rather than the last value written, so a host cannot mistake a
        // write-only register for a status it can poll.
        default:      reg_rdata = 64'd0;
        endcase
    end

    //==========================================================================
    // Writes to the request registers. Kept out of the sequencer's block so
    // that a descriptor can be filled in any order and, incidentally, so that
    // the synthesizer sees one writer per register rather than two.
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // The whole host-written set is reset here, and only here. The
            // comment above is the reason: this block exists so that one
            // register has one writer, and a reset in the sequencer's block
            // gave every one of these a second. It read as harmless because
            // the two branches can never both be taken in the same cycle --
            // but `always` blocks are not branches of each other, and the
            // rule Verilog applies is per register, not per cycle.
            req_seq <= 64'd0;  req_op <= 64'd0;
            req_a0 <= 64'd0;   req_a1 <= 64'd0;   req_a2 <= 64'd0;
            req_a3 <= 64'd0;   req_a4 <= 64'd0;   req_a5 <= 64'd0;
            req_deadline <= 64'd0;
            host_addr <= 64'd0;
            host_be   <= 8'hFF;
        end else if (reg_we) begin
            case (reg_addr)
            R_REQ_SEQ:  req_seq  <= reg_wdata;
            R_REQ_OP:   req_op   <= reg_wdata;
            R_REQ_A0:   req_a0   <= reg_wdata;
            R_REQ_A1:   req_a1   <= reg_wdata;
            R_REQ_A2:   req_a2   <= reg_wdata;
            R_REQ_A3:   req_a3   <= reg_wdata;
            R_REQ_A4:   req_a4   <= reg_wdata;
            R_REQ_A5:   req_a5   <= reg_wdata;
            R_REQ_DEAD: req_deadline <= reg_wdata;
            R_HOST_ADDR: host_addr <= reg_wdata;
            R_HOST_BE:   host_be   <= reg_wdata[7:0];
            default: ;
            endcase
        end
    end

    // Three inputs are taken and not read: `done_fault_rctx` and
    // `done_fault_detail` carry a fault's context index and 24-bit detail,
    // and the mailbox has four response fields of which transport.py reads
    // two -- r0 and r1 -- so there is nowhere for them to go. `reg_re` is a
    // read strobe this block does not need, because the read mux above is
    // combinational and `reg_rdata` is valid without it. They are kept on the
    // port rather than dropped: the shell drives them, and a wrapper that
    // silently stopped listening to the completion record would be the wrong
    // place to save two wires.

endmodule

`default_nettype wire
