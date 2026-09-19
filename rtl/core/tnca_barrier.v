//==============================================================================
// tnca_barrier -- per-core mbarrier table (ISA sec 9A, arch overview sec 13)
//
// Barriers are shared by every RCTX on the core, which is what makes them the
// synchronisation between roles (sec 14.3).
//
// WHY ARRIVALS AND OPERATIONS ARE COUNTED SEPARATELY
//
// Sec 13.1 insists that three completions are different things: data has
// arrived in registers, a computation has finished, and an operation has
// stopped reading its sources. A single "done" bit cannot express all three,
// and a double-buffered pipeline that implements only the first is the most
// common way to get this wrong (sec 13.5) -- a transfer completing is not the
// same as the consumer having finished reading it.
//
// Arrivals are what software signals with mb.arrive; operations are what
// asynchronous commands signal when they retire. Splitting them lets one
// generation mean "the producer has armed the slot AND both transfers landed",
// which is a statement neither count could make alone.
//
// COUNTING UNIT: ONE COMPLETE ASYNCHRONOUS COMMAND (sec 13.3). Not a cache
// line, not a byte count. Two t.ld.async are operations=2 however many line
// transactions they expand into, and a fully predicated-off command still
// completes once, so the count in mb.arm stays a static, checkable property.
//
// SAFETY PROPERTIES, not performance ones (sec 13.4):
//   - a generation completes exactly once
//   - neither count may underflow; an extra arrival or completion is a fault
//   - an error wakes every waiter, so a faulting command cannot become an
//     unbounded wait
//   - mb.wait sleeps the RCTX; it does not spin
//==============================================================================
`include "tnca_defs.vh"

module tnca_barrier #(
    parameter NUM_RCTX = 2,
    parameter RCTX_W   = 1,
    parameter BAR_N    = 32,
    parameter BAR_W    = 5,
    parameter EPOCH_W  = 16,
    parameter CNT_W    = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     flush,          // new kernel launch

    // ---- init ---------------------------------------------------------------
    input  wire                     init_en,
    input  wire [BAR_W-1:0]         init_bar,

    // ---- arm ----------------------------------------------------------------
    input  wire                     arm_en,
    input  wire [BAR_W-1:0]         arm_bar,
    input  wire [CNT_W-1:0]         arm_arrivals,
    input  wire [CNT_W-1:0]         arm_operations,
    output wire [63:0]              arm_token,      // valid in the arm cycle

    // ---- arrive (software) --------------------------------------------------
    input  wire                     arrive_en,
    input  wire [BAR_W-1:0]         arrive_bar,

    // ---- operation completion (asynchronous commands) -----------------------
    //
    // One port PER ENGINE, not one shared port. The matrix unit and the
    // transfer engine run independently and can finish in the same cycle;
    // funnelling them through a single `opdone_en` counted one of the two and
    // dropped the other, so a barrier waited for ever on an operation that
    // had in fact completed. It took a slow memory model to shift the timing
    // enough for the two to coincide.
    input  wire                     mtx_done_en,
    input  wire [BAR_W-1:0]         mtx_done_bar,

    input  wire                     txf_done_en,
    // Set when the completing command FAILED. ISA sec 9A.2: the barrier's
    // error bit is set by mb.inval or by a faulting command.
    input  wire                     txf_done_err,
    input  wire [BAR_W-1:0]         txf_done_bar,

    // ---- invalidate ---------------------------------------------------------
    input  wire                     inval_en,
    input  wire [BAR_W-1:0]         inval_bar,

    // ---- test (combinational query) -----------------------------------------
    input  wire [63:0]              test_token,
    output wire                     test_done,
    output wire                     test_error,

    // ---- sleeping RCTXs -----------------------------------------------------
    input  wire                     sleep_en,
    input  wire [RCTX_W-1:0]        sleep_rctx,
    input  wire [63:0]              sleep_token,
    output reg  [NUM_RCTX-1:0]      wake,           // pulse, one cycle
    // Set alongside `wake` when the waiter was released by an ERROR rather
    // than by the generation completing. ISA sec 10 makes a waiter woken this
    // way raise BARRIER_FAULT: resuming it silently sends it on to wait for
    // the next generation of a pipeline whose producer has already died,
    // which turns a located fault into a hang.
    output reg  [NUM_RCTX-1:0]      wake_err,

    // ---- faults -------------------------------------------------------------
    output reg                      fault,
    output reg  [BAR_W-1:0]         fault_bar
);

    integer i;

    reg [EPOCH_W-1:0] cur_epoch  [0:BAR_N-1];
    reg [EPOCH_W-1:0] done_epoch [0:BAR_N-1];
    reg [CNT_W-1:0]   n_arrive   [0:BAR_N-1];
    reg [CNT_W-1:0]   n_ops      [0:BAR_N-1];
    reg [BAR_N-1:0]   err;
    reg [BAR_N-1:0]   armed;

    // one sleeper record per RCTX
    reg [NUM_RCTX-1:0]  slp_v;
    reg [BAR_W-1:0]     slp_bar   [0:NUM_RCTX-1];
    reg [EPOCH_W-1:0]   slp_epoch [0:NUM_RCTX-1];

    //--------------------------------------------------------------------------
    // Token: { epoch, barrier_id }
    //--------------------------------------------------------------------------
    wire [BAR_W-1:0]   tst_bar   = test_token[BAR_W-1:0];
    wire [EPOCH_W-1:0] tst_epoch = test_token[16 +: EPOCH_W];

    // A generation is complete when the barrier's completed epoch has caught up
    // with it. Comparing ">=" rather than "==" means a waiter that arrives late
    // still sees its generation as finished instead of sleeping forever.
    //
    // No `armed` term. This predicate decides whether to SLEEP, and the
    // sleepers loop below decides whether to WAKE; the two have to agree or a
    // waiter oscillates. It required `armed` and the wake did not, so a wait
    // on generation 0 of a barrier nothing had armed slept, woke immediately,
    // re-executed, and slept again -- a livelock burning every issue slot on
    // squashes rather than a clean hang.
    //
    // The term was also wrong on its own terms: `armed` is only ever clear
    // before the first mb.arm, and generation 0 is the quiescent state every
    // barrier starts in (ISA sec 9A.7), so it is complete by definition --
    // nothing was ever pending. A wait on a generation that genuinely has not
    // happened still sleeps, because done_epoch has not reached it.
    assign test_done  = (done_epoch[tst_bar] >= tst_epoch);
    assign test_error = err[tst_bar];

    assign arm_token = {{(48-EPOCH_W){1'b0}}, (cur_epoch[arm_bar] + 1'b1),
                        {(16-BAR_W){1'b0}}, arm_bar};

    // How many operation completions land on each barrier this cycle. Two
    // engines report independently and nothing stops them naming the same
    // barrier, so this is 0, 1 or 2.
    //
    // Written as a macro rather than a function on purpose. A function called
    // from an always @(*) does not contribute the signals it reads internally
    // to the block's sensitivity list, so the block goes stale exactly when a
    // completion arrives -- which is the one moment it must not. The same
    // trap cost a pipeline deadlock earlier in this design.
    `define OP_DEC(b) \
        ((((mtx_done_en && (mtx_done_bar == (b))) ? 2'd1 : 2'd0)) + \
         (((txf_done_en && !txf_done_err && (txf_done_bar == (b))) \
           ? 2'd1 : 2'd0)))

    //--------------------------------------------------------------------------
    // A barrier reaches zero on both counts in this cycle
    //--------------------------------------------------------------------------
    integer od_i;
    reg [BAR_N-1:0] completes;
    reg [CNT_W-1:0] next_arr, next_ops;

    always @(*) begin
        completes = {BAR_N{1'b0}};
        for (i = 0; i < BAR_N; i = i + 1) begin
            next_arr = n_arrive[i] - ((arrive_en && (arrive_bar == i)) ? 1 : 0);
            next_ops = n_ops[i] - `OP_DEC(i[BAR_W-1:0]);
            // Only an armed generation that has not already completed.
            if (armed[i] && (done_epoch[i] != cur_epoch[i]) &&
                (next_arr == 0) && (next_ops == 0) &&
                !(arm_en && (arm_bar == i)))
                completes[i] = 1'b1;
        end
    end

    reg slp_on_init;
    integer si;
    always @(*) begin
        slp_on_init = 1'b0;
        for (si = 0; si < NUM_RCTX; si = si + 1)
            if (slp_v[si] && (slp_bar[si] == init_bar)) slp_on_init = 1'b1;
    end

    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n || flush) begin
            for (i = 0; i < BAR_N; i = i + 1) begin
                cur_epoch[i]  <= {EPOCH_W{1'b0}};
                done_epoch[i] <= {EPOCH_W{1'b0}};
                n_arrive[i]   <= {CNT_W{1'b0}};
                n_ops[i]      <= {CNT_W{1'b0}};
            end
            err       <= {BAR_N{1'b0}};
            armed     <= {BAR_N{1'b0}};
            slp_v     <= {NUM_RCTX{1'b0}};
            wake      <= {NUM_RCTX{1'b0}};
            wake_err  <= {NUM_RCTX{1'b0}};
            fault     <= 1'b0;
            fault_bar <= {BAR_W{1'b0}};
        end else begin
            wake     <= {NUM_RCTX{1'b0}};
            wake_err <= {NUM_RCTX{1'b0}};
            fault <= 1'b0;

            //------------------------------------------------------------------
            // init
            //------------------------------------------------------------------
            // Re-initialising a barrier somebody is asleep on would strand
            // that waiter forever: init resets completed_epoch to zero, so the
            // generation it is waiting for can never appear to complete.
            // doc/compiler_overview.md sec 13.3 asks for a deterministic
            // start-up protocol; making the race a detected fault is the
            // honest half of that, since no ordering the hardware could invent
            // would be the one the program meant.
            if (init_en && slp_on_init) begin
                fault     <= 1'b1;
                fault_bar <= init_bar;
                err[init_bar] <= 1'b1;
            end else if (init_en) begin
                cur_epoch[init_bar]  <= {EPOCH_W{1'b0}};
                done_epoch[init_bar] <= {EPOCH_W{1'b0}};
                n_arrive[init_bar]   <= {CNT_W{1'b0}};
                n_ops[init_bar]      <= {CNT_W{1'b0}};
                err[init_bar]        <= 1'b0;
                armed[init_bar]      <= 1'b0;
            end

            //------------------------------------------------------------------
            // arm: both counts fixed for this generation in one instruction
            //------------------------------------------------------------------
            if (arm_en) begin
                cur_epoch[arm_bar] <= cur_epoch[arm_bar] + 1'b1;
                n_arrive[arm_bar]  <= arm_arrivals;
                n_ops[arm_bar]     <= arm_operations;
                armed[arm_bar]     <= 1'b1;
            end

            //------------------------------------------------------------------
            // arrivals and operation completions
            //
            // Underflow is a fault, not a wrap: an extra arrival means the
            // program's own count was wrong, and silently wrapping would turn
            // that into a hang thousands of cycles later.
            //------------------------------------------------------------------
            if (arrive_en && !(arm_en && arm_bar == arrive_bar)) begin
                if (n_arrive[arrive_bar] == 0) begin
                    err[arrive_bar] <= 1'b1;
                    fault     <= 1'b1;
                    fault_bar <= arrive_bar;
                end else begin
                    n_arrive[arrive_bar] <= n_arrive[arrive_bar] - 1'b1;
                end
            end

            // A completion that reports failure sets the barrier's error and
            // releases its waiters, exactly as mb.inval does (ISA sec 9A.2).
            // Counting it as an ordinary completion would let the consumer
            // proceed on data that was never written -- a wrong answer, where
            // the waiter is at least a located one.
            // A failed completion sets the error and releases the waiters
            // instead of counting (ISA sec 9A.2).
            if (txf_done_en && txf_done_err) begin
                err[txf_done_bar]        <= 1'b1;
                done_epoch[txf_done_bar] <= cur_epoch[txf_done_bar];
            end

            // Both engines may report in the same cycle, and both may name
            // the same barrier, so the decrement is a count rather than a
            // flag. An arm in the same cycle re-sets the counts, so a
            // completion for the phase being replaced is not applied to it.
            // Private loop variable: `i` is also the index of the
            // combinational `completes` block, and sharing one between an
            // always @(*) and an always @(posedge) makes the two retrigger
            // each other.
            for (od_i = 0; od_i < BAR_N; od_i = od_i + 1) begin
                if ((`OP_DEC(od_i[BAR_W-1:0]) != 2'd0) &&
                    !(arm_en && (arm_bar == od_i[BAR_W-1:0]))) begin
                    if (n_ops[od_i] < `OP_DEC(od_i[BAR_W-1:0])) begin
                        err[od_i]  <= 1'b1;
                        fault      <= 1'b1;
                        fault_bar  <= od_i[BAR_W-1:0];
                    end else begin
                        n_ops[od_i] <= n_ops[od_i] - `OP_DEC(od_i[BAR_W-1:0]);
                    end
                end
            end

            //------------------------------------------------------------------
            // completion: exactly once per generation
            //------------------------------------------------------------------
            for (i = 0; i < BAR_N; i = i + 1) begin
                if (completes[i]) begin
                    done_epoch[i] <= cur_epoch[i];
                end
            end

            //------------------------------------------------------------------
            // invalidate wakes every waiter with an error, so a faulting
            // command can never become an unbounded wait (sec 13.4)
            //------------------------------------------------------------------
            if (inval_en) begin
                err[inval_bar]        <= 1'b1;
                done_epoch[inval_bar] <= cur_epoch[inval_bar];
            end

            //------------------------------------------------------------------
            // sleepers
            //------------------------------------------------------------------
            if (sleep_en) begin
                slp_v[sleep_rctx]     <= 1'b1;
                slp_bar[sleep_rctx]   <= sleep_token[BAR_W-1:0];
                slp_epoch[sleep_rctx] <= sleep_token[16 +: EPOCH_W];
            end

            for (i = 0; i < NUM_RCTX; i = i + 1) begin
                if (slp_v[i]) begin
                    if (err[slp_bar[i]] ||
                        (inval_en && (inval_bar == slp_bar[i]))) begin
                        wake[i]     <= 1'b1;
                        wake_err[i] <= 1'b1;
                        slp_v[i]    <= 1'b0;
                    end else if (completes[slp_bar[i]] ||
                                 (done_epoch[slp_bar[i]] >= slp_epoch[i]))
                    begin
                        wake[i]  <= 1'b1;
                        slp_v[i] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule

`undef OP_DEC
