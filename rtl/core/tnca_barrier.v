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
    input  wire                     opdone_en,
    input  wire [BAR_W-1:0]         opdone_bar,

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
    assign test_done  = (done_epoch[tst_bar] >= tst_epoch) && armed[tst_bar];
    assign test_error = err[tst_bar];

    assign arm_token = {{(48-EPOCH_W){1'b0}}, (cur_epoch[arm_bar] + 1'b1),
                        {(16-BAR_W){1'b0}}, arm_bar};

    //--------------------------------------------------------------------------
    // A barrier reaches zero on both counts in this cycle
    //--------------------------------------------------------------------------
    reg [BAR_N-1:0] completes;
    reg [CNT_W-1:0] next_arr, next_ops;

    always @(*) begin
        completes = {BAR_N{1'b0}};
        for (i = 0; i < BAR_N; i = i + 1) begin
            next_arr = n_arrive[i] - ((arrive_en && (arrive_bar == i)) ? 1 : 0);
            next_ops = n_ops[i]    - ((opdone_en && (opdone_bar == i)) ? 1 : 0);
            // Only an armed generation that has not already completed.
            if (armed[i] && (done_epoch[i] != cur_epoch[i]) &&
                (next_arr == 0) && (next_ops == 0) &&
                !(arm_en && (arm_bar == i)))
                completes[i] = 1'b1;
        end
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
            fault     <= 1'b0;
            fault_bar <= {BAR_W{1'b0}};
        end else begin
            wake  <= {NUM_RCTX{1'b0}};
            fault <= 1'b0;

            //------------------------------------------------------------------
            // init
            //------------------------------------------------------------------
            if (init_en) begin
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

            if (opdone_en && !(arm_en && arm_bar == opdone_bar)) begin
                if (n_ops[opdone_bar] == 0) begin
                    err[opdone_bar] <= 1'b1;
                    fault     <= 1'b1;
                    fault_bar <= opdone_bar;
                end else begin
                    n_ops[opdone_bar] <= n_ops[opdone_bar] - 1'b1;
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
                        (inval_en && (inval_bar == slp_bar[i])) ||
                        completes[slp_bar[i]] ||
                        (done_epoch[slp_bar[i]] >= slp_epoch[i])) begin
                        wake[i]  <= 1'b1;
                        slp_v[i] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
