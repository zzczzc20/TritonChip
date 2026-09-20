// RUN: tnca-opt --tnca-verify-async --verify-diagnostics %s
//
// IR-01 (R1-07): the supported subset, stated positively. Every program here
// is a single block of DMA / MMA / store / await, and every one of them is
// either legal and must pass, or is one of the OWN-series mistakes and must
// be refused with a diagnostic naming the rule.

// ---- legal: fill, await, use --------------------------------------------
func.func @load_await_use(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.use_complete"(%w) : (!tnca.window<8>) -> ()
  return
}

// ---- legal: a K loop accumulating into one accumulator -------------------
// Two commands writing the same accumulator in sequence is the ordinary
// K loop. What is forbidden is READING it meanwhile, not accumulating into
// it, and a verifier that conflated the two would reject every GEMM.
func.func @accumulate_then_read(%acc: !tnca.acc, %r: i32) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e1, %l1 = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e1) : (!tnca.event) -> ()
  %e2, %l2 = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e2) : (!tnca.event) -> ()
  %v = "tnca.acc.read"(%acc, %r) : (!tnca.acc, i32) -> i32
  return
}

// ---- legal: an asynchronous store, read concurrently ---------------------
// A store READS its source window, so another reader is not a conflict. Only
// a writer is.
func.func @store_then_read_source(%d: !tnca.desc, %acc: !tnca.acc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.store"(%w, %d, %c)
      : (!tnca.window<8>, !tnca.desc, i32) -> (!tnca.event, !tnca.lease)
  %e2, %l2 = "tnca.mma.async"(%acc, %w, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}

// ---- ISA sec 9C.3: the accumulator may not be read while it is written ---
func.func @acc_read_before_await(%acc: !tnca.acc, %r: i32) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  // expected-error @+1 {{reads a accumulator that command tnca.mma.async has not finished writing}}
  %v = "tnca.acc.read"(%acc, %r) : (!tnca.acc, i32) -> i32
  "tnca.await"(%e) : (!tnca.event) -> ()
  return
}

// ---- two commands filling the same window at once ------------------------
func.func @two_fills_one_window(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  // expected-error @+1 {{writes a window another outstanding command is already writing}}
  %e2, %l2 = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}
