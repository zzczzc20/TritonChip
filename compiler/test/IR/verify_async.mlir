// RUN: tnca-opt --tnca-verify-async --verify-diagnostics %s
//
// The asynchronous-resource verifier of compiler_overview.md sec 5.4. Each
// case is a liveness mistake the memory-effects interface cannot express,
// because it says an op reads memory without saying which registers or until
// when.

// A well-formed transfer: filled, awaited, then used.
func.func @ok_load_then_use(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.use_complete"(%w) : (!tnca.window<8>) -> ()
  return
}

// ISA sec 9C.3: the sources of an incomplete m.mma.async may not be
// overwritten. The matrix unit reads the whole fragment for many cycles after
// this instruction retires.
func.func @refill_while_mma_reads(%acc: !tnca.acc, %d: !tnca.desc, %c: i32) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  // expected-error @+1 {{writes a window that an outstanding command is still reading}}
  %e2, %l2 = "tnca.dma.load"(%d, %a, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}

// Reading a window whose fill is still in flight. sec 8.1: publish means the
// fill completed, which is why it is not a mechanical rewrite of mb.arm.
func.func @read_before_fill_completes(%acc: !tnca.acc, %d: !tnca.desc, %c: i32) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %a, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  // expected-error @+1 {{reads a window an outstanding command is still writing}}
  %e2, %l2 = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}

// sec 10.2: an asynchronous lifetime ends at a completion event. Without one
// the registers are owned forever and the allocator can never reuse them.
func.func @never_awaited(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  // expected-error @+1 {{completion event is never awaited}}
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  return
}

// sec 13.4's third case: an asynchronous store reads its source window late,
// so the window is not free when the instruction retires.
func.func @refill_while_store_reads(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.store"(%w, %d, %c)
      : (!tnca.window<8>, !tnca.desc, i32) -> (!tnca.event, !tnca.lease)
  // expected-error @+1 {{writes a window that an outstanding command is still reading}}
  %e2, %l2 = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}
