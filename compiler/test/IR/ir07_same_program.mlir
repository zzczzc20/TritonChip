// RUN: tnca-opt --tnca-verify-async --verify-diagnostics %s
//
// IR-07 (R1-07): one program, two spellings.
//
// The Python IR and this module express the same two kernels -- refill after
// the wait, and refill before it -- and the contract requires the two paths
// to reach the same verdict on the supported subset. A difference is a bug in
// one of them, to be fixed rather than papered over by only running the one
// that agrees.

// The legal one: wait, then refill. No diagnostic.
func.func @refill_after_the_wait(%acc: !tnca.acc, %d: !tnca.desc, %c: i32) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  %e2, %l2 = "tnca.dma.load"(%d, %a, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e2) : (!tnca.event) -> ()
  return
}

// The illegal one: refill first.
func.func @refill_before_the_wait(%acc: !tnca.acc, %d: !tnca.desc, %c: i32) {
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
