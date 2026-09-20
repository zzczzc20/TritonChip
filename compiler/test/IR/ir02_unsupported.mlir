// RUN: tnca-opt --tnca-verify-async --verify-diagnostics %s
//
// IR-02 (R1-07): the frozen support boundary.
//
// This verifier reasons within one block. That is a real limit, and the
// contract requires it to be a STATED one -- "未实现的结构明确报 unsupported；
// 不得不检查直接接受". Two of these used to pass in silence: an event forwarded
// through a block argument hit a null defining op and was skipped, and a
// nested region was walked as an independent block, so the commands inside it
// and the awaits outside it were checked against each other not at all.
//
// Freezing a boundary is not deleting a capability. When general CFG support
// arrives these diagnostics come out.

// ---- multi-block: a lease that crosses a branch --------------------------
// expected-error @+1 {{unsupported: this region has 3 blocks}}
func.func @lease_across_a_branch(%d: !tnca.desc, %c: i32, %p: i1) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  cf.cond_br %p, ^await, ^skip
^await:
  "tnca.await"(%e) : (!tnca.event) -> ()
  return
^skip:
  return
}

// ---- an event arriving as a block argument -------------------------------
// The producing command is somewhere else entirely, so nothing here can say
// what it still owns.
// expected-error @+1 {{unsupported: a completion event or lease arrives as a block argument}}
func.func @event_as_an_argument(%e: !tnca.event) {
  // ...and the await itself, which is the site that used to be skipped.
  // expected-error @+1 {{unsupported: awaits an event that is not produced in this block}}
  "tnca.await"(%e) : (!tnca.event) -> ()
  return
}

// ---- a nested region -----------------------------------------------------
func.func @command_inside_a_loop(%d: !tnca.desc, %c: i32, %lo: index,
                                 %hi: index, %st: index) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  // expected-error @+1 {{unsupported: a nested region}}
  scf.for %i = %lo to %hi step %st {
    %e, %l = "tnca.dma.load"(%d, %w, %c)
        : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
    "tnca.await"(%e) : (!tnca.event) -> ()
  }
  return
}
