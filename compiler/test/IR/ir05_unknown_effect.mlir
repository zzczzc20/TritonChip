// RUN: tnca-opt --tnca-verify-async --verify-diagnostics --allow-unregistered-dialect %s
//
// IR-05 (R1-07): an unknown resource effect is refused, never assumed to be
// none. "在 lowering 前或当步明确拒绝，不生成可执行成功工件."
//
// The dangerous case is not a malformed program -- the parser catches those.
// It is a well-formed one whose effect on a window or accumulator nobody has
// described: a foreign operation handed a window, or a TNCA operation added
// later without teaching the verifier what it does. Both used to fall through
// the `touches.empty()` branch and be treated as inert.

// ---- a foreign operation holding a window --------------------------------
// `scf.execute_region` is well-formed MLIR and says nothing about what it
// does to a window. Assuming "nothing" is the assumption that costs an
// answer.
func.func @foreign_op_takes_a_window(%d: !tnca.desc, %c: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  %e, %l = "tnca.dma.load"(%d, %w, %c)
      : (!tnca.desc, !tnca.window<8>, i32) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  // expected-error @+1 {{unsupported: operation outside the TNCA dialect takes a window or accumulator operand}}
  %r = "test.opaque_use"(%w) : (!tnca.window<8>) -> i32
  return
}

// ---- a foreign operation holding an accumulator --------------------------
func.func @foreign_op_takes_an_accumulator(%acc: !tnca.acc) {
  // expected-error @+1 {{unsupported: operation outside the TNCA dialect takes a window or accumulator operand}}
  %r = "test.opaque_use"(%acc) : (!tnca.acc) -> i32
  return
}
