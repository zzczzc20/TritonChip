// RUN: tnca-opt %s | tnca-opt | FileCheck %s
//
// compiler_overview.md sec 16.2, "IR 单元测试". The first thing to check is
// that the dialect of sec 5.3 parses and prints what it claims to model:
// resources and views, asynchronous operations, roles and communication.

// CHECK-LABEL: func.func @resources
func.func @resources() {
  // CHECK: tnca.rf.reserve 8 : <8>
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  // CHECK: "tnca.acc.reserve"() : () -> !tnca.acc
  %a = "tnca.acc.reserve"() : () -> !tnca.acc
  return
}

// A descriptor is a versioned snapshot, not a pointer (ISA sec 9D.1).
// CHECK-LABEL: func.func @descriptor
func.func @descriptor(%addr: i64) {
  // CHECK: "tnca.desc.snapshot"(%{{.*}}) : (i64) -> !tnca.desc
  %d = "tnca.desc.snapshot"(%addr) : (i64) -> !tnca.desc
  return
}

// The event and the lease are SSA values, which is the point of sec 5.4: a
// completion that exists only as an attribute is invisible to every general
// MLIR transform.
// CHECK-LABEL: func.func @async_load
func.func @async_load(%d: !tnca.desc, %c0: i32, %c1: i32) {
  %w = tnca.rf.reserve 8 : !tnca.window<8>
  // CHECK: "tnca.dma.load"({{.*}}) : (!tnca.desc, !tnca.window<8>, i32, i32) -> (!tnca.event, !tnca.lease)
  %e, %l = "tnca.dma.load"(%d, %w, %c0, %c1)
      : (!tnca.desc, !tnca.window<8>, i32, i32) -> (!tnca.event, !tnca.lease)
  // CHECK: "tnca.await"(%{{.*}}) : (!tnca.event) -> ()
  "tnca.await"(%e) : (!tnca.event) -> ()
  return
}

// CHECK-LABEL: func.func @mma
func.func @mma(%acc: !tnca.acc) {
  %a = tnca.rf.reserve 8 : !tnca.window<8>
  %b = tnca.rf.reserve 8 : !tnca.window<8>
  // CHECK: "tnca.mma.async"({{.*}}) -> (!tnca.event, !tnca.lease)
  %e, %l = "tnca.mma.async"(%acc, %a, %b)
      : (!tnca.acc, !tnca.window<8>, !tnca.window<8>) -> (!tnca.event, !tnca.lease)
  "tnca.await"(%e) : (!tnca.event) -> ()
  return
}

// An RF channel, not a shared-memory Aref (sec 7.4): TNCA has no scratchpad
// for an Aref to live in.
// CHECK-LABEL: func.func @channel
func.func @channel() {
  // CHECK: "tnca.channel.create"() <{depth = 2 : i32, numConsumers = 2 : i32}> : () -> !tnca.channel<2, 2>
  %ch = "tnca.channel.create"() {depth = 2 : i32, numConsumers = 2 : i32}
      : () -> !tnca.channel<2, 2>
  %w, %l = "tnca.producer.acquire"(%ch)
      : (!tnca.channel<2, 2>) -> (!tnca.window<8>, !tnca.lease)
  %cw, %cl = "tnca.consumer.acquire"(%ch)
      : (!tnca.channel<2, 2>) -> (!tnca.window<8>, !tnca.lease)
  return
}
