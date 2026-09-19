//===- TNCAOps.td - TNCA operations -------------------------*- tablegen -*-===//
//
// The operation set of sec 5.3, in the three groups that section names.
// Everything that computes a value stays in tt/arith/math/scf (sec 5.1).
//
//===----------------------------------------------------------------------===//

#ifndef TNCA_OPS
#define TNCA_OPS

include "tnca/Dialect/TNCA/IR/TNCADialect.td"
include "tnca/Dialect/TNCA/IR/TNCAInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Resources and views
//===----------------------------------------------------------------------===//

def TNCA_RFReserveOp : TNCA_Op<"rf.reserve"> {
  let summary = "Reserve a range of vector registers";
  let arguments = (ins I32Attr:$length);
  let results = (outs TNCA_WindowType:$window);
  let assemblyFormat = "$length attr-dict `:` type($window)";
}

def TNCA_RFViewOp : TNCA_Op<"rf.view", [Pure]> {
  let summary = "A sub-range or fragment view of a window";
  let description = [{
    sec 5.3 restricts this to register sub-ranges and fragment views the
    compiler can analyse. It is deliberately not a general RF byte pointer:
    an unanalysable view would defeat the ownership verifier, which is the
    only thing standing between a partitioned kernel and a silent race.
  }];
  let arguments = (ins TNCA_WindowType:$window, I32Attr:$offset, I32Attr:$length);
  let results = (outs TNCA_WindowType:$view);
}

def TNCA_AccReserveOp : TNCA_Op<"acc.reserve"> {
  let summary = "Reserve an accumulator slot";
  let results = (outs TNCA_AccType:$acc);
}

def TNCA_DescSnapshotOp : TNCA_Op<"desc.snapshot"> {
  let summary = "Snapshot a 256-byte TensorMap into a descriptor slot";
  let description = [{
    A snapshot with a version, per ISA sec 9D.3. Bytes 0-127 are semantics
    that must be executed exactly; 128-255 are hints that may be ignored
    entirely, and nothing in them may change an address, a value, an access or
    an order.
  }];
  let arguments = (ins AnyTypeOf<[I64]>:$address);
  let results = (outs TNCA_DescType:$desc);
}

//===----------------------------------------------------------------------===//
// Asynchronous operations
//===----------------------------------------------------------------------===//

def TNCA_DMALoadOp : TNCA_Op<"dma.load", [
    DeclareOpInterfaceMethods<TNCAAsyncOpInterface>,
    DeclareOpInterfaceMethods<TNCAExecutionResourceInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Asynchronous tensor load into an RF window";
  let description = [{
    The destination is a window -- a register range -- because there is
    nowhere else for it to land. Returns an event and a lease rather than
    nothing, so every later use of the range carries a dependence a general
    MLIR pass already honours.
  }];
  let arguments = (ins TNCA_DescType:$desc, TNCA_WindowType:$dest,
                       Variadic<I32>:$coords);
  let results = (outs TNCA_EventType:$event, TNCA_LeaseType:$lease);
}

def TNCA_DMAStoreOp : TNCA_Op<"dma.store", [
    DeclareOpInterfaceMethods<TNCAAsyncOpInterface>,
    DeclareOpInterfaceMethods<TNCAExecutionResourceInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Asynchronous tensor store from an RF window";
  let arguments = (ins TNCA_WindowType:$src, TNCA_DescType:$desc,
                       Variadic<I32>:$coords);
  let results = (outs TNCA_EventType:$event, TNCA_LeaseType:$lease);
}

def TNCA_MMAAsyncOp : TNCA_Op<"mma.async", [
    DeclareOpInterfaceMethods<TNCAAsyncOpInterface>,
    DeclareOpInterfaceMethods<TNCAExecutionResourceInterface>]> {
  let summary = "Asynchronous matrix multiply-accumulate into an accumulator";
  let description = [{
    ISA sec 9C.1 fixes the numerics: `RNE_f32(exact(A) * exact(B) + C)`, a
    single rounding, in increasing k order, bit-reproducible. That contract is
    why a general reassociating optimisation must not touch the k loop, and
    why sec 16.2 says numerical comparison and bit-exact comparison are
    separate questions -- TNCA is not required to match some other GPU's `dot`
    accumulation bit pattern, only its own.

    The lease is the interesting result: the A and B fragments may not be
    rewritten until the source-release condition, which on v0.1 is the whole
    command's completion.
  }];
  let arguments = (ins TNCA_AccType:$acc, TNCA_WindowType:$a,
                       TNCA_WindowType:$b);
  let results = (outs TNCA_EventType:$event, TNCA_LeaseType:$lease);
}

def TNCA_AwaitOp : TNCA_Op<"await"> {
  let summary = "Wait for completion events";
  let description = [{
    Takes the events themselves, so the wait cannot be hoisted above the
    commands it covers. sec 8.4 makes the barrier generation part of the
    verifier: waiting on the wrong phase is the failure this operation exists
    to make expressible and checkable.
  }];
  let arguments = (ins Variadic<TNCA_EventType>:$events);
}

//===----------------------------------------------------------------------===//
// Roles and communication
//===----------------------------------------------------------------------===//

def TNCA_SpecializeOp : TNCA_Op<"specialize", [RecursiveMemoryEffects]> {
  let summary = "Split a region into Role Contexts";
  let description = [{
    The role count is an execution-resource budget, not a layout dimension.
    sec 4.2 and frozen decision 2 keep it out of the warp layout: putting it
    there would make every layout conversion depend on the partition, which is
    chosen much later.
  }];
  let regions = (region VariadicRegion<SizedRegion<1>>:$roles);
}

def TNCA_ChannelCreateOp : TNCA_Op<"channel.create"> {
  let summary = "Create an RF channel between roles";
  let description = [{
    sec 7.4: an RF channel, not a shared-memory Aref. The upstream
    warp-specialization infrastructure communicates through shared memory;
    TNCA has none, so the algorithm is reusable but the communication
    mechanism is not.
  }];
  let arguments = (ins I32Attr:$depth, I32Attr:$numConsumers);
  let results = (outs TNCA_ChannelType:$channel);
}

def TNCA_ProducerAcquireOp : TNCA_Op<"producer.acquire"> {
  let summary = "Take ownership of the next free window in a channel";
  let arguments = (ins TNCA_ChannelType:$channel);
  let results = (outs TNCA_WindowType:$window, TNCA_LeaseType:$lease);
}

def TNCA_ProducerPublishOp : TNCA_Op<"producer.publish"> {
  let summary = "Hand a filled window to the consumers";
  let description = [{
    sec 8.1 is emphatic that this cannot be translated mechanically into an
    `mb.arm` in place: publishing means the fill has completed, the operation
    count for the next phase is known, and the generation has advanced. Those
    are three facts, and an in-place rewrite preserves none of them.
  }];
  let arguments = (ins TNCA_ChannelType:$channel, TNCA_WindowType:$window,
                       Variadic<TNCA_EventType>:$fillEvents);
}

def TNCA_ConsumerAcquireOp : TNCA_Op<"consumer.acquire"> {
  let summary = "Wait for a published window";
  let arguments = (ins TNCA_ChannelType:$channel);
  let results = (outs TNCA_WindowType:$window, TNCA_LeaseType:$lease);
}

def TNCA_ConsumerReleaseOp : TNCA_Op<"consumer.release"> {
  let summary = "Give a window back to the channel";
  let description = [{
    With two consumers the window returns to the free pool only when the
    second releases. sec 17 names this as one of the genuinely hard parts, and
    it is modelled and enumerated in compiler/test/test_protocol.py.
  }];
  let arguments = (ins TNCA_ChannelType:$channel, TNCA_WindowType:$window,
                       Variadic<TNCA_EventType>:$useEvents);
}

//===----------------------------------------------------------------------===//
// Performance metadata
//
// sec 5.3: this group may be dropped, or survive only into the manifest, when
// the hardware has no corresponding instruction. None of it may carry a
// correctness dependence.
//===----------------------------------------------------------------------===//

def TNCA_PrefetchOp : TNCA_Op<"prefetch"> {
  let arguments = (ins TNCA_DescType:$desc, Variadic<I32>:$coords);
}

def TNCA_UseCompleteOp : TNCA_Op<"use_complete"> {
  let summary = "A hint that a window's last read has happened";
  let arguments = (ins TNCA_WindowType:$window);
}

#endif // TNCA_OPS
