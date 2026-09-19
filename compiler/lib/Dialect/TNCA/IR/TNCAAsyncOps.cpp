//===- TNCAAsyncOps.cpp - Footprints and costs of asynchronous ops --------===//
//
// compiler_overview.md sec 13.4: 大范围隐式寄存器读必须进入编译器模型.
//
// Each of these operations names one register and touches many. A liveness
// analysis built from the operand list sees `mma.async %acc, %a, %b` reading
// two values where the matrix unit will read sixteen registers, over many
// cycles, after the issuing instruction has retired. ISA sec 9C.3 states that
// the source registers of an incomplete m.mma.async may not be overwritten
// and assigns that to the compiler -- so the compiler has to know the extent,
// and that is what these methods return.
//
// sec 5.4 is why they exist as an interface rather than an attribute: MLIR's
// own documentation says MemoryEffectsOpInterface is not a fine-grained
// address-range or alias mechanism, so marking an op MemRead does not say
// *which* sixteen registers. The effects interface stops wrong deletion,
// duplication and motion; these methods carry the range.
//
//===----------------------------------------------------------------------===//

#include "tnca/Dialect/TNCA/IR/TNCADialect.h"

using namespace mlir;
using namespace mlir::tnca;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// The register range a window value covers.
///
/// The base is not known until allocation, so before then a footprint is a
/// length with an unassigned base. The verifier still uses the length: a
/// transfer that would write past it raises DESCRIPTOR_FAULT rather than
/// corrupting a neighbouring allocation (ISA sec 9D.2).
static RFFootprint windowFootprint(Value window) {
  RFFootprint fp;
  fp.file = RegFile::Vector;
  fp.first = 0;
  fp.count = 0;
  if (auto ty = dyn_cast<WindowType>(window.getType()))
    fp.count = ty.getLength();
  return fp;
}

/// The scalar registers a transfer reads for its coordinates.
///
/// ISA sec 9E: consecutive scalar registers starting at Sb, one per
/// descriptor rank. The instruction names only the first, which is exactly
/// the implicit read sec 13.4 warns about.
static RFFootprint coordinateFootprint(unsigned rank) {
  RFFootprint fp;
  fp.file = RegFile::Scalar;
  fp.first = 0;
  fp.count = rank;
  return fp;
}

//===----------------------------------------------------------------------===//
// dma.load
//===----------------------------------------------------------------------===//

SmallVector<RFFootprint> DMALoadOp::getReadFootprints() {
  return {coordinateFootprint(getCoords().size())};
}

SmallVector<RFFootprint> DMALoadOp::getWriteFootprints() {
  return {windowFootprint(getDest())};
}

SmallVector<CompletionKind> DMALoadOp::getCompletionKinds() {
  // One completion per complete command (ISA sec 9B), whatever the transfer
  // expanded into. That is what makes the operation counts in mb.arm a
  // static, checkable property rather than a run-time guess (sec 8.2).
  return {CompletionKind::DestinationReady};
}

SourceReleaseCondition DMALoadOp::getSourceReleaseCondition() {
  return SourceReleaseCondition::OnCommandCompletion;
}

ExecutionUnit DMALoadOp::getExecutionUnit() {
  return ExecutionUnit::TransferEngine;
}

unsigned DMALoadOp::getIssueDemand() { return 1; }

unsigned DMALoadOp::getRFPortDemand() {
  // The engine writes the whole window through the vector write port, one
  // register at a time, so the port cost is the window length -- not one.
  return windowFootprint(getDest()).count;
}

unsigned DMALoadOp::getCommandSlotDemand() { return 1; }

void DMALoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable());
}

//===----------------------------------------------------------------------===//
// dma.store
//===----------------------------------------------------------------------===//

SmallVector<RFFootprint> DMAStoreOp::getReadFootprints() {
  // sec 13.4's third case: the source window is read *late*, after the
  // issuing instruction retires. A liveness analysis that ends the range at
  // the instruction frees registers the engine is still reading.
  return {windowFootprint(getSrc()),
          coordinateFootprint(getCoords().size())};
}

SmallVector<RFFootprint> DMAStoreOp::getWriteFootprints() { return {}; }

SmallVector<CompletionKind> DMAStoreOp::getCompletionKinds() {
  return {CompletionKind::DestinationReady};
}

SourceReleaseCondition DMAStoreOp::getSourceReleaseCondition() {
  return SourceReleaseCondition::OnCommandCompletion;
}

ExecutionUnit DMAStoreOp::getExecutionUnit() {
  return ExecutionUnit::TransferEngine;
}

unsigned DMAStoreOp::getIssueDemand() { return 1; }

unsigned DMAStoreOp::getRFPortDemand() {
  return windowFootprint(getSrc()).count;
}

unsigned DMAStoreOp::getCommandSlotDemand() { return 1; }

void DMAStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable());
  effects.emplace_back(MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// mma.async
//===----------------------------------------------------------------------===//

SmallVector<RFFootprint> MMAAsyncOp::getReadFootprints() {
  // Both fragments in full. The op names the base of each; the matrix unit
  // reads every register of both, which is the whole point of sec 13.4.
  return {windowFootprint(getA()), windowFootprint(getB())};
}

SmallVector<RFFootprint> MMAAsyncOp::getWriteFootprints() {
  RFFootprint acc;
  acc.file = RegFile::Accumulator;
  acc.first = 0;
  acc.count = 1;
  return {acc};
}

SmallVector<CompletionKind> MMAAsyncOp::getCompletionKinds() {
  // One event, covering both ends. ISA v0.1 has no separate source release,
  // so a scheduler must not move a write to A or B ahead of this.
  return {CompletionKind::DestinationReady};
}

SourceReleaseCondition MMAAsyncOp::getSourceReleaseCondition() {
  // sec 10.2 asks the hardware for an earlier source-reusable event. Until
  // there is one, this is the honest answer, and reporting the loose bound
  // instead would let the scheduler overwrite a fragment mid-command.
  return SourceReleaseCondition::OnCommandCompletion;
}

ExecutionUnit MMAAsyncOp::getExecutionUnit() {
  return ExecutionUnit::MatrixUnit;
}

unsigned MMAAsyncOp::getIssueDemand() { return 1; }

unsigned MMAAsyncOp::getRFPortDemand() {
  return windowFootprint(getA()).count + windowFootprint(getB()).count;
}

unsigned MMAAsyncOp::getCommandSlotDemand() { return 1; }
