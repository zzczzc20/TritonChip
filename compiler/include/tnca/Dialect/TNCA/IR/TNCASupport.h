//===- TNCASupport.h - Plain types the TNCA interfaces speak in -----------===//
//
// The interfaces of compiler_overview.md sec 5.4 return facts about registers
// and completions. Those are ordinary values, not MLIR attributes, so they
// live here rather than in TableGen: a footprint is arithmetic the verifier
// does, not something the IR needs to print.
//
//===----------------------------------------------------------------------===//

#ifndef TNCA_DIALECT_TNCA_IR_TNCASUPPORT_H
#define TNCA_DIALECT_TNCA_IR_TNCASUPPORT_H

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tnca {

/// Which register file a footprint is in.
enum class RegFile : uint8_t { Vector, Scalar, Predicate, Accumulator };

/// A half-open range of registers an engine touches.
///
/// sec 13.4: the range is what matters, not the single index an encoding
/// names. An m.mma.async naming v32 reads v32..v39; building liveness from
/// the operand alone is how v33 gets overwritten mid-command.
struct RFFootprint {
  RegFile file = RegFile::Vector;
  unsigned first = 0;
  unsigned count = 0;

  unsigned last() const { return first + count - 1; }

  bool contains(unsigned index) const {
    return index >= first && index <= last();
  }

  bool overlaps(const RFFootprint &other) const {
    return file == other.file && first <= other.last() &&
           other.first <= last();
  }
};

/// What an asynchronous command signals when it finishes.
///
/// ISA sec 9B counts one *complete* command as one operation, which is why
/// these are counted after the command expands (sec 8.2) rather than per
/// source-level operation.
enum class CompletionKind : uint8_t {
  /// The destination registers or accumulator now hold the result.
  DestinationReady,
  /// The engine has stopped reading the sources.
  SourceReleased,
};

/// When an engine stops reading its source registers.
///
/// sec 10.2 asks the hardware for a source release earlier than completion.
/// ISA v0.1 does not provide one, so `OnCommandCompletion` is the only honest
/// answer today and a scheduler must take that bound. Keeping the enum means
/// a future hardware change tightens the schedule rather than changing the
/// model.
enum class SourceReleaseCondition : uint8_t {
  /// v0.1: one completion covers source and destination.
  OnCommandCompletion,
  /// A separate, earlier event, if the target advertises one.
  OnSeparateSourceEvent,
};

/// The shared unit an operation occupies.
///
/// sec 10.4: the cost model accumulates per shared execution resource,
/// because two roles that look independent in the IR still contend for one
/// SIMD pipe, one transfer engine and one set of RF ports.
enum class ExecutionUnit : uint8_t {
  ScalarALU,
  VectorALU,
  VectorFPU,
  SFU,
  LSU,
  TransferEngine,
  MatrixUnit,
  BarrierUnit,
};

llvm::StringRef getRegFileName(RegFile file);
llvm::StringRef getExecutionUnitName(ExecutionUnit unit);
void printFootprint(llvm::raw_ostream &os, const RFFootprint &fp);

/// Registers a fragment of `rows` x `k` elements occupies.
///
/// ISA sec 9C.2: a fragment is a window's registers read as a flat row-major
/// byte array, which is exactly what the transfer engine writes -- so nothing
/// repacks between a t.ld.async and the m.mma.async consuming it.
unsigned fragmentRegisters(unsigned rows, unsigned k, unsigned dtypeBits,
                           unsigned registerBytes = 64);

} // namespace tnca
} // namespace mlir

#endif // TNCA_DIALECT_TNCA_IR_TNCASUPPORT_H
