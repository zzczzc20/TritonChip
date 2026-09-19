//===- TNCASupport.cpp ----------------------------------------------------===//

#include "tnca/Dialect/TNCA/IR/TNCASupport.h"

namespace mlir {
namespace tnca {

llvm::StringRef getRegFileName(RegFile file) {
  switch (file) {
  case RegFile::Vector:
    return "v";
  case RegFile::Scalar:
    return "s";
  case RegFile::Predicate:
    return "p";
  case RegFile::Accumulator:
    return "a";
  }
  return "?";
}

llvm::StringRef getExecutionUnitName(ExecutionUnit unit) {
  switch (unit) {
  case ExecutionUnit::ScalarALU:
    return "scalar-alu";
  case ExecutionUnit::VectorALU:
    return "vector-alu";
  case ExecutionUnit::VectorFPU:
    return "vector-fpu";
  case ExecutionUnit::SFU:
    return "sfu";
  case ExecutionUnit::LSU:
    return "lsu";
  case ExecutionUnit::TransferEngine:
    return "transfer-engine";
  case ExecutionUnit::MatrixUnit:
    return "matrix-unit";
  case ExecutionUnit::BarrierUnit:
    return "barrier-unit";
  }
  return "unknown";
}

void printFootprint(llvm::raw_ostream &os, const RFFootprint &fp) {
  llvm::StringRef f = getRegFileName(fp.file);
  if (fp.count == 1) {
    os << f << fp.first;
    return;
  }
  os << f << fp.first << ".." << f << fp.last();
}

unsigned fragmentRegisters(unsigned rows, unsigned k, unsigned dtypeBits,
                           unsigned registerBytes) {
  unsigned totalBytes = rows * k * (dtypeBits / 8);
  return (totalBytes + registerBytes - 1) / registerBytes;
}

} // namespace tnca
} // namespace mlir
