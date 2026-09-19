//===- TNCADialect.cpp - TNCA dialect registration ------------------------===//
//
// The opcode numbers this backend emits are not written here. They come from
// tnca/ISA/TNCAOpcodes.inc, generated from isa/tnca-isa-v0.1.json, which is
// generated from the same tables the assembler and the RTL tests use
// (compiler_overview.md sec 11.3: "不要再手工维护第三套独立 opcode 数字").
//
//===----------------------------------------------------------------------===//

#include "tnca/Dialect/TNCA/IR/TNCADialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tnca;

#include "tnca/Dialect/TNCA/IR/TNCADialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCATypes.cpp.inc"

#include "tnca/Dialect/TNCA/IR/TNCAInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCAOps.cpp.inc"

//===----------------------------------------------------------------------===//
// The generated ISA table
//===----------------------------------------------------------------------===//

static constexpr InstDesc kInstructions[] = {
#define TNCA_INST(ID, MNEMONIC, MAJOR, FUNC, MASKABLE)                         \
  {MNEMONIC, MAJOR, FUNC, MASKABLE},
#include "tnca/ISA/TNCAOpcodes.inc"
#undef TNCA_INST
};

namespace mlir {
namespace tnca {

const InstDesc *lookupInstruction(llvm::StringRef mnemonic) {
  for (const auto &desc : kInstructions)
    if (mnemonic == desc.mnemonic)
      return &desc;
  return nullptr;
}

unsigned getNumInstructions() { return std::size(kInstructions); }

} // namespace tnca
} // namespace mlir

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void TNCADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tnca/Dialect/TNCA/IR/TNCAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tnca/Dialect/TNCA/IR/TNCATypes.cpp.inc"
      >();
}
