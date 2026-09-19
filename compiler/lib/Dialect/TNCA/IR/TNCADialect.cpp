//===- TNCADialect.cpp - TNCA dialect registration ------------------------===//
//
// The opcode numbers this backend emits are not written here. They come from
// tnca/ISA/TNCAOpcodes.inc, generated from isa/tnca-isa-v0.1.json, which is
// generated from the same tables the assembler and the RTL tests use
// (compiler_overview.md sec 11.3: "不要再手工维护第三套独立 opcode 数字").
//
//===----------------------------------------------------------------------===//

#include "tnca/Dialect/TNCA/IR/TNCADialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tnca;

#include "tnca/Dialect/TNCA/IR/TNCADialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCATypes.cpp.inc"

#define GET_OP_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCAOps.cpp.inc"

namespace {
// The ISA, as generated. Kept in one place so a lowering never spells an
// opcode literally.
struct TNCAInstDesc {
  const char *mnemonic;
  unsigned major;
  unsigned func;
  bool maskable;
};

constexpr TNCAInstDesc kInstructions[] = {
#define TNCA_INST(ID, MNEMONIC, MAJOR, FUNC, MASKABLE)                         \
  {MNEMONIC, MAJOR, FUNC, MASKABLE},
#include "tnca/ISA/TNCAOpcodes.inc"
#undef TNCA_INST
};
} // namespace

namespace mlir::tnca {
const TNCAInstDesc *lookupInstruction(llvm::StringRef mnemonic) {
  for (const auto &desc : kInstructions)
    if (mnemonic == desc.mnemonic)
      return &desc;
  return nullptr;
}

unsigned getNumInstructions() { return std::size(kInstructions); }
} // namespace mlir::tnca

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
