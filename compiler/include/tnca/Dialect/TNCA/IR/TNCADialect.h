//===- TNCADialect.h ------------------------------------------------------===//
//
// Include order matters: the interface declarations must land before the op
// declarations, because an op that carries DeclareOpInterfaceMethods inherits
// from the interface's trait class.
//
//===----------------------------------------------------------------------===//

#ifndef TNCA_DIALECT_TNCA_IR_TNCADIALECT_H
#define TNCA_DIALECT_TNCA_IR_TNCADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tnca/Dialect/TNCA/IR/TNCASupport.h"

#include "tnca/Dialect/TNCA/IR/TNCADialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCATypes.h.inc"

#include "tnca/Dialect/TNCA/IR/TNCAInterfaces.h.inc"

#define GET_OP_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCAOps.h.inc"

namespace mlir {
namespace tnca {

/// One encodable TNCA instruction, from the generated ISA table.
///
/// compiler_overview.md sec 11.3: the numbers come from
/// isa/tnca-isa-v0.1.json, which is generated from the same tables the
/// assembler and the RTL tests use. Nothing here is typed by hand.
struct InstDesc {
  const char *mnemonic;
  unsigned major;
  unsigned func;
  bool maskable;
};

/// Look up an instruction by mnemonic, or null.
const InstDesc *lookupInstruction(llvm::StringRef mnemonic);

/// How many instructions the generated table holds.
unsigned getNumInstructions();

} // namespace tnca
} // namespace mlir

#endif // TNCA_DIALECT_TNCA_IR_TNCADIALECT_H
