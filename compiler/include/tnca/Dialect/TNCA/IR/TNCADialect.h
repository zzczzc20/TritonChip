//===- TNCADialect.h ------------------------------------------------------===//
#ifndef TNCA_DIALECT_TNCA_IR_TNCADIALECT_H
#define TNCA_DIALECT_TNCA_IR_TNCADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tnca/Dialect/TNCA/IR/TNCADialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCATypes.h.inc"

#define GET_OP_CLASSES
#include "tnca/Dialect/TNCA/IR/TNCAOps.h.inc"

#endif // TNCA_DIALECT_TNCA_IR_TNCADIALECT_H
