//===- Passes.h - TNCA analysis and verification passes ------------------===//
#ifndef TNCA_DIALECT_TNCA_TRANSFORMS_PASSES_H
#define TNCA_DIALECT_TNCA_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tnca {

/// Check that no asynchronous command's registers are disturbed while it is
/// still reading or writing them.
///
/// compiler_overview.md sec 5.4 asks for three mechanisms together, and this
/// is the third: SSA events make the dependence visible, the memory-effects
/// interface stops wrong deletion and motion, and a dedicated verifier checks
/// what neither can -- which registers, across which instructions, roles and
/// loop iterations, are still owned by an engine.
std::unique_ptr<Pass> createVerifyAsyncOwnershipPass();

void registerTNCAPasses();

} // namespace tnca
} // namespace mlir

#endif // TNCA_DIALECT_TNCA_TRANSFORMS_PASSES_H
