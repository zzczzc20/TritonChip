//===- tnca-opt.cpp - The TNCA IR driver ----------------------------------===//
//
// compiler_overview.md sec 15 C0 lists `tnca-opt` among the first
// deliverables, before any lowering. It is the tool every IR unit test of
// sec 16.2 runs against, so it exists first and stays boring: it registers the
// upstream dialects and the TNCA dialect, and runs whatever pass pipeline the
// command line names.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "tnca/Dialect/TNCA/IR/TNCADialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::tnca::TNCADialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TNCA optimiser driver\n", registry));
}
