add_mlir_dialect_library(MLIRTNCA
  TNCADialect.cpp
  TNCASupport.cpp
  TNCAAsyncOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tnca/Dialect/TNCA/IR

  DEPENDS
  TNCAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSideEffectInterfaces
)
