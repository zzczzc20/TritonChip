add_mlir_dialect_library(MLIRTNCATransforms
  VerifyAsyncOwnership.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tnca/Dialect/TNCA/Transforms

  DEPENDS
  TNCAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTNCA
  MLIRIR
  MLIRPass
)
