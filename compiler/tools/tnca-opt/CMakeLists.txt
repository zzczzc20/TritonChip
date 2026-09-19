get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tnca-opt tnca-opt.cpp)
llvm_update_compile_flags(tnca-opt)

# Out of tree those two globals are empty -- they are populated by
# add_mlir_dialect_library calls, which happen in LLVM's own build, not ours.
# So the aggregate registration libraries are named explicitly; without them
# registerAllDialects and registerAllPasses have no definition.
target_link_libraries(tnca-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRTNCA
  MLIRTNCATransforms
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIROptLib
)
mlir_check_all_link_libraries(tnca-opt)
