get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tnca-opt tnca-opt.cpp)
llvm_update_compile_flags(tnca-opt)
target_link_libraries(tnca-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRTNCA
  MLIROptLib
)
mlir_check_all_link_libraries(tnca-opt)
