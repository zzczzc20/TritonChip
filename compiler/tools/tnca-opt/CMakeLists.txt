get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tnca-opt tnca-opt.cpp)
llvm_update_compile_flags(tnca-opt)

# Out of tree those two globals are empty -- they are populated by
# add_mlir_dialect_library calls, which happen in LLVM's own build, not ours.
# So the aggregate registration libraries are named explicitly; without them
# registerAllDialects and registerAllPasses have no definition.
#
# They exist only from LLVM 21. Before that, registerAllDialects and friends
# are header-only and the dialect libraries linked above supply the
# definitions -- so naming them unconditionally makes the build fail on an
# MLIR that is otherwise perfectly capable of building this, which is what
# happened on the Linux acceptance environment's MLIR 20.1.2. Checked for,
# not assumed: a hard-coded list of library names is a claim about the
# LLVM being built against.
set(TNCA_REGISTER_LIBS "")
foreach(lib MLIRRegisterAllDialects MLIRRegisterAllPasses
            MLIRRegisterAllExtensions)
  if(TARGET ${lib} OR EXISTS "${LLVM_LIBRARY_DIR}/lib${lib}.a")
    list(APPEND TNCA_REGISTER_LIBS ${lib})
  endif()
endforeach()
if(TNCA_REGISTER_LIBS)
  message(STATUS "Aggregate registration libraries: ${TNCA_REGISTER_LIBS}")
else()
  message(STATUS "No aggregate registration libraries (MLIR < 21); "
                 "registerAll* comes from the dialect libraries")
endif()

target_link_libraries(tnca-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRTNCA
  MLIRTNCATransforms
  ${TNCA_REGISTER_LIBS}
  MLIROptLib
)
mlir_check_all_link_libraries(tnca-opt)
