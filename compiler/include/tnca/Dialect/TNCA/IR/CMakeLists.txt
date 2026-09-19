set(LLVM_TARGET_DEFINITIONS TNCAOps.td)
mlir_tablegen(TNCAOps.h.inc -gen-op-decls)
mlir_tablegen(TNCAOps.cpp.inc -gen-op-defs)
mlir_tablegen(TNCATypes.h.inc -gen-typedef-decls)
mlir_tablegen(TNCATypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TNCADialect.td)
mlir_tablegen(TNCADialect.h.inc -gen-dialect-decls -dialect=tnca)
mlir_tablegen(TNCADialect.cpp.inc -gen-dialect-defs -dialect=tnca)

set(LLVM_TARGET_DEFINITIONS TNCAInterfaces.td)
mlir_tablegen(TNCAInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TNCAInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TNCAOpsIncGen)
