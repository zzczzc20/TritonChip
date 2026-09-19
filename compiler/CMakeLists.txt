# TNCA out-of-tree MLIR backend (compiler_overview.md sec 15, C0).
#
# Needs the LLVM/MLIR revision pinned in compiler/pin/toolchain.json. Triton
# requires its own pinned revision rather than an arbitrary release, so
# building against a system LLVM is not supported here either.
#
#   cmake -B build -G Ninja \
#         -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
#         -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm
#   cmake --build build
#
# See compiler/README.md for how to obtain that LLVM.

cmake_minimum_required(VERSION 3.20)
project(tnca-backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# The generated ISA tables (sec 11.3). These are produced by `make isa` from
# isa/tnca-isa-v0.1.json and checked in, so a build needs no Python; the
# custom target exists so an edit to the schema is not silently ignored.
add_custom_target(tnca-isa-check
  COMMAND ${CMAKE_COMMAND} -E echo "checking generated ISA tables"
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../tools/isa/gen.py --check all
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  COMMENT "sec 11.3: generated ISA tables must match the schema")

add_subdirectory(include/tnca/Dialect/TNCA/IR)
add_subdirectory(lib/Dialect/TNCA/IR)
add_subdirectory(tools/tnca-opt)
