cmake_minimum_required(VERSION 3.16)
project(tnca_tlm VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(SystemCLanguage 3.0 REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT SystemCLanguage_VERSION VERSION_LESS 4)
  message(FATAL_ERROR "This model is qualified with SystemC 3.x only")
endif()
add_executable(tnca-tlm src/main.cpp)
target_compile_definitions(tnca-tlm PRIVATE SC_INCLUDE_DYNAMIC_PROCESSES)
target_compile_options(tnca-tlm PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(tnca-tlm PRIVATE SystemC::systemc)
install(TARGETS tnca-tlm RUNTIME DESTINATION bin)
set(ISA_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/../../../isa/tnca-isa-v0.1.json")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tnca_isa_generated.hpp"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_isa.py"
          "${ISA_SCHEMA}" "${CMAKE_CURRENT_BINARY_DIR}/tnca_isa_generated.hpp"
  DEPENDS "${ISA_SCHEMA}" generate_isa.py VERBATIM)
set(SFU_TABLES "${CMAKE_CURRENT_SOURCE_DIR}/../../../xls/tnca_sfu_tables.x")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tnca_sfu_tables_generated.hpp"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_sfu.py"
          "${SFU_TABLES}" "${CMAKE_CURRENT_BINARY_DIR}/tnca_sfu_tables_generated.hpp"
  DEPENDS "${SFU_TABLES}" generate_sfu.py VERBATIM)
add_executable(tnca-chip src/chip.cpp "${CMAKE_CURRENT_BINARY_DIR}/tnca_isa_generated.hpp"
  "${CMAKE_CURRENT_BINARY_DIR}/tnca_sfu_tables_generated.hpp")
target_include_directories(tnca-chip PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(tnca-chip PRIVATE SC_INCLUDE_DYNAMIC_PROCESSES)
target_compile_options(tnca-chip PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(tnca-chip PRIVATE SystemC::systemc)
install(TARGETS tnca-chip RUNTIME DESTINATION bin)
add_executable(tnca-dram-path-test src/dram_path_test.cpp)
target_compile_options(tnca-dram-path-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(tnca-dram-path-test PRIVATE SystemC::systemc)
add_executable(tnca-numeric-probe src/numeric_probe.cpp "${CMAKE_CURRENT_BINARY_DIR}/tnca_sfu_tables_generated.hpp")
target_include_directories(tnca-numeric-probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(tnca-numeric-probe PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Slow DRAM path (dram_path=1): Ramulator2 behind a C++20 bridge library with a
# C++17-clean header, so the SystemC translation units stay C++17. Pinned
# commit: ramulator/PIN. Configure with -DRAMULATOR2_DIR=<built checkout>.
set(RAMULATOR2_DIR "" CACHE PATH "Built Ramulator2 checkout (libramulator.so in its root)")
if(RAMULATOR2_DIR)
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/ramulator/PIN" RAMULATOR2_PIN LIMIT_COUNT 1)
  execute_process(COMMAND git -C "${RAMULATOR2_DIR}" rev-parse HEAD OUTPUT_VARIABLE RAMULATOR2_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git -C "${RAMULATOR2_DIR}" status --porcelain --untracked-files=no OUTPUT_VARIABLE RAMULATOR2_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT RAMULATOR2_COMMIT STREQUAL RAMULATOR2_PIN)
    message(FATAL_ERROR "Ramulator2 at ${RAMULATOR2_COMMIT}, pinned ${RAMULATOR2_PIN} (ramulator/PIN)")
  endif()
  if(RAMULATOR2_DIRTY)
    message(FATAL_ERROR "Ramulator2 checkout has local changes; the slow path is qualified on the pristine pin")
  endif()
  add_library(tnca-dram-bridge STATIC ramulator/tnca_dram_bridge.cpp)
  set_target_properties(tnca-dram-bridge PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(tnca-dram-bridge PRIVATE "${RAMULATOR2_DIR}/src" "${RAMULATOR2_DIR}/ext/yaml-cpp/include"
    "${RAMULATOR2_DIR}/ext/fmt/include")
  target_include_directories(tnca-dram-bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ramulator")
  target_link_libraries(tnca-dram-bridge PUBLIC "${RAMULATOR2_DIR}/libramulator.so")
  set_target_properties(tnca-chip PROPERTIES BUILD_RPATH "${RAMULATOR2_DIR}" INSTALL_RPATH "${RAMULATOR2_DIR}")
  target_compile_definitions(tnca-chip PRIVATE TNCA_HAVE_RAMULATOR2=1
    TNCA_RAMULATOR2_COMMIT="${RAMULATOR2_COMMIT}"
    TNCA_RAMULATOR2_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/ramulator/hbm2_u280_pc.yaml")
  target_link_libraries(tnca-chip PRIVATE tnca-dram-bridge)
  add_executable(tnca-dram-bridge-test ramulator/bridge_test.cpp)
  target_compile_options(tnca-dram-bridge-test PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(tnca-dram-bridge-test PRIVATE tnca-dram-bridge)
  set_target_properties(tnca-dram-bridge-test PROPERTIES BUILD_RPATH "${RAMULATOR2_DIR}")
endif()
