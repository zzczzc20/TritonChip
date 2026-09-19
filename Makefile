#==============================================================================
# TNCA -- Tensor-Native Compute Array
#
#   make            build everything and run the full regression
#   make test       run the full regression
#   make unit       unit tests only (FMA, decoder)
#   make isa        regenerate every ISA artifact from the single source
#   make target     regenerate the compiler target description from the RTL
#   make compiler   compiler unit tests + generated-file staleness checks
#   make triton-e2e Triton -> RTL closure (needs the pinned Triton build)
#   make tnca-opt   build the MLIR dialect and tnca-opt (needs the pinned LLVM)
#   make core       core tests only
#   make synth      Yosys synthesis + PPA proxy report
#   make xls        regenerate rtl/gen from the DSLX in xls/
#   make xls-check  fail if rtl/gen has drifted from xls/
#   make xls-report measure the scheduled clock period of each generated unit
#   make lint       parse/elaborate every RTL file standalone
#   make clean
#
# Tools: iverilog (>= 11), yosys (>= 0.40), python3 (>= 3.9). No PDK required.
#==============================================================================

SHELL      := /bin/bash
BUILD      := sim/build
RTL        := rtl
INC        := $(RTL)/common
TB         := tb
TOOLS      := tools

IVERILOG   := iverilog
IVFLAGS    := -g2005 -I $(INC)
VVP        := vvp
YOSYS      := yosys
PYTHON     := python3

ASM        := $(PYTHON) $(TOOLS)/asm/tnca_asm.py

RTL_COMMON := $(wildcard $(RTL)/common/*.v)
RTL_CORE   := $(wildcard $(RTL)/core/*.v)
RTL_NUCA   := $(wildcard $(RTL)/nuca/*.v)
RTL_NOC    := $(wildcard $(RTL)/noc/*.v)
RTL_TILE   := $(wildcard $(RTL)/tile/*.v) $(wildcard $(RTL)/*.v)
RTL_GEN    := $(wildcard $(RTL)/gen/*.v)
RTL_ALL    := $(RTL_COMMON) $(RTL_CORE) $(RTL_GEN) $(RTL_NUCA) $(RTL_NOC) $(RTL_TILE)

# Every RTL directory, for library search and include paths.
RTLY       := -y $(RTL)/common -y $(RTL)/core -y $(RTL)/gen -y $(RTL)/nuca \
              -y $(RTL)/noc -y $(RTL)/tile -I $(RTL)/noc

# Number of random FMA vectors in the default regression. The deep sweep used
# during bring-up is `make fma-deep`.
FMA_VECTORS ?= 20000

# Vectors for the generated units. The conversion bench is exhaustive over
# every narrow format regardless of this; the knob only sizes its binary32
# sample.
SFU_VECTORS ?= 6000
FPU_MULTI_VECTORS ?= 1500

CORE_PROGS := t00_scalar t01_vector t02_mem t03_fp t04_roles t05_fault t06_alu \
              t11_matrix t12_tensor t13_share t14_caps t15_target t16_vecadd t17_descfault t18_squash t19_waitgen0 t20_sfu

# Image size in 64-bit words; must match tb_tnca_core's WORDS parameter.
IMAGE_WORDS := 16384

# Per-program plusargs. t04 and t05 need every role launched; t05 is a fault
# test, so a clean run is the failure.
ARGS_t04_roles := +allroles
ARGS_t05_fault := +allroles +expectfault=01
# t13 needs two contexts AND an aliased vector base: that is the whole point.
ARGS_t13_share := +allroles +aliasvrf
ARGS_t17_descfault := +expectfault=04
ARGS_t18_squash := +allroles

.PHONY: all test unit core synth lint clean dirs help isa target compiler

all: test

help:
	@sed -n '2,14p' Makefile

dirs:
	@mkdir -p $(BUILD)

#------------------------------------------------------------------------------
# Lint: elaborate each RTL file as its own top. Submodules resolve through the
# library path, so this catches per-file syntax, width and port errors without
# needing a testbench for each one.
#------------------------------------------------------------------------------
lint: dirs
	@fail=0; \
	for f in $(RTL_ALL); do \
	  if ! $(IVERILOG) $(IVFLAGS) $(RTLY) -t null $$f 2>$(BUILD)/lint.err; then \
	    echo "LINT FAIL $$f"; cat $(BUILD)/lint.err; fail=1; \
	  fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo "lint: all $(words $(RTL_ALL)) files elaborate"; fi; \
	exit $$fail

#------------------------------------------------------------------------------
# Unit tests
#------------------------------------------------------------------------------
$(BUILD)/tb_fp32_fma.vvp: $(TB)/core/tb_fp32_fma.v $(RTL)/core/tnca_fp32_fma.v \
                          $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^

$(BUILD)/tb_bitops.vvp: $(TB)/core/tb_bitops.v $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^

$(BUILD)/tb_mul64.vvp: $(TB)/core/tb_mul64.v $(RTL)/core/tnca_mul64.v | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^

$(BUILD)/tb_cvt16.vvp: $(TB)/core/tb_cvt16.v $(RTL)/core/tnca_cvt16_f32.v \
                       $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^

$(BUILD)/cvt16_vectors.txt: $(TOOLS)/model/gen_cvt16_vectors.py | dirs
	$(PYTHON) $< -o $@

$(BUILD)/fma_vectors.txt: $(TOOLS)/model/gen_fma_vectors.py $(TOOLS)/model/fp32.py | dirs
	$(PYTHON) $< -o $@ -n $(FMA_VECTORS)

$(BUILD)/tb_decoder.vvp: $(TB)/core/tb_decoder.v $(RTL)/core/tnca_decoder.v | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^

$(BUILD)/dec_vectors.txt: $(TOOLS)/model/gen_decoder_vectors.py \
                          $(TOOLS)/asm/tnca_isa.py $(TOOLS)/asm/tnca_asm.py | dirs
	$(PYTHON) $< -o $@

unit: $(BUILD)/tb_fp32_fma.vvp $(BUILD)/fma_vectors.txt \
      $(BUILD)/tb_decoder.vvp $(BUILD)/dec_vectors.txt $(BUILD)/tb_bitops.vvp \
      $(BUILD)/tb_mul64.vvp \
      $(TB)/core/progs/t03_fp.s $(TB)/core/progs/t06_alu.s \
      $(TB)/core/progs/t11_matrix.s $(TB)/core/progs/t12_tensor.s \
      $(BUILD)/tb_cvt16.vvp $(BUILD)/cvt16_vectors.txt \
      $(BUILD)/tb_cvt_xls.vvp $(BUILD)/cvt_xls_vectors.txt \
      $(BUILD)/tb_sfu_xls.vvp $(BUILD)/tb_sfu_core.vvp $(BUILD)/sfu_vectors.txt \
      $(BUILD)/tb_fpu_multi.vvp $(BUILD)/fpu_multi_vectors.txt
	@$(TOOLS)/run_test.sh "bitops" "$(VVP) $(BUILD)/tb_bitops.vvp"
	@$(TOOLS)/run_test.sh "cvt16"  \
	   "$(VVP) $(BUILD)/tb_cvt16.vvp +vec=$(BUILD)/cvt16_vectors.txt"
	@$(TOOLS)/run_test.sh "mul64"  "$(VVP) $(BUILD)/tb_mul64.vvp"
	@$(TOOLS)/run_test.sh "fp32_fma" \
	   "$(VVP) $(BUILD)/tb_fp32_fma.vvp +vec=$(BUILD)/fma_vectors.txt"
	@$(TOOLS)/run_test.sh "decoder" \
	   "$(VVP) $(BUILD)/tb_decoder.vvp +vec=$(BUILD)/dec_vectors.txt"
	@$(TOOLS)/run_test.sh "cvt_xls" \
	   "$(VVP) $(BUILD)/tb_cvt_xls.vvp +vec=$(BUILD)/cvt_xls_vectors.txt"
	@$(TOOLS)/run_test.sh "sfu_xls" \
	   "$(VVP) $(BUILD)/tb_sfu_xls.vvp +vec=$(BUILD)/sfu_vectors.txt"
	@$(TOOLS)/run_test.sh "sfu_core" \
	   "$(VVP) $(BUILD)/tb_sfu_core.vvp +vec=$(BUILD)/sfu_vectors.txt"
	@$(TOOLS)/run_test.sh "fpu_multi" \
	   "$(VVP) $(BUILD)/tb_fpu_multi.vvp +vec=$(BUILD)/fpu_multi_vectors.txt"
	@$(PYTHON) $(TOOLS)/asm/roundtrip_all.py $(TB)/core/progs/*.s

# Deeper FMA sweep; slow, run it before declaring the unit done.
.PHONY: fma-deep
fma-deep: $(BUILD)/tb_fp32_fma.vvp
	$(PYTHON) $(TOOLS)/model/gen_fma_vectors.py -o $(BUILD)/fma_deep.txt \
	   -n 400000 --seed 424242
	@$(TOOLS)/run_test.sh "fp32_fma (deep)" \
	   "$(VVP) $(BUILD)/tb_fp32_fma.vvp +vec=$(BUILD)/fma_deep.txt"

#------------------------------------------------------------------------------
# Core tests
#------------------------------------------------------------------------------
$(BUILD)/tb_core.vvp: $(TB)/core/tb_tnca_core.v $(RTL_CORE) $(RTL_COMMON) \
                      $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common -o $@ $<

$(TB)/core/progs/t03_fp.s: $(TB)/core/progs/gen_t03_fp.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t06_alu.s: $(TB)/core/progs/gen_t06_alu.py
	$(PYTHON) $<

$(TB)/core/progs/t11_matrix.s: $(TB)/core/progs/gen_t11_matrix.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t12_tensor.s: $(TB)/core/progs/gen_t12_tensor.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t14_caps.s: $(TB)/core/progs/gen_t14_caps.py
	$(PYTHON) $<

$(TB)/core/progs/t15_target.s: $(TB)/core/progs/gen_t15_target.py \
                               compiler/targets/tnca-sim-v0.1.json
	$(PYTHON) $<

$(TB)/core/progs/t16_vecadd.s: $(TB)/core/progs/gen_t16_vecadd.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t17_descfault.s: $(TB)/core/progs/gen_t17_descfault.py
	$(PYTHON) $<

$(TB)/core/progs/t18_squash.s: $(TB)/core/progs/gen_t18_squash.py
	$(PYTHON) $<

$(TB)/core/progs/t20_sfu.s: $(TB)/core/progs/gen_t20_sfu.py \
                            $(TOOLS)/model/gen_sfu_vectors.py
	$(PYTHON) $<

$(TB)/core/progs/t19_waitgen0.s: $(TB)/core/progs/gen_t19_waitgen0.py
	$(PYTHON) $<

#------------------------------------------------------------------------------
# XLS-generated RTL
#
# The Verilog in rtl/gen is checked in, so simulation and synthesis need no
# XLS anywhere. `make xls` regenerates it from the DSLX in xls/ and needs the
# generator host in tools/xls/config.json; `make xls-check` fails if the
# checked-in files have drifted from the sources. Neither is part of `make
# test`, on purpose: a clone should be able to run the whole regression
# without a toolchain that publishes linux-x64 binaries only.
#------------------------------------------------------------------------------
GEN_RTL := $(wildcard $(RTL)/gen/*.v)

.PHONY: xls xls-check xls-report
xls:
	@$(PYTHON) $(TOOLS)/model/gen_sfu_tables.py
	@$(PYTHON) $(TOOLS)/xls/xlsgen.py

xls-check:
	@$(PYTHON) $(TOOLS)/xls/xlsgen.py --check

# What the pipelining bought, in picoseconds under XLS's asap7 delay model.
# Needs the generator host too; the table it prints is in doc/xls.md.
xls-report:
	@$(PYTHON) $(TOOLS)/xls/xlsgen.py --report

$(BUILD)/tb_cvt_xls.vvp: $(TB)/xls/tb_cvt_xls.v $(RTL)/gen/tnca_cvt_xls.v | dirs
	$(IVERILOG) -g2005 -o $@ $^

$(BUILD)/cvt_xls_vectors.txt: $(TOOLS)/model/gen_cvt_vectors.py \
                              $(TOOLS)/model/floatfmt.py | dirs
	$(PYTHON) $< -o $@

$(BUILD)/tb_sfu_xls.vvp: $(TB)/xls/tb_sfu_xls.v $(RTL)/gen/tnca_sfu_exp2_xls.v \
                         $(RTL)/gen/tnca_sfu_log2_xls.v \
                         $(RTL)/gen/tnca_sfu_rcp_xls.v \
                         $(RTL)/gen/tnca_sfu_rsqrt_xls.v | dirs
	$(IVERILOG) -g2005 -o $@ $^

$(BUILD)/tb_sfu_core.vvp: $(TB)/core/tb_sfu_core.v $(RTL)/core/tnca_sfu.v \
                          $(RTL)/gen/tnca_sfu_exp2_xls.v \
                          $(RTL)/gen/tnca_sfu_log2_xls.v \
                          $(RTL)/gen/tnca_sfu_rcp_xls.v \
                          $(RTL)/gen/tnca_sfu_rsqrt_xls.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -o $@ $<

$(BUILD)/sfu_vectors.txt: $(TOOLS)/model/gen_sfu_vectors.py | dirs
	$(PYTHON) $< -o $@ -n $(SFU_VECTORS)

$(BUILD)/tb_fpu_multi.vvp: $(TB)/xls/tb_fpu_multi.v $(RTL)/core/tnca_fpu_multi.v \
                           $(RTL)/core/tnca_fp32_fma.v $(RTL)/gen/tnca_widen_xls.v \
                           $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen \
	   -o $@ $(TB)/xls/tb_fpu_multi.v

$(BUILD)/fpu_multi_vectors.txt: $(TOOLS)/model/gen_fpu_multi_vectors.py \
                                $(TOOLS)/model/floatfmt.py \
                                $(TOOLS)/model/fp32.py | dirs
	$(PYTHON) $< -o $@ -n $(FPU_MULTI_VECTORS)

$(BUILD)/%.hex: $(TB)/core/progs/%.s $(TOOLS)/asm/tnca_asm.py $(TOOLS)/asm/tnca_isa.py | dirs
	$(ASM) $< -o $@ --lst $(BUILD)/$*.lst --size $(IMAGE_WORDS)

CORE_HEX := $(patsubst %,$(BUILD)/%.hex,$(CORE_PROGS))

core: export ARGS_t04_roles := $(ARGS_t04_roles)
core: export ARGS_t05_fault := $(ARGS_t05_fault)
core: export ARGS_t13_share := $(ARGS_t13_share)
core: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core: export ARGS_t18_squash := $(ARGS_t18_squash)
core: $(BUILD)/tb_core.vvp $(CORE_HEX)
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p" \
	     "$(VVP) $(BUILD)/tb_core.vvp +hex=$(BUILD)/$$p.hex +maxcycles=100000 $$extra" \
	     || exit 1; \
	 done

# The same programs under a slower, back-pressuring memory. A core that assumes
# a fixed memory latency passes the first run and fails this one.
.PHONY: core-slowmem
core-slowmem: export ARGS_t04_roles := $(ARGS_t04_roles)
core-slowmem: export ARGS_t05_fault := $(ARGS_t05_fault)
core-slowmem: export ARGS_t13_share := $(ARGS_t13_share)
core-slowmem: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-slowmem: export ARGS_t18_squash := $(ARGS_t18_squash)
core-slowmem: $(CORE_HEX) $(BUILD)/tb_core_slow.vvp | dirs
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (slow mem)" \
	     "$(VVP) $(BUILD)/tb_core_slow.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# And again with responses returning OUT OF ORDER. Once requests reach different
# NUCA slices through a mesh, ordered returns stop being a property anything
# guarantees, so arch overview sec 20.3 lists arbitrary return order as a
# required correctness test. An LSU that silently assumes FIFO passes the two
# runs above and fails this one.
.PHONY: core-ooo
core-ooo: export ARGS_t04_roles := $(ARGS_t04_roles)
core-ooo: export ARGS_t05_fault := $(ARGS_t05_fault)
core-ooo: export ARGS_t13_share := $(ARGS_t13_share)
core-ooo: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-ooo: export ARGS_t18_squash := $(ARGS_t18_squash)
core-ooo: $(CORE_HEX) $(BUILD)/tb_core_ooo.vvp | dirs
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

#------------------------------------------------------------------------------
# NUCA slice
#------------------------------------------------------------------------------
NUCA_REQS ?= 2000

$(BUILD)/tb_nuca_slice.vvp: $(TB)/nuca/tb_nuca_slice.v $(RTL)/nuca/tnca_nuca_slice.v \
                            $(TB)/common/tnca_dram_model.v $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/common -o $@ $<
	@true

# Run the slice under three memory behaviours. A cache that quietly assumes
# ordered returns or a never-stalling memory passes one of these and fails the
# others.
$(BUILD)/tb_nuca_slice_%.vvp: $(TB)/nuca/tb_nuca_slice.v $(RTL)/nuca/tnca_nuca_slice.v \
                              $(TB)/common/tnca_dram_model.v | dirs
	@true

.PHONY: nuca
nuca: | dirs
	@for cfg in "40 1 1 ooo 40" "4 0 0 fast 0" "64 1 0 slow 0"; do \
	   set -- $$cfg; \
	   $(IVERILOG) $(IVFLAGS) -y $(RTL)/common \
	     -P tb_nuca_slice.DRAM_LAT=$$1 -P tb_nuca_slice.DRAM_STALL=$$2 \
	     -P tb_nuca_slice.DRAM_OOO=$$3 \
	     -o $(BUILD)/tb_nuca_$$4.vvp $(TB)/nuca/tb_nuca_slice.v \
	     $(RTL)/nuca/tnca_nuca_slice.v $(TB)/common/tnca_dram_model.v || exit 1; \
	   $(TOOLS)/run_test.sh "nuca slice ($$4)" \
	     "$(VVP) $(BUILD)/tb_nuca_$$4.vvp +nreq=$(NUCA_REQS) +min_wb=1 +min_block=$$5" \
	     || exit 1; \
	 done

#------------------------------------------------------------------------------
# Network on chip
#------------------------------------------------------------------------------
MESH_PKTS ?= 20000

$(BUILD)/tb_mesh.vvp: $(TB)/noc/tb_mesh.v $(RTL_NOC) $(RTL_COMMON) | dirs
	$(IVERILOG) -g2012 $(IVFLAGS) $(RTLY) -o $@ $< $(RTL)/noc/tnca_mesh.v

.PHONY: mesh
mesh: $(BUILD)/tb_mesh.vvp
	@$(TOOLS)/run_test.sh "mesh 2x2" \
	   "$(VVP) $(BUILD)/tb_mesh.vvp +npkt=$(MESH_PKTS)"

#------------------------------------------------------------------------------
# Full 2x2 system: cores, slices, mesh, DRAM
#------------------------------------------------------------------------------
TOP_PROGS := t10_mesh

$(BUILD)/tb_top.vvp: $(TB)/top/tb_tnca_top.v $(RTL_ALL) \
                     $(TB)/common/tnca_dram_model.v | dirs
	$(IVERILOG) -g2012 $(IVFLAGS) $(RTLY) -o $@ $< $(RTL)/tnca_top.v \
	   $(TB)/common/tnca_dram_model.v

.PHONY: top
top: $(BUILD)/tb_top.vvp $(patsubst %,$(BUILD)/%.hex,$(TOP_PROGS))
	@for p in $(TOP_PROGS); do \
	   $(TOOLS)/run_test.sh "$$p (2x2, 1 role)" \
	     "$(VVP) $(BUILD)/tb_top.vvp +hex=$(BUILD)/$$p.hex +maxcycles=2000000" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "$$p (2x2, all roles)" \
	     "$(VVP) $(BUILD)/tb_top.vvp +hex=$(BUILD)/$$p.hex +maxcycles=2000000 +allroles" \
	     || exit 1; \
	 done

#------------------------------------------------------------------------------
#==============================================================================
# The compiler side (doc/compiler_overview.md)
#==============================================================================

# sec 11.3: one machine-readable ISA source. Everything below is generated
# from tools/asm/tnca_isa.py; nothing downstream may be edited by hand.
ISA_SCHEMA  := isa/tnca-isa-v0.1.json
ISA_GEN     := compiler/include/tnca/ISA/TNCAOpcodes.inc \
               compiler/include/tnca/ISA/TNCAInstrInfo.td \
               doc/isa/opcode-map.md

isa:
	@$(PYTHON) $(TOOLS)/isa/export_schema.py
	@$(PYTHON) $(TOOLS)/isa/gen.py all
	@$(PYTHON) $(TOOLS)/isa/check_consistency.py

$(ISA_SCHEMA): $(TOOLS)/asm/tnca_isa.py $(TOOLS)/isa/export_schema.py
	@$(PYTHON) $(TOOLS)/isa/export_schema.py

# sec 15 C0: the target contract, read out of the RTL rather than retyped.
TARGET_JSON := compiler/targets/tnca-sim-v0.1.json

target: $(TARGET_JSON)

$(TARGET_JSON): $(RTL)/core/tnca_core.v $(RTL)/tnca_top.v $(ISA_SCHEMA) \
                $(TOOLS)/isa/gen_target.py
	@$(PYTHON) $(TOOLS)/isa/gen_target.py

# Generated files are checked, not regenerated, so a stale commit fails here
# instead of being quietly repaired by the person who runs `make`.
compiler: $(ISA_SCHEMA) $(TARGET_JSON)
	@$(PYTHON) $(TOOLS)/isa/export_schema.py --check
	@$(PYTHON) $(TOOLS)/isa/gen.py --check all
	@$(PYTHON) $(TOOLS)/isa/gen_target.py --check
	@$(PYTHON) $(TOOLS)/isa/check_consistency.py
	@$(PYTHON) compiler/test/run_tests.py --no-rtl
	@$(PYTHON) compiler/test/IR/run_ir_tests.py

.PHONY: pin
pin:
	@$(PYTHON) $(TOOLS)/isa/verify_pin.py

# The MLIR half. Needs the pinned LLVM, which compiler/pin/fetch_toolchain.sh
# obtains; `make compiler` skips the IR tests rather than failing when the
# build is absent, so a checkout with no LLVM still runs everything else.
LLVM_PREFIX ?= $(HOME)/.triton/llvm/llvm-macos-arm64

.PHONY: tnca-opt llvm-toolchain
llvm-toolchain:
	@compiler/pin/fetch_toolchain.sh

tnca-opt:
	@test -d $(LLVM_PREFIX) || { \
	   echo "pinned LLVM not found at $(LLVM_PREFIX)"; \
	   echo "run: make llvm-toolchain"; exit 1; }
	@cmake -S compiler -B compiler/build -G Ninja \
	       -DMLIR_DIR=$(LLVM_PREFIX)/lib/cmake/mlir \
	       -DLLVM_DIR=$(LLVM_PREFIX)/lib/cmake/llvm \
	       -DCMAKE_BUILD_TYPE=Release > /dev/null
	@ninja -C compiler/build

# The Triton frontend lives in its own virtualenv, because Triton must be
# built against its own pinned LLVM (compiler/pin/toolchain.json) rather than
# whatever the system Python has.
TRITON_PYTHON ?= $(HOME)/src/tritonenv/bin/python

.PHONY: triton-e2e
triton-e2e: $(BUILD)/tb_core.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) compiler/test/test_triton_e2e.py 2>&1 | tail -4; \
	 else \
	   echo "  triton-e2e             SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

# The slow and reordering memory models, as file targets. They used to be
# built only inside the core-slowmem / core-ooo recipes, so a compiler test
# that picked one up was running against whatever RTL happened to be current
# when that target last ran -- which silently produced failures against
# already-fixed bugs.
$(BUILD)/tb_core_slow.vvp: $(RTL_CORE) $(RTL_COMMON) $(TB)/core/tb_tnca_core.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=7 -P tb_tnca_core.RANDOM_STALL=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v

$(BUILD)/tb_core_ooo.vvp: $(RTL_CORE) $(RTL_COMMON) $(TB)/core/tb_tnca_core.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v

# A four-Role-Context elaboration. compiler_overview.md sec 15 C4 defers the
# P + M0 + M1 + E check to "四 RCTX 硬件 profile 可用后" -- once such a profile
# exists. This is that profile: the same core with NUM_RCTX=4, which needs
# half as many logical vector registers each to keep the physical file the
# same size.
$(BUILD)/tb_core4.vvp: $(RTL_CORE) $(RTL_COMMON) $(TB)/core/tb_tnca_core.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 \
	   -P tb_tnca_core.VREGS_PER_RCTX=32 \
	   -o $@ $(TB)/core/tb_tnca_core.v

.PHONY: core4
core4: $(BUILD)/tb_core4.vvp

.PHONY: compiler-rtl
# sec 16.2 third layer: run the compiler's own output on the RTL. Needs
# sim/build/tb_core.vvp, so it comes after `core`.
compiler-rtl: $(BUILD)/tb_core.vvp $(BUILD)/tb_core4.vvp \
              $(BUILD)/tb_core_slow.vvp $(BUILD)/tb_core_ooo.vvp
	@$(PYTHON) compiler/test/run_tests.py --rtl

test: lint unit compiler core compiler-rtl triton-e2e core-slowmem core-ooo nuca mesh top
	@echo
	@echo "=============================================="
	@echo " TNCA regression: all tests passed"
	@echo "=============================================="

#------------------------------------------------------------------------------
synth: | dirs
	$(PYTHON) flow/synth/synth.py --json $(BUILD)/synth.json --logs $(BUILD)/synth

clean:
	rm -rf $(BUILD)
