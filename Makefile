#==============================================================================
# TNCA -- Tensor-Native Compute Array
#
#   make            build everything and run the full regression
#   make test       run the full regression
#   make unit       unit tests only (FMA, decoder)
#   make core       core tests only
#   make synth      Yosys synthesis + PPA proxy report
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
RTL_ALL    := $(RTL_COMMON) $(RTL_CORE) $(RTL_NUCA) $(RTL_NOC) $(RTL_TILE)

# Every RTL directory, for library search and include paths.
RTLY       := -y $(RTL)/common -y $(RTL)/core -y $(RTL)/nuca -y $(RTL)/noc \
              -y $(RTL)/tile -I $(RTL)/noc

# Number of random FMA vectors in the default regression. The deep sweep used
# during bring-up is `make fma-deep`.
FMA_VECTORS ?= 20000

CORE_PROGS := t00_scalar t01_vector t02_mem t03_fp t04_roles t05_fault t06_alu

# Image size in 64-bit words; must match tb_tnca_core's WORDS parameter.
IMAGE_WORDS := 16384

# Per-program plusargs. t04 and t05 need every role launched; t05 is a fault
# test, so a clean run is the failure.
ARGS_t04_roles := +allroles
ARGS_t05_fault := +allroles +expectfault=01

.PHONY: all test unit core synth lint clean dirs help

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
      $(TB)/core/progs/t03_fp.s $(TB)/core/progs/t06_alu.s
	@$(TOOLS)/run_test.sh "bitops" "$(VVP) $(BUILD)/tb_bitops.vvp"
	@$(TOOLS)/run_test.sh "mul64"  "$(VVP) $(BUILD)/tb_mul64.vvp"
	@$(TOOLS)/run_test.sh "fp32_fma" \
	   "$(VVP) $(BUILD)/tb_fp32_fma.vvp +vec=$(BUILD)/fma_vectors.txt"
	@$(TOOLS)/run_test.sh "decoder" \
	   "$(VVP) $(BUILD)/tb_decoder.vvp +vec=$(BUILD)/dec_vectors.txt"
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
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(TB)/common -o $@ $<

$(TB)/core/progs/t03_fp.s: $(TB)/core/progs/gen_t03_fp.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t06_alu.s: $(TB)/core/progs/gen_t06_alu.py
	$(PYTHON) $<

$(BUILD)/%.hex: $(TB)/core/progs/%.s $(TOOLS)/asm/tnca_asm.py $(TOOLS)/asm/tnca_isa.py | dirs
	$(ASM) $< -o $@ --lst $(BUILD)/$*.lst --size $(IMAGE_WORDS)

CORE_HEX := $(patsubst %,$(BUILD)/%.hex,$(CORE_PROGS))

core: export ARGS_t04_roles := $(ARGS_t04_roles)
core: export ARGS_t05_fault := $(ARGS_t05_fault)
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
core-slowmem: $(CORE_HEX) | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=7 -P tb_tnca_core.RANDOM_STALL=1 \
	   -o $(BUILD)/tb_core_slow.vvp $(TB)/core/tb_tnca_core.v
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
core-ooo: $(CORE_HEX) | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 \
	   -o $(BUILD)/tb_core_ooo.vvp $(TB)/core/tb_tnca_core.v
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
test: lint unit core core-slowmem core-ooo nuca mesh top
	@echo
	@echo "=============================================="
	@echo " TNCA regression: all tests passed"
	@echo "=============================================="

#------------------------------------------------------------------------------
synth: | dirs
	$(PYTHON) flow/synth/synth.py --json $(BUILD)/synth.json --logs $(BUILD)/synth

clean:
	rm -rf $(BUILD)
