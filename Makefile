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
#   make e2e        Triton source -> the four-tile machine -> checked numbers
#
#   R1 acceptance (doc/phase1_tasks_and_acceptance.md). These are expected to
#   FAIL until R1 is finished; a gate that passes on partial delivery is the
#   thing the contract exists to prevent.
#   make r1-doctor          what this machine can and cannot certify
#   make r1-check-generated generated artefacts match their sources (read-only)
#   make r1-gate-selftest   ten injected failures, each must be detected
#   make r1-acceptance      the full gate; writes an evidence directory
#   make r1-verify-evidence EVIDENCE=<dir>   re-verify a run's evidence
#   make xls        regenerate rtl/gen from the DSLX in xls/
#   make xls-check  fail if rtl/gen has drifted from xls/
#   make xls-report measure the scheduled clock period of each generated unit
#   make lint       parse/elaborate every RTL file standalone
#   make clean
#
# Tools: iverilog (>= 11), yosys (>= 0.40), python3 (>= 3.9). No PDK required.
#==============================================================================

SHELL      := /bin/bash
# NOTE ON PIPELINES IN RECIPES
#
# `python ... | tail -4` reports tail's status, so a failing interpreter, a
# failing test and a crashing assembler all exit 0. The audit reproduced
# exactly that: `make triton-e2e e2e TRITON_PYTHON=/usr/bin/false` returned
# success.
#
# The fix is not `.SHELLFLAGS := -o pipefail`: that was added in GNU Make 3.82
# and the make shipped with macOS is 3.81, which ignores it silently -- so the
# fix would appear to work in CI and do nothing here, which is worse than no
# fix. Recipes therefore have no pipes on a command whose exit code matters,
# and a recipe that truly needs one writes its own `set -o pipefail`.
# tools/r1/selftest.py RUN-01 fails the gate if this ever regresses.

# Release mode. A missing tool is a SKIP while developing and a FAILURE when
# the run is supposed to be evidence: doc/phase1_tasks_and_acceptance.md
# forbids a skip from producing an R1 PASS.
R1_STRICT  ?= 0
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
# Include files are prerequisites too. tb_core.vvp had neither rtl/gen nor a
# single .vh in its prerequisites, so editing either left a stale simulator in
# place and the next run "verified" RTL that no longer existed (audit F04).
RTL_VH     := $(wildcard $(RTL)/common/*.vh) $(wildcard $(RTL)/noc/*.vh)
RTL_ALL    := $(RTL_COMMON) $(RTL_CORE) $(RTL_GEN) $(RTL_NUCA) $(RTL_NOC) $(RTL_TILE)

# What any bench built with `-y` library search can pull in. Over-approximate
# on purpose: a missing edge is a stale artefact presented as a test result,
# and an extra edge is a rebuild.
RTL_DEPS   := $(RTL_ALL) $(RTL_VH)

#------------------------------------------------------------------------------
# Build fingerprints (doc/phase1_tasks_and_acceptance.md R1-03)
#
# File prerequisites cover what make can see. A fingerprint covers what it
# cannot: the iverilog version, and the -P parameter overrides a bench is
# elaborated with. Those change what the output *is* without changing any
# file's timestamp, so without this a 2-RCTX simulator can be reused to test a
# 4-RCTX profile. tools/r1/fingerprint.py rewrites the stamp only when the
# hash changes, which is what makes it usable as a prerequisite.
#------------------------------------------------------------------------------
.PHONY: FORCE
FORCE:

$(BUILD)/%.fp: FORCE | dirs
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py stamp $@ $(FP_PARAMS_$*) -- $(FP_FILES_$*)

FP_FILES_tb_core       := $(RTL_DEPS) $(TB)/core/tb_tnca_core.v $(TB)/common/tnca_mem_model.v
FP_FILES_tb_core_slow  := $(FP_FILES_tb_core)
FP_FILES_tb_core_ooo   := $(FP_FILES_tb_core)
FP_FILES_tb_core4      := $(FP_FILES_tb_core)
FP_FILES_tb_top        := $(RTL_DEPS) $(TB)/top/tb_tnca_top.v $(TB)/common/tnca_dram_model.v $(RTL)/tnca_top.v
FP_FILES_tb_mesh       := $(RTL_DEPS) $(TB)/noc/tb_mesh.v
FP_FILES_tb_nuca_slice := $(RTL_DEPS) $(TB)/nuca/tb_nuca_slice.v

FP_PARAMS_tb_core       := profile=core-sim
FP_PARAMS_tb_core_slow  := profile=core-sim MEM_LATENCY=7 RANDOM_STALL=1
FP_PARAMS_tb_core_ooo   := profile=core-sim MEM_LATENCY=6 RANDOM_STALL=1 MEM_REORDER=1
FP_PARAMS_tb_core4      := profile=core-4r NUM_RCTX=4 RCTX_W=2 VREGS_PER_RCTX=32
FP_PARAMS_tb_top        := profile=system-2x2
FP_PARAMS_tb_mesh       := profile=system-2x2 npkt=$(MESH_PKTS)
FP_PARAMS_tb_nuca_slice := profile=system-2x2

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
              t11_matrix t12_tensor t13_share t14_caps t15_target t16_vecadd t17_descfault t18_squash t19_waitgen0 t20_sfu t21_cvt t22_atomic t23_amofault t25_mmasf t26_stfault t28_count t29_vrfbank

# Image size in 64-bit words; must match tb_tnca_core's WORDS parameter.
IMAGE_WORDS := 16384

# Per-program plusargs. t04 and t05 need every role launched; t05 is a fault
# test, so a clean run is the failure.
ARGS_t04_roles := +allroles
ARGS_t05_fault := +allroles +expectfault=01
# t13 needs two contexts AND an aliased vector base: that is the whole point.
ARGS_t13_share := +allroles +aliasvrf
ARGS_t17_descfault := +expectfault=04
ARGS_t23_amofault := +expectfault=03
# t26 issues t.st.async, which this engine cannot source; ILLEGAL_INSTRUCTION
# is the contract, and running to completion is the failure.
ARGS_t26_stfault := +expectfault=01
# t15 checks target facts against the hardware, and NUM_ROLES is the count this
# LAUNCH activated (ISA sec 5.1) rather than the core's architectural NUM_RCTX.
# Those agree only when every context is started, so this one starts them all.
ARGS_t15_target := +allroles
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
      $(BUILD)/tb_cvt_xls.vvp $(BUILD)/cvt_xls_vectors.txt \
      $(BUILD)/tb_sfu_xls.vvp $(BUILD)/tb_sfu_core.vvp $(BUILD)/sfu_vectors.txt \
      $(BUILD)/tb_fpu_multi.vvp $(BUILD)/fpu_multi_vectors.txt
	@$(TOOLS)/run_test.sh "bitops" "$(VVP) $(BUILD)/tb_bitops.vvp"
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
$(BUILD)/tb_core.vvp: $(TB)/core/tb_tnca_core.v $(RTL_DEPS) \
                      $(TB)/common/tnca_mem_model.v $(BUILD)/tb_core.fp | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common -o $@ $<
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_core.fp

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

$(TB)/core/progs/t21_cvt.s: $(TB)/core/progs/gen_t21_cvt.py \
                            $(TOOLS)/model/floatfmt.py
	$(PYTHON) $<

$(TB)/core/progs/t22_atomic.s: $(TB)/core/progs/gen_t22_atomic.py
	$(PYTHON) $<

$(TB)/core/progs/t25_mmasf.s: $(TB)/core/progs/gen_t25_mmasf.py \
                              $(TOOLS)/model/fp32.py
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

$(BUILD)/tb_cvt_xls.vvp: $(TB)/xls/tb_cvt_xls.v $(RTL)/gen/tnca_cvt_xls.v \
                         $(RTL)/gen/tnca_cvt_comb_xls.v | dirs
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
core: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core: export ARGS_t26_stfault := $(ARGS_t26_stfault)
core: export ARGS_t15_target := $(ARGS_t15_target)
core: export ARGS_t18_squash := $(ARGS_t18_squash)
# The candidate, on the same programs with the same arguments as `core`.
# A difference in the result is then a difference in the register file and
# not in how the two were run.
.PHONY: core-vrf
core-vrf: export ARGS_t04_roles := $(ARGS_t04_roles)
core-vrf: export ARGS_t05_fault := $(ARGS_t05_fault)
core-vrf: export ARGS_t13_share := $(ARGS_t13_share)
core-vrf: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-vrf: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-vrf: export ARGS_t26_stfault := $(ARGS_t26_stfault)
core-vrf: export ARGS_t15_target := $(ARGS_t15_target)
core-vrf: $(BUILD)/tb_core_vrf.vvp $(CORE_HEX)
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (banked vrf)" \
	     "$(VVP) $(BUILD)/tb_core_vrf.vvp +hex=$(BUILD)/$$p.hex +maxcycles=200000 $$extra" \
	     || exit 1; \
	 done

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
core-slowmem: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-slowmem: export ARGS_t26_stfault := $(ARGS_t26_stfault)
core-slowmem: export ARGS_t15_target := $(ARGS_t15_target)
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
core-ooo: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-ooo: export ARGS_t26_stfault := $(ARGS_t26_stfault)
core-ooo: export ARGS_t15_target := $(ARGS_t15_target)
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

# doc/phase1_tasks_and_acceptance.md R1-08 REG-06/REG-07: three seeds, and the
# seed must reach the PRNG that controls the stimulus rather than appearing in
# a log line. Both benches take +seed and print a trace hash, so "three seeds"
# is checkable: three different hashes, and the same seed replayable.
R1_SEEDS ?= 1 7 42

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
	   for sd in $(R1_SEEDS); do \
	     $(TOOLS)/run_test.sh "BASE-NUCA-$$4-seed$$sd" \
	       "$(VVP) $(BUILD)/tb_nuca_$$4.vvp +nreq=$(NUCA_REQS) +min_wb=1 +min_block=$$5 +seed=$$sd" \
	       || exit 1; \
	   done; \
	 done

#------------------------------------------------------------------------------
# Network on chip
#------------------------------------------------------------------------------
MESH_PKTS ?= 20000

$(BUILD)/tb_mesh.vvp: $(TB)/noc/tb_mesh.v $(RTL_DEPS) $(BUILD)/tb_mesh.fp | dirs
	$(IVERILOG) -g2012 $(IVFLAGS) $(RTLY) -o $@ $< $(RTL)/noc/tnca_mesh.v
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_mesh.fp

.PHONY: mesh
mesh: $(BUILD)/tb_mesh.vvp
	@for sd in $(R1_SEEDS); do \
	   $(TOOLS)/run_test.sh "BASE-MESH-2x2-seed$$sd" \
	     "$(VVP) $(BUILD)/tb_mesh.vvp +npkt=$(MESH_PKTS) +seed=$$sd" \
	     || exit 1; \
	 done

#------------------------------------------------------------------------------
# Full 2x2 system: cores, slices, mesh, DRAM
#------------------------------------------------------------------------------
TOP_PROGS := t10_mesh t24_amo_mesh
ARGS_t24_amo_mesh := +needmesh
ARGS_t10_mesh := +needevict +needmesh

$(BUILD)/tb_top.vvp: $(TB)/top/tb_tnca_top.v $(RTL_DEPS) $(RTL)/tnca_top.v \
                     $(TB)/common/tnca_dram_model.v $(BUILD)/tb_top.fp | dirs
	$(IVERILOG) -g2012 $(IVFLAGS) $(RTLY) -o $@ $< $(RTL)/tnca_top.v \
	   $(TB)/common/tnca_dram_model.v
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_top.fp

.PHONY: top
top: export ARGS_t10_mesh := $(ARGS_t10_mesh)
top: export ARGS_t24_amo_mesh := $(ARGS_t24_amo_mesh)
top: $(BUILD)/tb_top.vvp $(patsubst %,$(BUILD)/%.hex,$(TOP_PROGS))
	@for p in $(TOP_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "BASE-TOP-$$p-single-role" \
	     "$(VVP) $(BUILD)/tb_top.vvp +hex=$(BUILD)/$$p.hex +maxcycles=2000000 $$extra" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "BASE-TOP-$$p-all-roles" \
	     "$(VVP) $(BUILD)/tb_top.vvp +hex=$(BUILD)/$$p.hex +maxcycles=2000000 +allroles $$extra" \
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
	@$(PYTHON) $(TOOLS)/isa/gen_support_matrix.py
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
	@$(PYTHON) $(TOOLS)/isa/gen_support_matrix.py --check
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
# Where MLIR lives. The default is the development machine's pinned Triton
# LLVM; an R1 acceptance environment sets it to the distribution's, which is
# what spec/r1_toolchain.lock.json pins (`mlir.mlir_dir`). R1-09 forbids
# referencing a maintainer's own tree, so this must be settable without
# editing anything:
#
#   make tnca-opt LLVM_PREFIX=/usr/lib/llvm-18
#
# TNCA_LLVM does the same for FileCheck, which the IR tests invoke directly.
#
# The default follows the machine rather than the maintainer: the fetched
# toolchain where it exists, then the distribution's LLVM 20 that CI
# installs. A macOS path printed as "not found" on a Linux box is a
# diagnosis of the wrong machine.
LLVM_FETCHED := $(HOME)/.triton/llvm/llvm-macos-arm64
LLVM_PREFIX ?= $(firstword $(wildcard $(LLVM_FETCHED)) \
                           $(wildcard /usr/lib/llvm-20) $(LLVM_FETCHED))
export TNCA_LLVM ?= $(LLVM_PREFIX)

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
# The acceptance environment installs the locked wheel into its own venv and
# points this at it; see spec/r1_toolchain.lock.json `triton`.
TRITON_PYTHON ?= $(HOME)/src/tritonenv/bin/python
# Exported, because the gate runs tools/r1/ir_tests.py directly rather than
# through a make recipe, and that script re-execs itself under this
# interpreter instead of reporting "triton not importable" as a skip.
export TRITON_PYTHON

.PHONY: triton-e2e e2e
triton-e2e: $(BUILD)/tb_core.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) compiler/test/run_suite.py test_triton_e2e \
	       triton-e2e BASE-TRITON-CORE; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  triton-e2e             FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  triton-e2e             SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

# The whole flow: Triton source, through the compiler, onto the four-tile
# machine as a grid, with the answers read back out of the machine and checked
# in Python. `triton-e2e` above stops at one core and lets the program check
# itself; this does not.
e2e: $(BUILD)/tb_top.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) compiler/test/run_suite.py test_e2e_system \
	       e2e BASE-TRITON-SYSTEM; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  e2e                    FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  e2e                    SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

# The slow and reordering memory models, as file targets. They used to be
# built only inside the core-slowmem / core-ooo recipes, so a compiler test
# that picked one up was running against whatever RTL happened to be current
# when that target last ran -- which silently produced failures against
# already-fixed bugs.
# The candidate: the same bench source, the same programs, a different
# vector register file. One source for both implementations means a
# difference between them cannot be a difference in the testbench.
$(BUILD)/tb_core_vrf.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                          $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.USE_BANKED_VRF=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v

$(BUILD)/tb_core_slow.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                           $(TB)/common/tnca_mem_model.v $(BUILD)/tb_core_slow.fp | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=7 -P tb_tnca_core.RANDOM_STALL=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_core_slow.fp

$(BUILD)/tb_core_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                          $(TB)/common/tnca_mem_model.v $(BUILD)/tb_core_ooo.fp | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_core_ooo.fp

# A four-Role-Context elaboration. compiler_overview.md sec 15 C4 defers the
# P + M0 + M1 + E check to "四 RCTX 硬件 profile 可用后" -- once such a profile
# exists. This is that profile: the same core with NUM_RCTX=4, which needs
# half as many logical vector registers each to keep the physical file the
# same size.
$(BUILD)/tb_core4.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                       $(TB)/common/tnca_mem_model.v $(BUILD)/tb_core4.fp | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 \
	   -P tb_tnca_core.VREGS_PER_RCTX=32 \
	   -o $@ $(TB)/core/tb_tnca_core.v
	@$(PYTHON) $(TOOLS)/r1/fingerprint.py provenance $@ $(BUILD)/tb_core4.fp

# The same four-context profile under the two random memory models. R1-08
# REG-09 asks for the last-release semantics to hold "在 slow/ooo 下各通过":
# a consumer that finishes late is exactly the case a window reclaimed on the
# FIRST release gets wrong, and on a fast, in-order memory the two releases
# are so close together that the difference never shows.
$(BUILD)/tb_core4_slow.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v                             $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common 	   -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 	   -P tb_tnca_core.VREGS_PER_RCTX=32 	   -P tb_tnca_core.MEM_LATENCY=7 -P tb_tnca_core.RANDOM_STALL=1 	   -o $@ $(TB)/core/tb_tnca_core.v

$(BUILD)/tb_core4_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v                            $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common 	   -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 	   -P tb_tnca_core.VREGS_PER_RCTX=32 	   -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 	   -P tb_tnca_core.MEM_REORDER=1 	   -o $@ $(TB)/core/tb_tnca_core.v

.PHONY: core4
# All three, not just the fast one. The slow and out-of-order four-context
# benches had rules and no target that asked for them, so a clean environment
# never built them and REG-09 reported "not built" instead of a result -- on
# the acceptance runner, where nobody had built them by hand.
core4: $(BUILD)/tb_core4.vvp $(BUILD)/tb_core4_slow.vvp $(BUILD)/tb_core4_ooo.vvp

.PHONY: compiler-rtl
# sec 16.2 third layer: run the compiler's own output on the RTL. Needs
# sim/build/tb_core.vvp, so it comes after `core`.
compiler-rtl: $(BUILD)/tb_core.vvp $(BUILD)/tb_core4.vvp \
              $(BUILD)/tb_core_slow.vvp $(BUILD)/tb_core_ooo.vvp
	@$(PYTHON) compiler/test/run_tests.py --rtl

test: lint unit compiler core compiler-rtl triton-e2e core-slowmem core-ooo nuca mesh top e2e
	@echo
	@echo "=============================================="
	@echo " TNCA regression: all tests passed"
	@echo "=============================================="

#------------------------------------------------------------------------------
synth: | dirs
	$(PYTHON) flow/synth/synth.py --json $(BUILD)/synth.json --logs $(BUILD)/synth

clean:
	rm -rf $(BUILD)

#------------------------------------------------------------------------------
# R1 acceptance gate -- doc/phase1_tasks_and_acceptance.md
#
# These entries are deliberately separate from `make test`. `test` is the
# development regression and may skip an unavailable tool; the R1 entries are
# evidence, run strict, and treat a skip as a failure.
#------------------------------------------------------------------------------
#==============================================================================
# R2: the AOT compiler and simulation runtime
# (doc/phase2_tasks_and_acceptance.md)
#
# These entries exist as the contract's stable interface. A target that is not
# yet backed by an implementation runs the part that IS implemented and says
# so; none of them prints a pass for work that has not happened.
#==============================================================================
.PHONY: r2-baseline r2-inventory r2-abi-tests r2-artifact-tests \
        r2-compile-tests r2-cache-tests r2-runtime-tests r2-fault-tests \
        r2-grid-tests r2-memory-tests r2-mutation-tests \
        r2-obs-tests r2-e2e-tests r2-dump-tests r3-rf-tests r3-flow-tests \
        r3-count-tests r3-synth-tests r3-mem-synth \
        r3-mutation-tests r3v2-plan r3v2-scout r3v2-prereqs r3v2-mutations r3v2-runtime r3v2-release r3v2-record-environment r3v2-acceptance r3v2-verify-evidence r3v2 \
        r3v2-components r3v2-core r3v2-flow r3v2-e2e r3v2-count r3v2-model r3v2-synth core-vrf \
        r2-package r2-stage-data \
        r3-scout r3-plan r3-model-tests \
        r36-check-plan r36-backend-tests r36-config-tests \
        r36-export r36-rtl-tests r36-mem-tests r36-conformance \
        r36-fidelity \
        r36-io-tests r36-build r36-build-tests \
        r35-scout r35-check-plan r35-layout-tests r35-upstream-tests r35-opt-tests r35-prereqs r35-alloc-tests r35-safety-tests r35-package-tests \
        r35-acceptance r35-verify-evidence r35-perf

# DEP-01. EVIDENCE is an R1 run directory; COMBINED is compare_envs.py's
# output over A and B, which is what makes an R1 verdict complete.
r2-baseline:
	@$(PYTHON) tools/r2/baseline.py $(if $(EVIDENCE),--evidence $(EVIDENCE),) \
	   $(if $(COMBINED),--combined $(COMBINED),)

r2-inventory:
	@$(PYTHON) tools/r2/inventory.py

# Schema cases run anywhere; the device cases added for R2-05/R2-06 need
# the Triton interpreter and a built device, like the rest of the R2 suites.
r2-abi-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/abi_tests.py \
	     --json $(BUILD)/r2_abi_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-abi-tests          FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   $(PYTHON) tools/r2/abi_tests.py \
	     --json $(BUILD)/r2_abi_tests.json; \
	 fi

# Schema cases run anywhere; the device cases added for R2-05/R2-06 need
# the Triton interpreter and a built device, like the rest of the R2 suites.
r2-artifact-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/artifact_tests.py \
	     --json $(BUILD)/r2_artifact_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-artifact-tests     FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   $(PYTHON) tools/r2/artifact_tests.py \
	     --json $(BUILD)/r2_artifact_tests.json; \
	 fi

# The compile driver needs the real Triton frontend and a built core bench:
# CMP-01 runs what it compiled, and an entry point that only type-checks is
# not the entry point the contract asks for.
# The persistent device. R2-08 RT-02 wants 32 launches in one session, and
# R2-11 makes restarting an explicit device-loss recovery, so this bench does
# not exit: it serves requests until told to stop.
# C04 and C05 are specified on a back-pressuring, reordering memory, so
# both implementations need a device elaborated that way. Four devices, one
# bench source: reference and candidate, fast and slow.
$(BUILD)/tnca_device_slow.vvp: $(RTL_DEPS) $(TB)/device/tnca_device.v \
                               $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tnca_device.MEM_LATENCY=7 -P tnca_device.RANDOM_STALL=1 \
	   -P tnca_device.MEM_REORDER=1 \
	   -o $@ $(TB)/device/tnca_device.v

$(BUILD)/tnca_device_vrf_slow.vvp: $(RTL_DEPS) $(TB)/device/tnca_device.v \
                                   $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tnca_device.USE_BANKED_VRF=1 \
	   -P tnca_device.MEM_LATENCY=7 -P tnca_device.RANDOM_STALL=1 \
	   -P tnca_device.MEM_REORDER=1 \
	   -o $@ $(TB)/device/tnca_device.v

$(BUILD)/tnca_device_vrf.vvp: $(RTL_DEPS) $(TB)/device/tnca_device.v \
                              $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tnca_device.USE_BANKED_VRF=1 \
	   -o $@ $(TB)/device/tnca_device.v

# tnca_shell is named rather than found by library search: it lives in
# $(RTL) itself, which is deliberately not on the -y path, because a bench
# picking up a top level by accident is how two benches end up disagreeing
# about which top they are testing.
$(BUILD)/tnca_device.vvp: $(RTL_DEPS) $(TB)/device/tnca_device.v \
                          $(RTL)/tnca_shell.v $(R2_DATA_STAMP) \
                          $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -o $@ $(TB)/device/tnca_device.v $(RTL)/tnca_shell.v

.PHONY: device
device: $(BUILD)/tnca_device.vvp

#==============================================================================
# R3: implementation, performance model and small-scale RTL calibration
# (doc/phase3_tasks_and_acceptance.md)
#==============================================================================

# How fast THIS host simulates, measured rather than remembered. R3's whole
# execution budget is derived from it, and it is not a clock frequency.
r3-scout: $(BUILD)/tb_core.vvp $(BUILD)/tb_top.vvp
	@$(PYTHON) tools/r3/scout.py --json $(BUILD)/r3_scout.json

# Budgets and caps, checked before any simulator starts: the expensive
# mistake in R3 is not a wrong answer, it is an hour spent getting one.
r3-plan:
	@$(PYTHON) tools/r3/plan.py --json $(BUILD)/r3_plan.json

r3-model-tests:
	@$(PYTHON) tools/r3/model_tests.py --json $(BUILD)/r3_model_tests.json

# Stage the data an installed package reads. It lives at the repository root
# for R1's sake and inside the package for everyone else's; copying rather
# than symlinking, because a wheel cannot contain a symlink to a checkout.
R2_DATA := python/tnca/data
R2_DATA_STAMP := $(R2_DATA)/.staged

# A STAMP with prerequisites, not a phony copy.
#
# `python/tnca/data` is what tnca/_data.py reads when it exists, and it
# exists as soon as anything has packaged once. Nothing re-made it when a
# source changed, so an edit to rtl/ or tb/ reached the Makefile's own
# benches and NOT the runtime's -- which went on building its cached
# device from the snapshot and reporting results for code that was no
# longer in the tree. That is audit F04 again, in a second place: a stale
# artefact presented as a test result.
$(R2_DATA_STAMP): $(RTL_DEPS) $(wildcard $(TB)/common/*.v) \
                  $(wildcard $(TB)/common/*.vh) $(TB)/device/tnca_device.v \
                  $(wildcard spec/*.json) $(wildcard compiler/targets/*.json) \
                  $(wildcard isa/*.json) \
                  tools/asm/tnca_asm.py tools/asm/tnca_isa.py
	@$(MAKE) --no-print-directory r2-stage-data

r2-stage-data:
	@rm -rf $(R2_DATA)
	@mkdir -p $(R2_DATA)/spec $(R2_DATA)/compiler/targets $(R2_DATA)/isa
	@cp spec/*.json $(R2_DATA)/spec/
	@cp compiler/targets/*.json $(R2_DATA)/compiler/targets/
	@cp isa/*.json $(R2_DATA)/isa/
	@mkdir -p $(R2_DATA)/rtl $(R2_DATA)/tb
	@cp -R rtl/. $(R2_DATA)/rtl/
	@cp -R tb/common tb/device $(R2_DATA)/tb/
	@rm -rf python/_staged && mkdir -p python/_staged
	@cp tools/asm/tnca_asm.py tools/asm/tnca_isa.py python/_staged/
	@touch $(R2_DATA_STAMP)
	@echo "  staged $$(find $(R2_DATA) -type f | wc -l | tr -d ' ') data file(s) into $(R2_DATA)"

# A non-editable wheel. PKG-01 moves the repository off sys.path and out of
# the working directory afterwards, so an editable install would prove
# nothing.
# No pipeline. `... | tail -3` reports tail's exit code, so a failed build
# would look like a successful one -- which is R1-02's F02 exactly, and
# tools/r1/inventory.py's static scan caught this recipe the first time it
# ran on a clean checkout. `-q` keeps it quiet without a pipe.
r2-package: r2-stage-data
	@rm -rf build $(BUILD)/dist/*.whl
	@$(PYTHON) -m pip wheel --no-deps -q -w $(BUILD)/dist .
	@$(PYTHON) -c "import glob, os; [print('  %d bytes  %s' % (os.path.getsize(w), w)) for w in sorted(glob.glob('$(BUILD)/dist/*.whl'))]"

r3-mutation-tests:
	@$(PYTHON) tools/r3/mutation_tests.py \
	   --json $(BUILD)/r3_mutation_tests.json

#------------------------------------------------------------------------------
# Phase 3.6: multi-backend engineering integration
# (doc/phase3_6_tasks_and_acceptance.md, version 1.0)
#
# Section 7 names the gates. This stage prepares the mainline for a
# separate platform branch; it does not put anything on a board, and the
# sign-off says only R36_ENGINEERING_READY.
#------------------------------------------------------------------------------
# Tutorial
#
# tutorial/ promises that every number on its pages came from running the
# script beside it. That promise decays the moment the compiler changes
# unless something runs them, so this does. Each example checks its own
# answers against an oracle and exits non-zero when they disagree, so
# there is nothing to assert here beyond the exit status.
#------------------------------------------------------------------------------
.PHONY: tutorial
tutorial: device $(R2_DATA_STAMP)
	@if [ ! -x "$(TRITON_PYTHON)" ]; then \
	   if [ "$(R1_STRICT)" = "1" ]; then \
	     echo "  tutorial               FAIL   no Triton at $(TRITON_PYTHON)"; \
	     exit 1; \
	   else \
	     echo "  tutorial               SKIP   no Triton at $(TRITON_PYTHON)"; \
	     exit 0; \
	   fi; \
	 fi; \
	 fail=0; n=0; \
	 for f in tutorial/examples/*.py; do \
	   n=$$((n+1)); \
	   if PYTHONPATH=python:compiler/python $(TRITON_PYTHON) $$f \
	        > $(BUILD)/tutorial-$$(basename $$f .py).log 2>&1; then \
	     printf "    %-40s PASS\n" "$$(basename $$f)"; \
	   else \
	     printf "    %-40s FAIL\n" "$$(basename $$f)"; \
	     tail -12 $(BUILD)/tutorial-$$(basename $$f .py).log; \
	     fail=1; \
	   fi; \
	 done; \
	 if [ $$fail -eq 0 ]; then \
	   echo "  tutorial               PASS   $$n example(s), each against its own oracle"; \
	 else \
	   echo "  tutorial               FAIL"; \
	 fi; \
	 exit $$fail

#------------------------------------------------------------------------------
R36_EVIDENCE ?= $(BUILD)/r36-evidence

r36-check-plan: | dirs
	@$(TRITON_PYTHON) tools/r36/plan.py --json $(BUILD)/r36_plan.json

# Gate A: the backend boundary and the configuration rules.
r36-backend-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/backend_tests.py \
	   --json $(BUILD)/r36_backend_tests.json

r36-config-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/config_tests.py \
	   --json $(BUILD)/r36_config_tests.json

# One suite, every backend. This is the suite a downstream branch runs.
r36-conformance: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/conformance.py \
	   --json $(BUILD)/r36_conformance.json

# Section 5.2: the machine code and the cycles against the frozen
# baseline, measured in a worktree at that commit rather than read out
# of a file this tree wrote.
r36-fidelity: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/fidelity.py --modes B0 O2 \
	   --json $(BUILD)/r36_fidelity.json

# The export package, built and then proved somewhere else entirely.
# No device prerequisite: the point of these cases is that the package
# needs nothing of ours, so the recipe gives it nothing of ours.
r36-export: | dirs
	@$(PYTHON) tools/r36/export.py --out $(BUILD)/tnca-export \
	   --json $(BUILD)/r36_export.json

r36-rtl-tests: | dirs
	@$(PYTHON) tools/r36/rtl_tests.py --json $(BUILD)/r36_rtl_tests.json

# Two memories behind one boundary, and reset with a kernel in flight.
# Fourteen elaborations and fourteen runs, so it is not a per-push check.
r36-mem-tests: | dirs
	@$(PYTHON) tools/r36/mem_tests.py --json $(BUILD)/r36_mem_tests.json

# The single public build entry point. `--what` selects what to
# produce; a platform target that cannot be built says so and does not
# block the rest.
r36-build: | dirs
	@$(PYTHON) tools/r36/build.py --what $(or $(WHAT),all) \
	   --out $(BUILD)/tnca-export --json $(BUILD)/r36_build.json

r36-build-tests: | dirs
	@$(TRITON_PYTHON) tools/r36/build_tests.py \
	   --json $(BUILD)/r36_build_tests.json

# The host protocol, against the real device in a child process.
r36-io-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/io_tests.py \
	   --json $(BUILD)/r36_io_tests.json

#------------------------------------------------------------------------------
# Phase 3.5: reuse the upstream compiler infrastructure
# (doc/phase3_5_tasks_and_acceptance.md, version 1.0)
#
# Section 8 names these entry points. As with R3 v2, a sub-target passing is
# not the stage passing: `r35-acceptance` is the only thing that produces a
# verdict, and a checker that does not exist yet is a non-zero exit.
#------------------------------------------------------------------------------
R35_EVIDENCE ?= $(BUILD)/r35-evidence

# Before any percentage is promised: what the compiler emits today, and
# which of it is waste with a named mechanism.
r35-scout: | dirs
	@$(TRITON_PYTHON) tools/r35/scout.py --json $(BUILD)/r35_scout.json

r35-check-plan: | dirs
	@$(TRITON_PYTHON) tools/r35/plan.py --json $(BUILD)/r35_plan.json

# The layout semantics, against an enumeration that never calls
# LinearLayout. No simulator: a layout is decided at compile time.
r35-layout-tests: | dirs
	@$(TRITON_PYTHON) tools/r35/layout_tests.py \
	   --json $(BUILD)/r35_layout_tests.json

# That the pinned upstream really runs, and that its answers reach the
# machine code rather than being computed and dropped.
r35-upstream-tests: | dirs
	@$(TRITON_PYTHON) tools/r35/upstream_tests.py \
	   --json $(BUILD)/r35_upstream_tests.json

# The optimisation, and whether it is safe: effects intact, floating
# point unchanged bit for bit.
r35-opt-tests: | dirs
	@$(TRITON_PYTHON) tools/r35/opt_tests.py \
	   --json $(BUILD)/r35_opt_tests.json

# The allocation cost, over physical banks, cross-checked against the
# hardware R3 v2 accepted.
r35-alloc-tests: $(BUILD)/tb_core_vrf.vvp $(BUILD)/t29_vrfbank.hex | dirs
	@$(TRITON_PYTHON) tools/r35/alloc_tests.py \
	   --json $(BUILD)/r35_alloc_tests.json

# The inherited safety cases, each pass's positive and negative case,
# and the eight mutations against the checkers the contract names.
r35-safety-tests: $(BUILD)/tb_core_vrf.vvp $(BUILD)/t29_vrfbank.hex | dirs
	@$(TRITON_PYTHON) tools/r35/safety_tests.py \
	   --json $(BUILD)/r35_safety_tests.json

r35-package-tests: | dirs
	@$(TRITON_PYTHON) tools/r35/package_tests.py \
	   --json $(BUILD)/r35_package_tests.json

# What the checkers read. ALLOC-01 cross-checks its cost model against a
# running bench and REG-01 replays suites that need the device, so a
# fresh checkout has to build them before anything is asked.
r35-prereqs: $(BUILD)/tb_core_vrf.vvp $(BUILD)/t29_vrfbank.hex \
             $(BUILD)/tnca_device.vvp $(BUILD)/tnca_device_vrf.vvp \
             $(BUILD)/tb_core.vvp $(CORE_HEX)
	@$(MAKE) --no-print-directory tnca-opt >/dev/null 2>&1 || \
	  echo "  r35-prereqs            NOTE   tnca-opt did not build"
	@echo "  r35-prereqs            OK     benches, programs and tnca-opt"

# The gate. It runs every checker into a per-run evidence directory and
# is the only thing that produces a verdict.
r35-acceptance: | dirs
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(PYTHON) tools/r35/acceptance.py --evidence $(R35_EVIDENCE) \
	     --python $(TRITON_PYTHON); \
	 else \
	   $(PYTHON) tools/r35/acceptance.py --evidence $(R35_EVIDENCE); \
	 fi

R35_EVIDENCE_DIR ?= $(R35_EVIDENCE)/latest

r35-verify-evidence:
	@$(PYTHON) tools/r35/verify_evidence.py $(R35_EVIDENCE_DIR) \
	   --json $(BUILD)/r35_verify_evidence.json

# Device cycles for the frozen workloads, by mode, with the numbers
# checked against the independent oracle. Section 6.3's C, not a host
# wall time and not a model prediction.
r35-perf: | dirs
	@$(TRITON_PYTHON) tools/r35/perf.py --modes B0 O2 O2-no-LL \
	   --json $(BUILD)/r35_perf.json

#------------------------------------------------------------------------------
# R3 v2: the single-core VRF integration
# (doc/phase3_tasks_and_acceptance.md, version 2.0)
#
# Section 6 names these entry points. A sub-target passing is not the stage
# passing: `r3v2-acceptance` is the only thing that produces a verdict, and
# it must return non-zero for a checker that does not exist yet.
#------------------------------------------------------------------------------
R3V2_EVIDENCE ?= $(BUILD)/r3v2-evidence
R3V2_PEER     ?= evidence/r3v2-second-env.json

# What the checkers read. The gate builds these before it runs anything,
# so a checker that reports "not built" is reporting a build failure and
# not a fresh checkout.
# `tnca-opt` is here because REG-01 replays `make r1-ir-tests`, and IR-07
# compares the Python IR against the MLIR one -- with no tnca-opt there is
# no MLIR one, and the case fails for a missing binary.
r3v2-prereqs: $(BUILD)/tb_core.vvp $(BUILD)/tb_core_vrf.vvp $(CORE_HEX) \
              $(BUILD)/tnca_device.vvp $(BUILD)/tnca_device_vrf.vvp
	@$(MAKE) --no-print-directory tnca-opt >/dev/null || \
	  echo "  r3v2-prereqs           NOTE   tnca-opt did not build"
	@echo "  r3v2-prereqs           OK     benches, programs and tnca-opt"

r3v2-plan:
	@$(PYTHON) tools/r3v2/plan.py --json $(BUILD)/r3v2_plan.json

# The gate. It runs every checker, writes the section 6 evidence tree, and
# only then decides -- so a checker that is missing, crashes, or reports an
# open case is the stage not passing, not a line of output to read past.
# The checkers run under the locked Triton where there is one: the
# end-to-end and release cases compile real kernels, and a bare python3
# would report them as absent rather than as run.
r3v2-acceptance: | dirs
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(PYTHON) tools/r3v2/acceptance.py --evidence $(R3V2_EVIDENCE) \
	     --python $(TRITON_PYTHON); \
	 else \
	   $(PYTHON) tools/r3v2/acceptance.py --evidence $(R3V2_EVIDENCE); \
	 fi

# The verifier, run against an evidence tree by someone who did not
# produce it. It recomputes the verdict from the frozen inventory and the
# raw results and ignores what the run claimed.
# EVIDENCE names one run directory. `latest` is the pointer the gate
# leaves behind; an older run is verified by naming it instead.
EVIDENCE ?= $(R3V2_EVIDENCE)/latest

r3v2-verify-evidence:
	@$(PYTHON) tools/r3v2/verify_evidence.py $(EVIDENCE) \
	   --json $(BUILD)/r3v2_verify_evidence.json

# The gate and then an independent recount of its own evidence.
r3v2: r3v2-acceptance r3v2-verify-evidence

r3v2-mutations:
	@$(PYTHON) tools/r3v2/mutation_tests.py \
	   --json $(BUILD)/r3v2_mutation_tests.json

# The main environment compared against a second one's recorded smoke.
r3v2-release:
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r3v2/rel_tests.py \
	     --json $(BUILD)/r3v2_rel_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r3v2-release-tests     FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r3v2-release-tests     SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

# Run in the SECOND environment to produce the record the above reads.
r3v2-record-environment:
	@$(TRITON_PYTHON) tools/r3v2/rel_tests.py --record $(R3V2_PEER)

r3v2-synth:
	@$(PYTHON) tools/r3v2/syn_tests.py --json $(BUILD)/r3v2_syn_tests.json

r3v2-count: $(BUILD)/tb_core.vvp $(BUILD)/tb_core_vrf.vvp $(CORE_HEX)
	@$(PYTHON) tools/r3v2/count_tests.py --json $(BUILD)/r3v2_count_tests.json

r3v2-model:
	@$(PYTHON) tools/r3v2/model_tests.py --json $(BUILD)/r3v2_model_tests.json

r3v2-e2e: $(BUILD)/tnca_device.vvp $(BUILD)/tnca_device_vrf.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r3v2/e2e_tests.py \
	     --json $(BUILD)/r3v2_e2e_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r3v2-e2e-tests         FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r3v2-e2e-tests         SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

# Section 6 names this entry point: the installed candidate profile and
# three dynamic launches from one artifact.
r3v2-runtime: $(BUILD)/tnca_device_vrf.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r3v2/e2e_tests.py --only R3V2-E2E-02 \
	     --json $(BUILD)/r3v2_runtime_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r3v2-runtime-tests     FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r3v2-runtime-tests     SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r3v2-flow: $(BUILD)/tb_core_vrf.vvp $(CORE_HEX)
	@$(PYTHON) tools/r3v2/flow_tests.py --json $(BUILD)/r3v2_flow_tests.json

r3v2-core: $(BUILD)/tb_core.vvp $(BUILD)/tb_core_vrf.vvp $(CORE_HEX)
	@$(PYTHON) tools/r3v2/core_tests.py --json $(BUILD)/r3v2_core_tests.json

r3v2-components:
	@$(PYTHON) tools/r3v2/rf_tests.py --json $(BUILD)/r3v2_rf_tests.json

r3v2-scout: | dirs
	@$(PYTHON) tools/r3v2/scout.py

r3-synth-tests:
	@$(PYTHON) tools/r3/synth_tests.py --json $(BUILD)/r3_synth_tests.json

r3-mem-synth: | dirs
	$(PYTHON) flow/synth/mem_synth.py --json $(BUILD)/mem_synth.json

r3-count-tests: $(BUILD)/tb_core.vvp $(CORE_HEX)
	@$(PYTHON) tools/r3/count_tests.py --json $(BUILD)/r3_count_tests.json

r3-flow-tests:
	@$(PYTHON) tools/r3/flow_tests.py --json $(BUILD)/r3_flow_tests.json

r3-rf-tests:
	@$(PYTHON) tools/r3/rf_tests.py --json $(BUILD)/r3_rf_tests.json

r2-e2e-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/e2e_tests.py \
	     --json $(BUILD)/r2_e2e_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-e2e-tests           FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-e2e-tests           SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-dump-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/dump_tests.py \
	     --json $(BUILD)/r2_dump_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-dump-tests          FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-dump-tests          SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-obs-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/obs_tests.py \
	     --json $(BUILD)/r2_obs_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-obs-tests           FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-obs-tests           SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-mutation-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/mutation_tests.py \
	     --json $(BUILD)/r2_mutation_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-mutation-tests      FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-mutation-tests      SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-memory-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/memory_tests.py \
	     --json $(BUILD)/r2_memory_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-memory-tests        FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-memory-tests        SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-grid-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/grid_tests.py \
	     --json $(BUILD)/r2_grid_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-grid-tests          FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-grid-tests          SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-fault-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/fault_tests.py \
	     --json $(BUILD)/r2_fault_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-fault-tests         FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-fault-tests         SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-runtime-tests: $(BUILD)/tnca_device.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/runtime_tests.py \
	     --json $(BUILD)/r2_runtime_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-runtime-tests       FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-runtime-tests       SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-cache-tests: $(BUILD)/tb_core.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/cache_tests.py \
	     --json $(BUILD)/r2_cache_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-cache-tests         FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-cache-tests         SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

r2-compile-tests: $(BUILD)/tb_core.vvp
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) tools/r2/compile_tests.py \
	     --json $(BUILD)/r2_compile_tests.json; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  r2-compile-tests       FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  r2-compile-tests       SKIP   no Triton at $(TRITON_PYTHON)"; \
	 fi

.PHONY: r1-doctor r1-check-generated r1-gate-selftest r1-acceptance \
        r1-verify-evidence r1-inventory r1-check-artifacts r1-build-tests r1-profile-tests \
        r1-resource-tests r1-mutation-tests r1-ir-tests r1-regression r1-evidence-tests r1-signoff

r1-doctor:
	@$(PYTHON) tools/r1/doctor.py

r1-inventory:
	@$(PYTHON) tools/r1/inventory.py

# Read-only, and with NO prerequisites on purpose.
#
# `make compiler` depends on $(ISA_SCHEMA) and $(TARGET_JSON), so it
# regenerates them and *then* runs --check -- which can only ever agree. The
# audit's F04 names that; this is the separated check, and a stale commit fails
# here instead of being quietly repaired by whoever runs make.
#
# tools/xls/xlsgen.py --check needs the generator host, so it is opt-in via
# R1_XLS=1 rather than turning an unreachable machine into a false failure.
r1-check-generated:
	@$(PYTHON) $(TOOLS)/isa/export_schema.py --check
	@$(PYTHON) $(TOOLS)/isa/gen.py --check all
	@$(PYTHON) $(TOOLS)/isa/gen_support_matrix.py --check
	@$(PYTHON) $(TOOLS)/isa/gen_target.py --check
	@$(PYTHON) $(TOOLS)/isa/check_consistency.py
	@$(PYTHON) tools/r1/inventory.py
	@if [ "$(R1_XLS)" = "1" ]; then $(PYTHON) $(TOOLS)/xls/xlsgen.py --check; \
	 else echo "  r1-check-generated     NOTE   rtl/gen not re-checked "\
	           "(needs the XLS host; set R1_XLS=1)"; fi

# Before anything runs: is the simulator on disk the one these inputs build?
# make can only ask whether the target is newer than its prerequisites, and an
# artefact copied in from another profile is newer than everything. BUILD-07.
r1-check-artifacts:
	@for b in tb_core tb_core_slow tb_core_ooo tb_core4 tb_top tb_mesh; do \
	   if [ -f $(BUILD)/$$b.vvp ]; then \
	     $(PYTHON) $(TOOLS)/r1/fingerprint.py check $(BUILD)/$$b.vvp $(BUILD)/$$b.fp \
	       || exit 1; \
	   fi; \
	 done; \
	 echo "  r1-check-artifacts     PASS   built simulators match their inputs"

r1-build-tests:
	@$(PYTHON) tools/r1/build_tests.py --json $(BUILD)/build_tests.json

r1-profile-tests: $(BUILD)/tb_core.vvp $(BUILD)/tb_core4.vvp $(BUILD)/tb_top.vvp
	@$(PYTHON) tools/r1/profile_tests.py --json $(BUILD)/profile_tests.json

r1-signoff:
	@$(PYTHON) tools/r1/signoff.py $(if $(EVIDENCE),--evidence $(EVIDENCE),)

r1-evidence-tests:
	@$(PYTHON) tools/r1/evidence_tests.py --json $(BUILD)/evidence_tests.json

r1-regression:
	@$(PYTHON) tools/r1/regression.py --json $(BUILD)/regression.json

r1-ir-tests:
	@$(PYTHON) tools/r1/ir_tests.py --json $(BUILD)/ir_tests.json

r1-mutation-tests:
	@$(PYTHON) tools/r1/mutation_tests.py --json $(BUILD)/mutation_tests.json

r1-resource-tests:
	@$(PYTHON) tools/r1/resource_tests.py --json $(BUILD)/resource_tests.json

r1-gate-selftest:
	@$(PYTHON) tools/r1/selftest.py

# R1_EVIDENCE names the run directory, so two environments building the same
# candidate can be told apart in an archive. It changes where the evidence is
# written and nothing about what runs: strict and retry=0 are not options.
R1_EVIDENCE ?=
r1-acceptance: r1-gate-selftest
	@$(PYTHON) tools/r1/gate.py $(if $(R1_EVIDENCE),--evidence sim/r1/$(R1_EVIDENCE),)

r1-verify-evidence:
	@test -n "$(EVIDENCE)" || { echo "usage: make r1-verify-evidence EVIDENCE=<dir>"; exit 2; }
	@$(PYTHON) tools/r1/verify_evidence.py $(EVIDENCE)
