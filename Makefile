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

# R41 stage inputs are checked with the Torch environment on a remote host.
# r41-freeze-check writes one JSON record (verification, command, software
# tree, git identity) to R41_FREEZE_OUT; copy it into evidence deliberately.
# r41-stage1-acceptance judges the frozen plan's 61 required items from
# evidence/r41/stage1/required_index.json; it fails until every item passes.
.PHONY: r41-freeze-check r41-harness-check r41-stage1-acceptance r41-harness-tests
STAGE ?= 1
R41_HARNESS_OUT ?= $(CURDIR)/sim/build/r41/harness-check
R41_FREEZE_OUT ?= $(CURDIR)/sim/build/r41/freeze-check-stage$(STAGE).json
R41_ACCEPTANCE_OUT ?= $(CURDIR)/sim/build/r41/stage1-acceptance.json
r41-freeze-check:
	$(TORCH_PYTHON) tools/performance_harness/stage_contract.py --stage $(STAGE) --out $(R41_FREEZE_OUT)
r41-harness-check:
	$(TORCH_PYTHON) tools/performance_harness/experiment.py spec/r41/stage1/experiments/harness-smoke.json --root $(CURDIR) --out $(R41_HARNESS_OUT) --python $(TORCH_PYTHON)
r41-stage1-acceptance:
	$(PYTHON) tools/performance_harness/stage1_acceptance.py --root $(CURDIR) --json $(R41_ACCEPTANCE_OUT)
# Torch-free unit tests of the runner, board batches, residuals and acceptance.
r41-harness-tests:
	cd tools/performance_harness && $(PYTHON) test_experiment.py && $(PYTHON) test_board_tools.py && $(PYTHON) test_stage1_acceptance.py && $(PYTHON) test_stage_contract.py
	cd tools/performance_harness/systemc && $(PYTHON) test_run_graphs.py
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
# The board wrapper, which has a top of its own. Nothing else here would find
# it: tools/r36/export.py walks the module graph out from tnca_shell, and
# tnca_fpga_top sits above that rather than below it, so it is neither in the
# manifest nor reached by any wildcard above. Three files were written under
# rtl/fpga before anyone noticed that not one of them had ever been compiled.
RTL_FPGA   := $(wildcard $(RTL)/fpga/*.v)
RTL_ALL    := $(RTL_COMMON) $(RTL_CORE) $(RTL_GEN) $(RTL_NUCA) $(RTL_NOC) $(RTL_TILE) $(RTL_FPGA)

# What any bench built with `-y` library search can pull in. Over-approximate
# on purpose: a missing edge is a stale artefact presented as a test result,
# and an extra edge is a rebuild.
RTL_DEPS   := $(RTL_ALL) $(RTL_VH)

# Where the data an installed package reads is staged, and the stamp that
# says it is current.
#
# Defined HERE, with the other path variables, and not beside the recipe
# that fills it. `:=` expands when make reads the line, so a rule that
# named $(R2_DATA_STAMP) above the definition got an EMPTY prerequisite
# and quietly lost its dependency on staging -- which is the exact
# failure the stamp was added to prevent, reintroduced by where the
# variable sat. `make sim/build/tnca_device.vvp` reported nothing to do
# with a stamp six minutes newer than the target.
R2_DATA := python/tnca/data
R2_DATA_STAMP := $(R2_DATA)/.staged

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

# Every RTL directory, for library search and include paths. `-y $(RTL)` is
# here for rtl/tnca_shell.v: the benches pass it by name, but `lint` compiles
# one file at a time and rtl/fpga/tnca_fpga_top.v instantiates it, so without
# this the fpga wrapper lints as a missing-module error.
RTLY       := -y $(RTL)/common -y $(RTL)/core -y $(RTL)/gen -y $(RTL)/nuca \
              -y $(RTL)/noc -y $(RTL)/tile -y $(RTL)/fpga -y $(RTL) \
              -I $(RTL)/noc

# Number of random FMA vectors in the default regression. The deep sweep used
# during bring-up is `make fma-deep`.
FMA_VECTORS ?= 20000

# Vectors for the generated units. The conversion bench is exhaustive over
# every narrow format regardless of this; the knob only sizes its binary32
# sample.
SFU_VECTORS ?= 6000
FPU_MULTI_VECTORS ?= 1500

# The rank 1-5 transfer refusals (gen_t33_tdesc.py): each is DESCRIPTOR_FAULT,
# and running to completion is the failure.
TDESC_FAULTS := t35_tdesc_frank t35_tdesc_fpad2 t35_tdesc_fnanint t35_tdesc_fsb \
                t35_tdesc_fblock t35_tdesc_fclamp
$(foreach p,$(TDESC_FAULTS),$(eval export ARGS_$(p) := +expectfault=04))

CORE_PROGS := t00_scalar t01_vector t02_mem t03_fp t04_roles t05_fault t06_alu \
              t11_matrix t12_tensor t13_share t14_caps t15_target t16_vecadd t17_descfault t18_squash t19_waitgen0 t20_sfu t21_cvt t22_atomic t23_amofault t25_mmasf t28_count t29_vrfbank t40_f2i \
              t33_tdesc_load t34_tdesc_store t36_reduce t37_lsu_straddle t38_wbfreeze t39_freeze_drop t39_switch t39_isolate t39_slice $(TDESC_FAULTS) \
              t41_boundary

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
# t15 checks target facts against the hardware, and NUM_ROLES is the count this
# LAUNCH activated (ISA sec 5.1) rather than the core's architectural NUM_RCTX.
# Those agree only when every context is started, so this one starts them all.
ARGS_t15_target := +allroles
ARGS_t18_squash := +allroles
# t39 runs the same program on every context at once (R39-05).
export ARGS_t39_switch := +allroles
export ARGS_t39_isolate := +allroles
# t39_slice names the same resources in every context: all of them, each in
# its own slice, where the core has RSLICE (core-r39sw, core-r39sw4); one
# context elsewhere, where the names would really be shared.
export ARGS_t39_slice := +roles=1
# R41 Stage 2: the core's new timing boundaries, on every context at once.
export ARGS_t41_boundary := +allroles

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
# The FMA is generated (xls/tnca_fma.x); tnca_fp32_fma picks the instance for
# its latency. Both latencies are tested against the same oracle.
FMA_GEN := $(RTL)/gen/tnca_fma_align_xls.v $(RTL)/gen/tnca_fma_norm_xls.v \
           $(RTL)/gen/tnca_fma_round_xls.v $(RTL)/gen/tnca_fma_l3_xls.v
$(BUILD)/tb_fp32_fma.vvp: $(TB)/core/tb_fp32_fma.v $(RTL)/core/tnca_fp32_fma.v \
                          $(FMA_GEN) $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -o $@ $^
$(BUILD)/tb_fp32_fma_l3.vvp: $(TB)/core/tb_fp32_fma.v $(RTL)/core/tnca_fp32_fma.v \
                             $(FMA_GEN) $(RTL_COMMON) | dirs
	$(IVERILOG) $(IVFLAGS) -P tb_fp32_fma.FMA_LAT=3 -o $@ $^

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

unit: $(BUILD)/tb_fp32_fma.vvp $(BUILD)/tb_fp32_fma_l3.vvp $(BUILD)/fma_vectors.txt \
      $(BUILD)/tb_decoder.vvp $(BUILD)/dec_vectors.txt $(BUILD)/tb_bitops.vvp \
      $(BUILD)/tb_mul64.vvp \
      $(TB)/core/progs/t03_fp.s $(TB)/core/progs/t06_alu.s \
      $(TB)/core/progs/t11_matrix.s $(TB)/core/progs/t12_tensor.s \
      $(BUILD)/tb_cvt_xls.vvp $(BUILD)/cvt_xls_vectors.txt \
      $(BUILD)/tb_sfu_xls.vvp $(BUILD)/tb_sfu_core.vvp $(BUILD)/sfu_vectors.txt \
      $(BUILD)/tb_fpu_multi.vvp $(BUILD)/fpu_multi_vectors.txt \
      $(BUILD)/tb_vector_fpu_f2i.vvp $(BUILD)/f2i_vectors.txt $(TB)/core/progs/t40_f2i.s
	@$(TOOLS)/run_test.sh "bitops" "$(VVP) $(BUILD)/tb_bitops.vvp"
	@$(TOOLS)/run_test.sh "mul64"  "$(VVP) $(BUILD)/tb_mul64.vvp"
	@$(TOOLS)/run_test.sh "fp32_fma" \
	   "$(VVP) $(BUILD)/tb_fp32_fma.vvp +vec=$(BUILD)/fma_vectors.txt"
	@$(TOOLS)/run_test.sh "fp32_fma (latency 3)" \
	   "$(VVP) $(BUILD)/tb_fp32_fma_l3.vvp +vec=$(BUILD)/fma_vectors.txt"
	@$(TOOLS)/run_test.sh "decoder" \
	   "$(VVP) $(BUILD)/tb_decoder.vvp +vec=$(BUILD)/dec_vectors.txt"
	@$(TOOLS)/run_test.sh "cvt_xls" \
	   "$(VVP) $(BUILD)/tb_cvt_xls.vvp +vec=$(BUILD)/cvt_xls_vectors.txt"
	@$(TOOLS)/run_test.sh "vector_fpu_f2i" \
	   "$(VVP) $(BUILD)/tb_vector_fpu_f2i.vvp +vec=$(BUILD)/f2i_vectors.txt"
	@$(TOOLS)/run_test.sh "sfu_xls" \
	   "$(VVP) $(BUILD)/tb_sfu_xls.vvp +vec=$(BUILD)/sfu_vectors.txt"
	@$(TOOLS)/run_test.sh "sfu_core" \
	   "$(VVP) $(BUILD)/tb_sfu_core.vvp +vec=$(BUILD)/sfu_vectors.txt"
	@$(TOOLS)/run_test.sh "fpu_multi" \
	   "$(VVP) $(BUILD)/tb_fpu_multi.vvp +vec=$(BUILD)/fpu_multi_vectors.txt"
	@$(PYTHON) $(TOOLS)/asm/roundtrip_all.py $(TB)/core/progs/*.s

# Deeper FMA sweep; slow, run it before declaring the unit done.
.PHONY: fma-deep
fma-deep: $(BUILD)/tb_fp32_fma.vvp $(BUILD)/tb_fp32_fma_l3.vvp
	$(PYTHON) $(TOOLS)/model/gen_fma_vectors.py -o $(BUILD)/fma_deep.txt \
	   -n 400000 --seed 424242
	@$(TOOLS)/run_test.sh "fp32_fma (deep)" \
	   "$(VVP) $(BUILD)/tb_fp32_fma.vvp +vec=$(BUILD)/fma_deep.txt"
	@$(TOOLS)/run_test.sh "fp32_fma (deep, latency 3)" \
	   "$(VVP) $(BUILD)/tb_fp32_fma_l3.vvp +vec=$(BUILD)/fma_deep.txt"

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

$(TB)/core/progs/t36_reduce.s: $(TB)/core/progs/gen_t36_reduce.py
	$(PYTHON) $<

$(TB)/core/progs/t37_lsu_straddle.s: $(TB)/core/progs/gen_t37_lsu_straddle.py
	$(PYTHON) $<

$(TB)/core/progs/t38_wbfreeze.s: $(TB)/core/progs/gen_t38_wbfreeze.py $(TB)/core/progs/gen_t33_tdesc.py
	$(PYTHON) $<

$(TB)/core/progs/t39_freeze_drop.s: $(TB)/core/progs/gen_t39_freeze_drop.py $(TB)/core/progs/gen_t33_tdesc.py
	$(PYTHON) $<
$(TB)/core/progs/t39_slice.s: $(TB)/core/progs/gen_t39_slice.py $(TB)/core/progs/gen_t33_tdesc.py
	$(PYTHON) $<
$(TB)/core/progs/t39_pair.s: $(TB)/core/progs/gen_t39_pair.py $(TB)/core/progs/gen_t33_tdesc.py
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
$(TB)/core/progs/t15_target_vrf_v4.s: $(TB)/core/progs/gen_t15_target.py \
                                      compiler/targets/tnca-sim-vrf-v4.json
	$(PYTHON) $< tnca-sim-vrf-v4
$(TB)/core/progs/t15_target_vrf_v5.s: $(TB)/core/progs/gen_t15_target.py \
                                      compiler/targets/tnca-sim-vrf-v5.json
	$(PYTHON) $< tnca-sim-vrf-v5

$(TB)/core/progs/t16_vecadd.s: $(TB)/core/progs/gen_t16_vecadd.py $(TOOLS)/model/fp32.py
	$(PYTHON) $<

$(TB)/core/progs/t17_descfault.s: $(TB)/core/progs/gen_t17_descfault.py
	$(PYTHON) $<

$(TB)/core/progs/t33_tdesc_load.s $(TB)/core/progs/t34_tdesc_store.s \
$(patsubst %,$(TB)/core/progs/%.s,$(TDESC_FAULTS)): $(TB)/core/progs/gen_t33_tdesc.py
	$(PYTHON) $<

$(TB)/core/progs/t18_squash.s: $(TB)/core/progs/gen_t18_squash.py
	$(PYTHON) $<

$(TB)/core/progs/t20_sfu.s: $(TB)/core/progs/gen_t20_sfu.py \
                            $(TOOLS)/model/gen_sfu_vectors.py
	$(PYTHON) $<

$(TB)/core/progs/t40_f2i.s: $(TB)/core/progs/gen_t40_f2i.py \
                            $(TOOLS)/model/f2i.py $(TOOLS)/model/fp32.py
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

# binary32 -> int32/uint32 through tnca_vector_fpu against the ISA text.
$(BUILD)/tb_vector_fpu_f2i.vvp: $(TB)/core/tb_vector_fpu_f2i.v $(RTL)/core/tnca_vector_fpu.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -o $@ $<

$(BUILD)/f2i_vectors.txt: $(TOOLS)/model/gen_f2i_vectors.py $(TOOLS)/model/f2i.py \
                          $(TOOLS)/model/fp32.py | dirs
	$(PYTHON) $< -o $@

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
                           $(RTL)/core/tnca_fp32_fma.v $(FMA_GEN) $(RTL)/gen/tnca_widen_xls.v \
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
core-ooo: export ARGS_t15_target := $(ARGS_t15_target)
core-ooo: export ARGS_t18_squash := $(ARGS_t18_squash)
core-ooo: $(CORE_HEX) $(BUILD)/tb_core_ooo.vvp | dirs
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# R39: the banked core with the R39 hardware switches on (the candidate
# core-vrf-sync-v2), on the bench memory and on the back-pressuring,
# reordering one -- every core program, as core-vrf and core-ooo run them.
R39_HW := -P tb_tnca_core.USE_BANKED_VRF=1 -P tb_tnca_core.VRF_BYPASS=1 \
          -P tb_tnca_core.FE_NOBUBBLE=1 -P tb_tnca_core.VRF_PREFETCH=1
$(BUILD)/tb_core_r39hw.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                            $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_HW) -o $@ $(TB)/core/tb_tnca_core.v
$(BUILD)/tb_core_r39hw_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                                $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_HW) -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 -o $@ $(TB)/core/tb_tnca_core.v
.PHONY: core-r39hw
core-r39hw: export ARGS_t04_roles := $(ARGS_t04_roles)
core-r39hw: export ARGS_t05_fault := $(ARGS_t05_fault)
core-r39hw: export ARGS_t13_share := $(ARGS_t13_share)
core-r39hw: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-r39hw: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-r39hw: export ARGS_t15_target := $(ARGS_t15_target)
core-r39hw: export ARGS_t18_squash := $(ARGS_t18_squash)
core-r39hw: $(CORE_HEX) $(BUILD)/tb_core_r39hw.vvp $(BUILD)/tb_core_r39hw_ooo.vvp | dirs
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (r39 hw)" \
	     "$(VVP) $(BUILD)/tb_core_r39hw.vvp +hex=$(BUILD)/$$p.hex +maxcycles=200000 $$extra" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "$$p (r39 hw, ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_r39hw_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# R39-05/08/10: the same, with RCTX_SWITCH (an instruction held in R by a
# hazard gives way to another context's), LSU_INFLIGHT=2, FPU_OVERLAP and
# RSLICE -- the candidate core-vrf-sync-v3.
R39_SW := $(R39_HW) -P tb_tnca_core.RCTX_SWITCH=1 -P tb_tnca_core.LSU_INFLIGHT=2 \
          -P tb_tnca_core.FPU_OVERLAP=1 -P tb_tnca_core.RSLICE=1
$(BUILD)/tb_core_r39sw.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                            $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW) -o $@ $(TB)/core/tb_tnca_core.v
$(BUILD)/tb_core_r39sw_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                                $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW) -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 -o $@ $(TB)/core/tb_tnca_core.v
.PHONY: core-r39sw
core-r39sw: export ARGS_t04_roles := $(ARGS_t04_roles)
core-r39sw: export ARGS_t05_fault := $(ARGS_t05_fault)
core-r39sw: export ARGS_t13_share := $(ARGS_t13_share)
core-r39sw: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-r39sw: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-r39sw: export ARGS_t15_target := $(ARGS_t15_target)
core-r39sw: export ARGS_t18_squash := $(ARGS_t18_squash)
core-r39sw: export ARGS_t39_slice := +allroles +slice
core-r39sw: $(CORE_HEX) $(BUILD)/tb_core_r39sw.vvp $(BUILD)/tb_core_r39sw_ooo.vvp | dirs
	@for p in $(CORE_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (r39 switch)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw.vvp +hex=$(BUILD)/$$p.hex +maxcycles=200000 $$extra" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "$$p (r39 switch, ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# R39-05: and four Role Contexts of 64 vector registers (core-vrf-sync-v4).
R39_SW4 := $(R39_SW) -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 -P tb_tnca_core.PHYS_W=8
$(BUILD)/tb_core_r39sw4.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                             $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW4) -o $@ $(TB)/core/tb_tnca_core.v
$(BUILD)/tb_core_r39sw4_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                                 $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW4) -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 -o $@ $(TB)/core/tb_tnca_core.v
.PHONY: core-r39sw4
# The programs written for a two-context core start two contexts here
# (+roles=2); t39_switch and t39_isolate start all four.
core-r39sw4: export ARGS_t04_roles := +roles=2
core-r39sw4: export ARGS_t05_fault := +roles=2 +expectfault=01
core-r39sw4: export ARGS_t13_share := +roles=2 +aliasvrf
core-r39sw4: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-r39sw4: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-r39sw4: export ARGS_t15_target_vrf_v4 := +allroles
core-r39sw4: export ARGS_t18_squash := +roles=2
core-r39sw4: export ARGS_t39_slice := +allroles +slice
# t15 asserts the machine's facts against a target file: this one's.
R39SW4_PROGS := $(patsubst t15_target,t15_target_vrf_v4,$(CORE_PROGS))
core-r39sw4: $(patsubst %,$(BUILD)/%.hex,$(R39SW4_PROGS)) $(BUILD)/tb_core_r39sw4.vvp $(BUILD)/tb_core_r39sw4_ooo.vvp | dirs
	@for p in $(R39SW4_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (r39 4 rctx)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw4.vvp +hex=$(BUILD)/$$p.hex +maxcycles=200000 $$extra" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "$$p (r39 4 rctx, ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw4_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# R39-12: and the transposed transfer and Role Context pairs (core-vrf-sync-v5).
# RCTX_SWITCH=2: a context also gives way when it waits for the matrix unit
# or the transfer engine. MTX_ABUF: the matrix unit buffers an MMA's A.
R39_SW5 := $(R39_HW) -P tb_tnca_core.RCTX_SWITCH=2 -P tb_tnca_core.LSU_INFLIGHT=2 \
           -P tb_tnca_core.FPU_OVERLAP=1 -P tb_tnca_core.RSLICE=1 \
           -P tb_tnca_core.NUM_RCTX=4 -P tb_tnca_core.RCTX_W=2 -P tb_tnca_core.PHYS_W=8 \
           -P tb_tnca_core.HAS_TENSOR_XPOSE=1 -P tb_tnca_core.RPAIR=1 \
           -P tb_tnca_core.MTX_ABUF=1
$(BUILD)/tb_core_r39sw5.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                             $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW5) -o $@ $(TB)/core/tb_tnca_core.v
# R41 X8: core-vrf-sync-v5-x8's bench (the 1-byte transposed write).
$(BUILD)/tb_core_r39sw5x8.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                               $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(subst HAS_TENSOR_XPOSE=1,HAS_TENSOR_XPOSE=2,$(R39_SW5)) -o $@ $(TB)/core/tb_tnca_core.v
$(BUILD)/tb_core_r39sw5_ooo.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                                 $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   $(R39_SW5) -P tb_tnca_core.MEM_LATENCY=6 -P tb_tnca_core.RANDOM_STALL=1 \
	   -P tb_tnca_core.MEM_REORDER=1 -o $@ $(TB)/core/tb_tnca_core.v
.PHONY: core-r39sw5
# As core-r39sw4, and t39_pair (two producer/consumer pairs, +pair) and the
# transposed transfer's own programs.
core-r39sw5: export ARGS_t04_roles := +roles=2
core-r39sw5: export ARGS_t05_fault := +roles=2 +expectfault=01
core-r39sw5: export ARGS_t13_share := +roles=2 +aliasvrf
core-r39sw5: export ARGS_t17_descfault := $(ARGS_t17_descfault)
core-r39sw5: export ARGS_t23_amofault := $(ARGS_t23_amofault)
core-r39sw5: export ARGS_t15_target_vrf_v5 := +allroles
core-r39sw5: export ARGS_t18_squash := +roles=2
core-r39sw5: export ARGS_t39_slice := +allroles +slice
core-r39sw5: export ARGS_t39_pair := +allroles +slice +slicek=1 +pair
core-r39sw5: export ARGS_t32_xpose_badsize := +expectfault=04
R39SW5_PROGS := $(patsubst t15_target,t15_target_vrf_v5,$(CORE_PROGS)) t39_pair t30_xpose t32_xpose_badsize
core-r39sw5: $(patsubst %,$(BUILD)/%.hex,$(R39SW5_PROGS)) $(BUILD)/tb_core_r39sw5.vvp $(BUILD)/tb_core_r39sw5_ooo.vvp | dirs
	@for p in $(R39SW5_PROGS); do \
	   extra=$$(eval echo \$${ARGS_$$p}); \
	   $(TOOLS)/run_test.sh "$$p (r39 v5)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw5.vvp +hex=$(BUILD)/$$p.hex +maxcycles=200000 $$extra" \
	     || exit 1; \
	   $(TOOLS)/run_test.sh "$$p (r39 v5, ooo mem)" \
	     "$(VVP) $(BUILD)/tb_core_r39sw5_ooo.vvp +hex=$(BUILD)/$$p.hex +maxcycles=400000 $$extra" \
	     || exit 1; \
	 done

# R41 Stage 2 (F01/F02/F04/F05): t41_boundary on every core bench, with one
# context (every instruction directly behind its producer, so operands
# forward), two and all of them (contexts interleave, give way and wait on
# each other's barriers), printing how often each new boundary's corner case
# occurred (+r41cov). The core's own assertions check, every cycle, that the
# registered operands, busy bits and physical registers equal what the old
# combinational reads returned.
R41_BENCHES := tb_core tb_core_vrf tb_core_slow tb_core_ooo tb_core4 tb_core4_slow \
               tb_core4_ooo tb_core_r39hw tb_core_r39hw_ooo tb_core_r39sw \
               tb_core_r39sw_ooo tb_core_r39sw4 tb_core_r39sw4_ooo tb_core_r39sw5 \
               tb_core_r39sw5_ooo
.PHONY: r41-core-boundary
r41-core-boundary: $(patsubst %,$(BUILD)/%.vvp,$(R41_BENCHES)) $(BUILD)/t41_boundary.hex | dirs
	@for b in $(R41_BENCHES); do for r in +roles=1 +roles=2 +allroles; do \
	   $(TOOLS)/run_test.sh "t41_boundary $$b $$r" \
	     "$(VVP) $(BUILD)/$$b.vvp +hex=$(BUILD)/t41_boundary.hex +maxcycles=400000 $$r +r41cov" \
	     || exit 1; \
	   $(VVP) $(BUILD)/$$b.vvp +hex=$(BUILD)/t41_boundary.hex +maxcycles=400000 $$r +r41cov \
	     | grep '^r41cov' | sed 's/^/      /'; \
	 done; done

# R41 F10: the LSU pair's registered request boundary (REQ_REG=1) against the
# R40 combinational port, as a miter (tb/core/tb_lsu_reqreg.v): random
# programs under back-pressure, long stalls and reordered responses, one with
# an always-ready in-order memory, one with long latency, and the directed
# program that checks one beat per cycle and the one-cycle request latency.
LSU_REQREG_RUNS := "-Ptb_lsu_reqreg.SEED=1" \
  "-Ptb_lsu_reqreg.SEED=2 -Ptb_lsu_reqreg.STALL_PCT=60" \
  "-Ptb_lsu_reqreg.SEED=3 -Ptb_lsu_reqreg.STALL_PCT=0 -Ptb_lsu_reqreg.HOLD_PCT=0 -Ptb_lsu_reqreg.REORDER=0" \
  "-Ptb_lsu_reqreg.SEED=4 -Ptb_lsu_reqreg.MAXLAT=200" \
  "-Ptb_lsu_reqreg.DIRECTED=1"
.PHONY: lsu-reqreg
lsu-reqreg: | dirs
	@for args in $(LSU_REQREG_RUNS); do \
	   $(IVERILOG) -g2012 -I $(INC) -s tb_lsu_reqreg $$args -o $(BUILD)/tb_lsu_reqreg.vvp \
	     $(TB)/core/tb_lsu_reqreg.v $(RTL)/core/tnca_lsu_pair.v $(RTL)/core/tnca_lsu.v || exit $$?; \
	   $(VVP) $(BUILD)/tb_lsu_reqreg.vvp || exit $$?; \
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

# R41 G2: NUCA_SHARED_DATA_READ (one read address for the line array) against
# the delivered build, in lockstep: every output, every cycle, image slice
# parameters, with and without the cache data pipeline, two DCO policies.
SHARED_READ_SEEDS ?= 1 7 42
SHARED_READ_CYCLES ?= 200000
.PHONY: nuca-shared-read
nuca-shared-read: | dirs
	@for cfg in "1 1 0" "1 1 46" "1 0 0" "1 0 46" "2 1 0" "2 1 46" "2 0 0" "2 0 46"; do \
	   set -- $$cfg; \
	   $(IVERILOG) -g2012 $(IVFLAGS) -I $(RTL)/common -y $(RTL)/common \
	     -P tb_nuca_shared_read.SHARED_MODE=$$1 \
	     -P tb_nuca_shared_read.CACHE_DATA_PIPELINE=$$2 -P tb_nuca_shared_read.POLICY=$$3 \
	     -o $(BUILD)/tb_nuca_shared_read_$$1_$$2_$$3.vvp $(TB)/nuca/tb_nuca_shared_read.v \
	     $(RTL)/nuca/tnca_nuca_slice.v $(TB)/common/tnca_dram_model.v || exit 1; \
	   for sd in $(SHARED_READ_SEEDS); do \
	     $(TOOLS)/run_test.sh "NUCA-SHARED-READ-mode$$1-pipe$$2-policy$$3-seed$$sd" \
	       "$(VVP) $(BUILD)/tb_nuca_shared_read_$$1_$$2_$$3.vvp +seed=$$sd +cycles=$(SHARED_READ_CYCLES)" \
	       || exit 1; \
	   done; \
	 done

# The same randomised check under DCO's policies (doc/DCO.pdf; the POLICY
# register of rtl/nuca/tnca_tmu.v). The bench drives a bypassed tensor, a
# dead-tile FIFO that keeps changing and a flipping slow-core flag; whatever
# the cache keeps or skips, every read must still return the shadow.
#   1 LRU   2 anti-thrash   46 AT + DBP + dynamic and tensor bypass
#   62 the same with gqa_bypass   74 AT + a static bypass gear
DCO_POLICIES ?= 1 2 46 62 74
DCO_SEEDS    ?= 1 7

.PHONY: nuca-dco
nuca-dco: | dirs
	@for pol in $(DCO_POLICIES); do \
	   $(IVERILOG) $(IVFLAGS) -y $(RTL)/common \
	     -P tb_nuca_slice.DRAM_LAT=40 -P tb_nuca_slice.DRAM_STALL=1 \
	     -P tb_nuca_slice.DRAM_OOO=1 -P tb_nuca_slice.POLICY=$$pol \
	     -o $(BUILD)/tb_nuca_dco$$pol.vvp $(TB)/nuca/tb_nuca_slice.v \
	     $(RTL)/nuca/tnca_nuca_slice.v $(TB)/common/tnca_dram_model.v || exit 1; \
	   for sd in $(DCO_SEEDS); do \
	     $(TOOLS)/run_test.sh "DCO-NUCA-policy$$pol-seed$$sd" \
	       "$(VVP) $(BUILD)/tb_nuca_dco$$pol.vvp +nreq=$(NUCA_REQS) +min_wb=1 +seed=$$sd" \
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
#------------------------------------------------------------------------------
# DCO on the whole machine (tools/r38/dco_bench.py)
#------------------------------------------------------------------------------
# The 2x2 system with a last level small enough for a simulated kernel to
# thrash -- 8 sets x 4 ways per slice, 8 KiB in all -- and a DRAM that takes
# one line every DCO_DRAM_INTERVAL cycles, so traffic costs time. Triton
# kernels compiled with the DCO pass run under every policy; the bench checks
# that all policies give identical bits and reports cycles and DRAM traffic.
DCO_DRAM_INTERVAL ?= 48
DCO_DRAM_LAT      ?= 40
DCO_BENCH_ARGS    ?= --dram-interval $(DCO_DRAM_INTERVAL) \
                     --json $(BUILD)/r38/dco_bench_i$(DCO_DRAM_INTERVAL).json

$(BUILD)/tb_top_dco_i$(DCO_DRAM_INTERVAL).vvp: $(TB)/top/tb_tnca_top.v $(RTL_DEPS) $(RTL)/tnca_top.v \
                         $(TB)/common/tnca_dram_model.v | dirs
	$(IVERILOG) -g2012 $(IVFLAGS) $(RTLY) \
	   -P tb_tnca_top.SETS=8 -P tb_tnca_top.SET_W=3 \
	   -P tb_tnca_top.WAYS=4 -P tb_tnca_top.WAY_W=2 \
	   -P tb_tnca_top.DRAM_LINES=32768 -P tb_tnca_top.DRAM_STALL=0 \
	   -P tb_tnca_top.DRAM_OOO=1 -P tb_tnca_top.DRAM_LAT=$(DCO_DRAM_LAT) \
	   -P tb_tnca_top.DRAM_INTERVAL=$(DCO_DRAM_INTERVAL) \
	   -o $@ $< $(RTL)/tnca_top.v $(TB)/common/tnca_dram_model.v

.PHONY: dco-bench
dco-bench: $(BUILD)/tb_top_dco_i$(DCO_DRAM_INTERVAL).vvp
	@mkdir -p $(BUILD)/r38
	$(TRITON_PYTHON) tools/r38/dco_bench.py $(DCO_BENCH_ARGS)

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
                $(TOOLS)/isa/gen_target.py spec/r1_profiles.json
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

# DCO's compiler pass (tnca_cc/dco.py): nAcc, bypass and the tile granule for
# three kernels whose read counts are known, registration ahead of every load,
# and the refusal of a loop it cannot count. Compile-only.
.PHONY: dco-compiler
dco-compiler:
	@if [ -x "$(TRITON_PYTHON)" ]; then \
	   $(TRITON_PYTHON) compiler/test/run_suite.py test_dco dco DCO-COMPILER; \
	 elif [ "$(R1_STRICT)" = "1" ]; then \
	   echo "  dco                    FAIL   no Triton at $(TRITON_PYTHON)"; \
	   exit 1; \
	 else \
	   echo "  dco                    SKIP   no Triton at $(TRITON_PYTHON)"; \
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

# R38 S1-07: core-sim with the transposed tensor-DMA write (core-xpose-v1).
$(BUILD)/tb_core_xpose.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                            $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.HAS_TENSOR_XPOSE=1 \
	   -o $@ $(TB)/core/tb_tnca_core.v

# R41 X8: the transposed write for 16-bit and 1-byte elements.
$(BUILD)/tb_core_xpose8.vvp: $(RTL_DEPS) $(TB)/core/tb_tnca_core.v \
                            $(TB)/common/tnca_mem_model.v | dirs
	$(IVERILOG) $(IVFLAGS) -y $(RTL)/core -y $(RTL)/common -y $(RTL)/gen -y $(TB)/common \
	   -P tb_tnca_core.HAS_TENSOR_XPOSE=2 \
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

test: lint unit compiler core compiler-rtl triton-e2e core-slowmem core-ooo nuca nuca-dco mesh top e2e
	@echo
	@echo "=============================================="
	@echo " TNCA regression: all tests passed"
	@echo "=============================================="

#------------------------------------------------------------------------------
synth: | dirs
	$(PYTHON) flow/synth/synth.py --json $(BUILD)/synth.json --logs $(BUILD)/synth

# FPGA timing of a unit or the whole device, post-synthesis and out of context
# (flow/synth/vivado_ooc.sh): minutes where a v++ link is hours, to find what
# sets the board clock and to check that a fix moved it. Needs Vivado.
#   make fpga-timing FPGA_UNIT=tnca_matrix FPGA_PERIOD=13.333 FPGA_PARAMS="FMA_LAT=3"
FPGA_UNIT   ?= tnca_fpga_top
FPGA_PERIOD ?= 13.333
FPGA_PARAMS ?=
.PHONY: fpga-timing
fpga-timing:
	flow/synth/vivado_ooc.sh $(FPGA_UNIT) $(FPGA_PERIOD) $(FPGA_PARAMS)

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
        r36-export r36-rtl-tests r36-fpga-tests r36-mem-tests r36-conformance \
        r36-fidelity r36-mutation-tests r36-regression \
        r36-io-tests r36-build r36-build-tests \
        r36-handoff-tests r36-acceptance r36-verify-evidence \
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
                  $(wildcard $(TB)/common/*.vh) $(wildcard $(TB)/device/*.v) \
                  $(wildcard spec/*.json) $(wildcard spec/r40/*.json) $(wildcard compiler/targets/*.json) \
                  $(wildcard isa/*.json) \
                  tools/asm/tnca_asm.py tools/asm/tnca_isa.py
	@$(MAKE) --no-print-directory r2-stage-data

r2-stage-data:
	@rm -rf $(R2_DATA)
	@mkdir -p $(R2_DATA)/spec $(R2_DATA)/compiler/targets $(R2_DATA)/isa
	@cp spec/*.json $(R2_DATA)/spec/
	@mkdir -p $(R2_DATA)/spec/r38 && cp spec/r38/*.json $(R2_DATA)/spec/r38/
	@mkdir -p $(R2_DATA)/spec/r40 && cp spec/r40/*.json $(R2_DATA)/spec/r40/
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

# PKG-01 smoke: the wheel, installed non-editable into a fresh environment and
# imported with -I from an empty directory, reads only data it shipped -- every
# file it opens exists inside the environment (tools/r2/wheel_smoke.py). Fast,
# and needs no Triton: it is the check for a module that finds its data by
# counting directories up from itself.
.PHONY: r2-wheel-smoke
r2-wheel-smoke: r2-package
	@$(PYTHON) tools/r2/wheel_smoke.py

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
r36-check-plan: | dirs
	@$(TRITON_PYTHON) tools/r36/plan.py --json $(BUILD)/r36_plan.json

# Gate A: the backend boundary and the configuration rules.
r36-backend-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/backend_tests.py \
	   --json $(BUILD)/r36_backend_tests.json

r36-config-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/config_tests.py \
	   --json $(BUILD)/r36_config_tests.json

#------------------------------------------------------------------------------
# R3.7: TileBench coverage, from Triton source.
#
# Two numbers, because they answer different questions. `r37-coverage`
# says which of the forty-five operators BUILD; `r37-run` executes each
# one on the core bench and checks what it computed against a reference.
# A compiler that emits plausible instructions for all forty-five and
# the wrong answer for one has not supported forty-five operators, and
# the first target alone would not notice.
.PHONY: r37-aot-tests r37-coverage r37-run r37 r37-direct-coverage r37-direct-run r37-representative-coverage r37-representative-run
TILEBENCH_SOURCE ?= ../Tilebench
R37_OPT_LEVELS ?= 0 1 2
r37-aot-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/aot_tests.py

r37-coverage: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/aot_run.py --source $(TILEBENCH_SOURCE) \
	   --levels $(R37_OPT_LEVELS) --compile-only --json $(BUILD)/r37_aot_compile.json

r37-run: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/aot_run.py --source $(TILEBENCH_SOURCE) \
	   --levels $(R37_OPT_LEVELS) --json $(BUILD)/r37_aot_run.json

# Direct compiler/bench diagnostics remain available separately.
r37-direct-coverage: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/upstream.py --source $(TILEBENCH_SOURCE) \
	   --json $(BUILD)/r37_upstream.json

r37-direct-run: $(BUILD)/tb_core.vvp $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/upstream_run.py --source $(TILEBENCH_SOURCE) \
	   --json $(BUILD)/r37_upstream_run.json

r37-representative-coverage: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/coverage.py --json $(BUILD)/r37_coverage.json

r37-representative-run: $(BUILD)/tb_core.vvp $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r37/run.py --json $(BUILD)/r37_run.json

r37: r37-coverage r37-run

#------------------------------------------------------------------------------
# R38 Stage 1 (doc/r38_stage1_tasks_and_acceptance.md)
#
# Every target measures the working tree: benches are built from rtl/ into
# sim/build/r38/, so a candidate is measured as it stands rather than as the
# device cache last saw it. A target passing is not Stage 1 passing; the
# status document names which work package each one backs.
.PHONY: r38-l0 r38-smoke r38-baseline r38-baseline-check r38-profiler r38-l2 r38-xpose-rtl r38-torch r38-tdesc
# P1+ (doc/r38_pytorch_p1plus.md): PyTorch lives in its own environment so no
# existing gate gains a dependency. T1: the R3.7 cases through native Triton
# launch; T2-T4: small graphs through torch.compile -> Inductor -> TNCA.
TORCH_PYTHON ?= $(HOME)/src/torchenv/bin/python
# A compile-path change must leave every existing artifact byte-identical:
# identical bytes cannot change a kernel's behaviour or its cycles.
.PHONY: r38-identity
r38-identity: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r38/artifact_identity.py --profile core-sim --levels 0 1 2
# Tensor descriptors (rank 1-5) through the real Triton front end, on the RTL.
r38-tdesc: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r38/tdesc_tests.py

r38-torch: $(R2_DATA_STAMP) | dirs
	@for l in 0 1 2; do $(TORCH_PYTHON) tools/r38/plugin_identity.py --level $$l || exit 1; done
	@$(TORCH_PYTHON) tools/r38/triton_native_run.py
	@$(TORCH_PYTHON) tools/r38/torch_tests.py
	@TNCA_TENSOR_DESCRIPTORS=0 $(TORCH_PYTHON) tools/r38/torch_tests.py --json $(BUILD)/r38/torch/torch_tests_nodesc.json
# S1-07: the transposed transfer at the ISA level, on both elaborations.
XPOSE_PROGS := t30_xpose t31_xpose_illegal t32_xpose_badsize
$(TB)/core/progs/t30_xpose.s $(TB)/core/progs/t31_xpose_illegal.s \
$(TB)/core/progs/t32_xpose_badsize.s: $(TB)/core/progs/gen_t30_xpose.py
	@$(PYTHON) $<
r38-xpose-rtl: $(BUILD)/tb_core.vvp $(BUILD)/tb_core_xpose.vvp $(patsubst %,$(BUILD)/%.hex,$(XPOSE_PROGS))
	@$(TOOLS)/run_test.sh t30_xpose "$(VVP) $(BUILD)/tb_core_xpose.vvp +hex=$(BUILD)/t30_xpose.hex +maxcycles=100000"
	@$(TOOLS)/run_test.sh t31_xpose_illegal "$(VVP) $(BUILD)/tb_core.vvp +hex=$(BUILD)/t31_xpose_illegal.hex +maxcycles=100000 +expectfault=01"
	@$(TOOLS)/run_test.sh t32_xpose_badsize "$(VVP) $(BUILD)/tb_core_xpose.vvp +hex=$(BUILD)/t32_xpose_badsize.hex +maxcycles=100000 +expectfault=04"

XPOSE8_PROGS := t33_xpose8 t34_xpose8_old t35_xpose8_badrows t36_xpose8_badcol
$(TB)/core/progs/t33_xpose8.s $(TB)/core/progs/t34_xpose8_old.s \
$(TB)/core/progs/t35_xpose8_badrows.s $(TB)/core/progs/t36_xpose8_badcol.s: $(TB)/core/progs/gen_t30_xpose.py
	@$(PYTHON) $<
.PHONY: r41-xpose8-rtl
r41-xpose8-rtl: $(BUILD)/tb_core.vvp $(BUILD)/tb_core_xpose.vvp $(BUILD)/tb_core_xpose8.vvp \
                $(patsubst %,$(BUILD)/%.hex,$(XPOSE_PROGS) $(XPOSE8_PROGS))
	@$(TOOLS)/run_test.sh "t30_xpose (x8)" "$(VVP) $(BUILD)/tb_core_xpose8.vvp +hex=$(BUILD)/t30_xpose.hex +maxcycles=100000"
	@$(TOOLS)/run_test.sh "t32_xpose_badsize (x8)" "$(VVP) $(BUILD)/tb_core_xpose8.vvp +hex=$(BUILD)/t32_xpose_badsize.hex +maxcycles=100000 +expectfault=04"
	@$(TOOLS)/run_test.sh t33_xpose8 "$(VVP) $(BUILD)/tb_core_xpose8.vvp +hex=$(BUILD)/t33_xpose8.hex +maxcycles=100000"
	@$(TOOLS)/run_test.sh "t33_xpose8 (no xpose)" "$(VVP) $(BUILD)/tb_core.vvp +hex=$(BUILD)/t33_xpose8.hex +maxcycles=100000 +expectfault=01"
	@$(TOOLS)/run_test.sh t34_xpose8_old "$(VVP) $(BUILD)/tb_core_xpose.vvp +hex=$(BUILD)/t34_xpose8_old.hex +maxcycles=100000 +expectfault=04"
	@$(TOOLS)/run_test.sh t35_xpose8_badrows "$(VVP) $(BUILD)/tb_core_xpose8.vvp +hex=$(BUILD)/t35_xpose8_badrows.hex +maxcycles=100000 +expectfault=04"
	@$(TOOLS)/run_test.sh t36_xpose8_badcol "$(VVP) $(BUILD)/tb_core_xpose8.vvp +hex=$(BUILD)/t36_xpose8_badcol.hex +maxcycles=100000 +expectfault=04"

r38-l0: | dirs
	@$(PYTHON) tools/r38/gen_counters.py --check
	@$(PYTHON) tools/r38/evidence_tests.py
	@$(PYTHON) tools/r38/archbench_tests.py

r38-smoke: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r38/run.py --suite smoke
	@$(TRITON_PYTHON) tools/r38/manifest_tests.py

r38-baseline: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r38/baseline.py --run --source $(TILEBENCH_SOURCE)

r38-baseline-check: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r38/baseline.py --check --source $(TILEBENCH_SOURCE)

.PHONY: r40-baseline-check
r40-baseline-check: $(R2_DATA_STAMP) | dirs
	@TNCA_BANK_ALLOC=0 $(TRITON_PYTHON) tools/r38/baseline.py --check --source $(TILEBENCH_SOURCE) --reference spec/r40/b39/core_sim_cycles.json

r38-profiler: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r38/profiler_tests.py

r38-l2: r38-l0 r37 r38-baseline-check r38-profiler r38-smoke

# ArchBench at the PyTorch level (archbench/README.md, archbench/PROTOCOL.md):
# one set of torch workloads, TNCA through torch.compile -> Inductor -> tnca,
# the GPU through torch.compile on CUDA, one bound and one scorer. Runs in
# TORCH_PYTHON. Records are JSON lines under $(BUILD)/archbench/.
.PHONY: archbench-selftest archbench-coverage archbench-tiny archbench-u50sim archbench-gpu
ARCHBENCH_OUT ?= $(BUILD)/archbench
ARCHBENCH_JOBS ?= 8
archbench-selftest:
	@$(TORCH_PYTHON) -m archbench.selftest
# Every tiny and small point compiles for TNCA; nothing is simulated.
archbench-coverage: $(R2_DATA_STAMP) | dirs
	@mkdir -p $(ARCHBENCH_OUT) && rm -f $(ARCHBENCH_OUT)/coverage.jsonl
	@$(TORCH_PYTHON) -m archbench.run --platform tnca --tier tiny small --compile-only \
	   --jobs $(ARCHBENCH_JOBS) --out $(ARCHBENCH_OUT)/coverage.jsonl
# The tiny tier on the RTL simulator (hours: the models are millions of cycles).
archbench-tiny: $(R2_DATA_STAMP) | dirs
	@mkdir -p $(ARCHBENCH_OUT) && rm -f $(ARCHBENCH_OUT)/tnca_tiny.jsonl
	@$(TORCH_PYTHON) -m archbench.run --platform tnca --tier tiny --jobs $(ARCHBENCH_JOBS) \
	   --out $(ARCHBENCH_OUT)/tnca_tiny.jsonl
# The same through the board's register protocol on the board's RTL top: the
# expected board results (doc/r38_board_bringup.md, step B4).
archbench-u50sim: $(R2_DATA_STAMP) | dirs
	@mkdir -p $(ARCHBENCH_OUT) && rm -f $(ARCHBENCH_OUT)/u50sim_tiny.jsonl
	@$(TORCH_PYTHON) -m archbench.run --platform tnca --backend fpga-u50-sim --profile core-vrf-sync-v2 \
	   --tier tiny --jobs $(ARCHBENCH_JOBS) --out $(ARCHBENCH_OUT)/u50sim_tiny.jsonl
archbench-gpu:
	@mkdir -p $(ARCHBENCH_OUT) && rm -f $(ARCHBENCH_OUT)/gpu.jsonl
	@$(TORCH_PYTHON) -m archbench.run --platform gpu --tier tiny small large --out $(ARCHBENCH_OUT)/gpu.jsonl

# R40 (doc/r40_tasks_and_acceptance.md). Each gate's evidence is
# $(BUILD)/r40/<work package>.json, judged by tools/r40/acceptance.py.
# R40-00: the protocol, the frozen B39 baseline (spec/r40/baseline.json) and
# the plan, subsets and ablation matrix frozen before any R40 measurement.
.PHONY: r40-protocol r40-freeze r40-freeze-check r40-api-lock r40-hbm-coverage \
        r40-b39-gates r40-b39 r40-acceptance
# Acceptance 1 and 3: every manipulation of a comparison refused, identity 1;
# nothing missing passes a gate. Protocol/manifest checks need no torch or board.
r40-protocol:
	@$(PYTHON) tools/r40/contract_v2_tests.py
	@$(PYTHON) tools/r40/contract_v3.py
	@$(PYTHON) tools/r40/device_time_v2_tests.py
	@$(PYTHON) tools/r40/acceptance_runner_tests.py
	@$(PYTHON) tools/r40/audit_xo_sources_tests.py
	@$(PYTHON) tools/r40/attention_checks_tests.py
	@$(PYTHON) tools/r40/protocol_tests.py --evidence
# Write the plan, subsets and ablation matrix (an amendment once frozen).
r40-freeze:
	@$(TORCH_PYTHON) tools/r40/freeze.py
r40-api-lock:
	@$(TORCH_PYTHON) tools/r40/api_lock.py --out spec/r40/api_lock.json
# The large and hbm points through the standard path, compiled, none run:
# compile coverage and capacity, recorded in spec/r40/hbm_coverage.json.
R40_COVERAGE_PROFILE ?= core-vrf-sync-v5
r40-hbm-coverage: $(R2_DATA_STAMP) | dirs
	@mkdir -p $(BUILD)/r40 && rm -f $(BUILD)/r40/hbm_compile.jsonl
	@$(TORCH_PYTHON) -m archbench.run --platform tnca --profile $(R40_COVERAGE_PROFILE) --tier large hbm \
	   --compile-only --jobs $(ARCHBENCH_JOBS) --out $(BUILD)/r40/hbm_compile.jsonl || true
	@$(TORCH_PYTHON) tools/r40/hbm_coverage.py $(BUILD)/r40/hbm_compile.jsonl
# The plan, subsets, suites, API lock, B39 records and HBM coverage as frozen.
r40-freeze-check:
	@$(TORCH_PYTHON) tools/r40/freeze.py --check --evidence
# R39's merge gates on this tree, against the counts frozen with B39.
r40-b39-gates: $(R2_DATA_STAMP) | dirs
	@TORCH_PYTHON=$(TORCH_PYTHON) TRITON_PYTHON=$(TRITON_PYTHON) \
	   $(PYTHON) tools/r40/b39_gates.py --run --out $(BUILD)/r40/b39_gates_run.json; \
	 $(PYTHON) tools/r40/b39_gates.py --check $(BUILD)/r40/b39_gates_run.json --evidence
# Acceptance 2: B39 rerun on its image (B39_RERUN: the frozen commit's
# records, u280_<tier>.jsonl) and the R40-00 tree's protocol-2 run
# (B39_P2), with the gate run above.
B39_RERUN ?= $(BUILD)/r40/b39_rerun
B39_P2 ?= $(BUILD)/r40/b39_p2
r40-b39:
	@$(PYTHON) tools/r40/b39_repro.py --b39 $(B39_RERUN) --p2 $(B39_P2) \
	   --gates $(BUILD)/r40/b39_gates_run.json --evidence
r40-acceptance:
	@$(PYTHON) tools/r40/acceptance.py --json $(BUILD)/r40/acceptance.json

.PHONY: r40-v2-contract-check r40-v2-acceptance
r40-v2-contract-check:
	@$(PYTHON) tools/r40/contract_v2.py
r40-v2-acceptance: r40-v2-contract-check
	@$(PYTHON) tools/r40/acceptance.py --plan spec/r40/v2/plan.json --json $(BUILD)/r40/v2/acceptance.json
.PHONY: r40-v3-contract-check r40-v3-acceptance
# R40 v3: the one-time relaxation for delivery on the 90 MHz image
# (doc/r40_tasks_and_acceptance_v3.md). Gate evidence: sim/build/r40/v3, or
# unchanged capability evidence bound to v2 and the v3 delivery identity.
r40-v3-contract-check:
	@cd tools/r40 && $(PYTHON) contract_v3.py
R40_EVIDENCE_ROOT ?= $(BUILD)/r40
r40-v3-acceptance: r40-v3-contract-check
	@R40_CONTRACT=v3 $(PYTHON) tools/r40/acceptance.py --plan doc/r40_v3/plan.json \
	  --evidence-root $(R40_EVIDENCE_ROOT) --json $(BUILD)/r40/v3/acceptance.json
.PHONY: r40-device-time-v2
# Three complete interleaved board blocks per side; no legacy B39 evidence.
R40_V2_CONTROL ?= $(BUILD)/r40/v2/device/control0.jsonl $(BUILD)/r40/v2/device/control1.jsonl $(BUILD)/r40/v2/device/control2.jsonl
R40_V2_NATIVE ?= $(BUILD)/r40/v2/device/native0.jsonl $(BUILD)/r40/v2/device/native1.jsonl $(BUILD)/r40/v2/device/native2.jsonl
r40-device-time-v2: r40-v2-contract-check
	@$(PYTHON) tools/r40/device_time_v2.py --control $(R40_V2_CONTROL) --native $(R40_V2_NATIVE) \
	  --json $(BUILD)/r40/v2/device_time.json --evidence
# R40-01: the tnca PyTorch device (doc/r40_tnca_device.md). The device tests
# on this host -- run once on a host with CUDA and once on one without
# (acceptance 2) -- then the native suite's records (R40_NATIVE: the
# u280_<tier>.jsonl of `archbench.run --set TNCA_TORCH_DEVICE=native` on the
# B39 image) against B39 (acceptance 1 and 3).
R40_NATIVE ?= $(BUILD)/r40/native
.PHONY: r40-device
r40-device: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r40/properties_tests.py
	@$(TORCH_PYTHON) tools/r40/device_tests.py --evidence
ifeq ($(R40_CONTRACT),v2)
	@$(PYTHON) tools/r40/native_inventory.py --native $(R40_NATIVE) --evidence
	@$(MAKE) r40-device-time-v2
else
	@$(PYTHON) tools/r40/device_time.py --native $(R40_NATIVE) --evidence
endif
# R40-02 capability checkpoint. Frozen performance and asynchronous stress
# remain not-measured until their independent runners supply the evidence.
.PHONY: r40-residency
r40-residency: $(R2_DATA_STAMP) | dirs
	@$(PYTHON) tools/r40/child_exec_tests.py
	@$(PYTHON) tools/r40/io_concurrency_tests.py
	@$(TORCH_PYTHON) tools/r40/lifetime_tests.py
	@TNCA_BACKEND=sim-file $(TORCH_PYTHON) tools/r40/allocation_pressure_tests.py
	@$(TORCH_PYTHON) tools/r40/residency_tests.py --evidence
	@$(TORCH_PYTHON) tools/r40/device_copy_tests.py
	@$(TORCH_PYTHON) tools/r40/batch_tests.py --calls 1000
	@TNCA_BACKEND=sim-file $(TORCH_PYTHON) tools/r40/recovery_tests.py
	@TNCA_BACKEND=sim-file $(TORCH_PYTHON) tools/r40/replay_tests.py --replays 1000
	@$(TORCH_PYTHON) tools/r40/frozen_residency_tests.py --point op.attention_decode/tiny/fp16-b1-h2-l16-d16

# Full frozen transfer-correctness inventory. RTL runs can take days; keep
# each attempt in a fresh directory and run heavy suites on the jump host.
.PHONY: r40-frozen-residency
R40_RESIDENCY_OUT ?= $(BUILD)/r40/frozen_residency_suite
R40_RESIDENCY_JOBS ?= 2
r40-frozen-residency: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r40/frozen_residency_suite.py \
	   --jobs $(R40_RESIDENCY_JOBS) --out $(R40_RESIDENCY_OUT)
# Growing state is separate from fixed-input frozen transfer audits.
.PHONY: r40-decode-residency
R40_DECODE_STEPS ?= 64
R40_HOST_TIMEOUT ?= 172800
r40-decode-residency: $(R2_DATA_STAMP) | dirs
	@TNCA_PROFILE=core-vrf-sync-v5/hbm TNCA_HOST_TIMEOUT=$(R40_HOST_TIMEOUT) \
	   $(TORCH_PYTHON) tools/r40/decode_residency_tests.py --steps $(R40_DECODE_STEPS)
.PHONY: r40-bucket-decode
r40-bucket-decode: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r40/bucket_decode_tests.py
	@TNCA_PROFILE=core-vrf-sync-v5/hbm TNCA_HOST_TIMEOUT=$(R40_HOST_TIMEOUT) \
	   $(TORCH_PYTHON) tools/r40/decode_residency_tests.py --steps $(R40_DECODE_STEPS) --bucket
# R40-03 correctness checkpoint; frozen performance gates remain independent
# of these standard Torch register-block, reduction and GEMM checks.
.PHONY: r40-tiles
r40-tiles: $(R2_DATA_STAMP) | dirs
	@for width in 16 32 64 128; do $(TORCH_PYTHON) tools/r40/tiles_tests.py --width $$width || exit $$?; done
	@for width in 16 32 64 128; do $(TORCH_PYTHON) tools/r40/reduction_tests.py --width $$width || exit $$?; $(TORCH_PYTHON) tools/r40/fold_tests.py --width $$width || exit $$?; done
	@for width in 16 32 64; do $(TORCH_PYTHON) tools/r40/reduction_tests.py --width $$width --rows 2 || exit $$?; done
	@$(TORCH_PYTHON) tools/r40/fold_tests.py --width 32 --rows 2 --index-bias 1099511627776
	@for columns in 16 32 64; do $(TORCH_PYTHON) tools/r40/gemm_tiles_tests.py --columns $$columns || exit $$?; done
	@for columns in 16 32; do for dtype in float16 bfloat16; do TNCA_HOST_TIMEOUT=$(R40_HOST_TIMEOUT) $(TORCH_PYTHON) tools/r40/gemm_tiles_tests.py --columns $$columns --depth 32 --dtype $$dtype || exit $$?; done; done
	@TNCA_WINDOW_REUSE=1 TNCA_HOST_TIMEOUT=$(R40_HOST_TIMEOUT) $(TORCH_PYTHON) tools/r40/gemm_tiles_tests.py --columns 64 --depth 32
	@for dtype in float16 bfloat16; do TNCA_WINDOW_REUSE=1 TNCA_HOST_TIMEOUT=$(R40_HOST_TIMEOUT) $(TORCH_PYTHON) tools/r40/gemm_tiles_tests.py --columns 16 --depth 64 --dtype $$dtype || exit $$?; done
	@$(TRITON_PYTHON) tools/r40/window_options_tests.py
	@$(TRITON_PYTHON) tools/r40/compile_switch_tests.py
	@$(PYTHON) tools/r40/packed_window_options_tests.py
	@for mode in 0 1; do $(TORCH_PYTHON) tools/r40/bank_tests.py --enabled $$mode --out $(BUILD)/r40/bank_$$mode.json || exit $$?; done
	@TNCA_BANK_ALLOC=0 TNCA_WINDOW_REUSE=0 TNCA_PACKED_WINDOW_UNPACK=0 $(TRITON_PYTHON) tools/r40/contract_tests.py

.PHONY: r40-packed-window-unpack
r40-packed-window-unpack: $(R2_DATA_STAMP) | dirs
	@PYTHONPATH=compiler/python $(PYTHON) -m unittest discover -s compiler/test -p test_packed_window_unpack.py
	@$(PYTHON) tools/r40/packed_window_options_tests.py
	@$(TRITON_PYTHON) tools/r40/packed_window_snapshot_tests.py

.PHONY: r40-online-attention
r40-online-attention: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r40/sdpa_configuration_tests.py
	@$(TORCH_PYTHON) tools/r40/online_attention_tests.py
# R40-04: single-core HBM (rtl/fpga/tnca_axi_bridge.v). What runs without a
# board: the bridge against a reference under late, reordered, backpressured,
# refusing and reset AXI traffic, its mutations, and the tnca device on the
# simulated /hbm memory mode. The board's measurements are added by their own
# runs; until then the gates say not-measured.
.PHONY: r40-barrier-epoch
r40-barrier-epoch: | dirs
	@mkdir -p $(BUILD)/r40
	@$(IVERILOG) $(IVFLAGS) -s tb_barrier_epoch -o $(BUILD)/r40/barrier_epoch.vvp \
	  $(RTL)/core/tnca_barrier.v $(TB)/core/tb_barrier_epoch.v
	@$(VVP) $(BUILD)/r40/barrier_epoch.vvp
	@$(IVERILOG) $(IVFLAGS) -DEPOCH_EXHAUST -s tb_barrier_epoch \
	  -o $(BUILD)/r40/barrier_exhaust.vvp $(RTL)/core/tnca_barrier.v $(TB)/core/tb_barrier_epoch.v
	@$(VVP) $(BUILD)/r40/barrier_exhaust.vvp

.PHONY: r40-hbm
r40-hbm: $(R2_DATA_STAMP) r40-barrier-epoch | dirs
	@$(PYTHON) tools/r40/hbm_topology_tests.py
	@$(PYTHON) tools/r40/tensor_bounds_tests.py --log $(BUILD)/r40/tensor_bounds_formal.log
	@$(PYTHON) tools/r40/ram_cache_inference_tests.py
	@TORCH_PYTHON=$(TORCH_PYTHON) $(PYTHON) tools/r40/hbm_tests.py --evidence

.PHONY: r40-large-grid
r40-large-grid: $(R2_DATA_STAMP) | dirs
	@$(PYTHON) tools/r40/large_grid_tests.py
	@$(TRITON_PYTHON) tools/r40/large_grid_rtl_tests.py

.PHONY: r40-core-parameters
r40-core-parameters: | dirs
	@$(PYTHON) tools/r40/core_parameter_tests.py

.PHONY: r40-device-clock
r40-device-clock: $(R2_DATA_STAMP) | dirs
	@$(TORCH_PYTHON) tools/r40/device_clock_tests.py
	@$(TORCH_PYTHON) tools/r40/host_copy_tests.py
	@$(PYTHON) tools/r36/fpga_tests.py --only R36-FPGA-08
	@$(TORCH_PYTHON) tools/r40/device_clock_native_tests.py

# One suite, every backend. This is the suite a downstream branch runs.
r36-conformance: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/conformance.py \
	   --json $(BUILD)/r36_conformance.json

# Six directed mutations, each in an isolated copy, each required to be
# caught by the checker named for it.
r36-mutation-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/mutation_tests.py \
	   --json $(BUILD)/r36_mutation_tests.json

# The frozen impact scope, run rather than cited. Long: it is the whole
# inherited regression plus the fidelity measurement.
r36-regression: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/regression.py \
	   --json $(BUILD)/r36_regression.json

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

# The board wrapper, which the export walk cannot reach: rtl/fpga sits
# ABOVE tnca_shell rather than below it, so r36-rtl-tests never saw it
# and neither did `make lint` until RTL_FPGA was added. Elaboration, then
# `proc; check -assert`, then memory inference, each module elaborated at
# the root. That last detail is not a style choice: `hierarchy` renames a
# parameterised module, so a `select` on the original matches nothing and
# exits zero, and a gate written that way reports the same thing whether
# or not it ran. The script's header says more.
r36-fpga-tests: | dirs
	@$(PYTHON) tools/r36/fpga_tests.py --json $(BUILD)/r36_fpga_tests.json

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

# The gate, and the independent verification of what it wrote.
R36_EVIDENCE ?= $(BUILD)/r36-evidence

r36-acceptance: | dirs
	@$(PYTHON) tools/r36/acceptance.py --evidence $(R36_EVIDENCE) \
	   --python $(TRITON_PYTHON)

r36-verify-evidence:
	@$(PYTHON) tools/r36/verify_evidence.py \
	   --evidence $(or $(EVIDENCE),$(R36_EVIDENCE)) \
	   --json $(BUILD)/r36_verify.json

# The downstream rehearsal, run from outside the repository.
r36-handoff-tests: $(R2_DATA_STAMP) | dirs
	@$(TRITON_PYTHON) tools/r36/handoff_tests.py \
	   --json $(BUILD)/r36_handoff_tests.json

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

# All FOUR benches, one per profile in spec/r1_profiles.json. CFG-01
# reads every capability and resource back out of each profile's own
# top, so a bench it cannot find is a profile it cannot check.
#
# tb_core_vrf.vvp was missing from this list. The core-vrf-sync-v1
# profile arrived with R3 v2 and the rule was never updated, so the
# suite passed whenever something else had already built that bench --
# which is almost always, on a developer's machine and in CI, because
# the CI workflow builds it by name. In a clean run where nothing had,
# CFG-01 reported "sim/build/tb_core_vrf.vvp not built" and failed.
# R3.6's regression, which claims to reproduce the inherited suites
# from a clean checkout, is what found it.
r1-profile-tests: $(BUILD)/tb_core.vvp $(BUILD)/tb_core4.vvp \
                  $(BUILD)/tb_top.vvp $(BUILD)/tb_core_vrf.vvp $(BUILD)/tb_core_xpose.vvp \
                  $(BUILD)/tb_core_r39hw.vvp $(BUILD)/tb_core_r39sw.vvp \
                  $(BUILD)/tb_core_r39sw4.vvp $(BUILD)/tb_core_r39sw5.vvp
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

# Scheduler protocol and real-core execution diagnostics. Native Torch and
# routed multi-SLR performance remain separate R40-05 acceptance requirements.
.PHONY: r40-grid-dispatch
r40-grid-dispatch: | dirs
	@$(PYTHON) tools/r40/grid_dispatch_tests.py

.PHONY: r40-coherent-host
r40-coherent-host: | dirs
	@$(PYTHON) tools/r40/coherent_host_tests.py

.PHONY: r40-multicore-shell
r40-multicore-shell: r40-grid-deadline | dirs
	@$(PYTHON) tools/r40/multicore_shell_tests.py

.PHONY: r40-grid-deadline
r40-grid-deadline: | dirs
	@mkdir -p $(BUILD)/r40
	@$(IVERILOG) $(IVFLAGS) -s tb_grid_deadline -o $(BUILD)/r40/grid_deadline.vvp \
	  $(RTL)/tnca_grid_deadline.v $(TB)/tb_grid_deadline.v
	@$(VVP) $(BUILD)/r40/grid_deadline.vvp

# Mailbox protocol diagnostic. Standard Torch multicore acceptance is separate.
.PHONY: r40-physical-grid-transport
r40-physical-grid-transport: | dirs
	@$(TRITON_PYTHON) tools/r40/physical_grid_option_tests.py
	@$(TRITON_PYTHON) tools/r40/multicore_transport_tests.py
	@$(TRITON_PYTHON) tools/r40/large_grid_tests.py

.PHONY: r40-physical-grid-runtime
r40-physical-grid-runtime: | dirs
	@$(TRITON_PYTHON) tools/r40/physical_grid_runtime_tests.py

# Real native Torch smoke; this is not the complete board/performance gate.
.PHONY: r40-multicore-backend r40-multicore-native-smoke
r40-multicore-backend: | dirs
	@$(TRITON_PYTHON) tools/r40/physical_grid_runtime_tests.py --automatic --cores 3
	@$(TRITON_PYTHON) tools/r36/conformance.py --backends sim-multicore --require-measured

r40-multicore-native-smoke: | dirs
	@$(TRITON_PYTHON) tools/r40/native_multicore_tests.py --cores 3
	@$(TRITON_PYTHON) tools/r40/native_multicore_tests.py --cores 8

.PHONY: r40-physical-chunks
r40-physical-chunks: r40-grid-dispatch | dirs
	@$(PYTHON) tools/r40/physical_chunk_tests.py
	@$(TRITON_PYTHON) tools/r40/physical_chunk_runtime_tests.py --cores 3
	@TNCA_RCTX_SPMD=1 TNCA_PHYSICAL_CHUNKS=1 $(TRITON_PYTHON) tools/r40/native_multicore_tests.py --cores 3
	@TNCA_RCTX_SPMD=1 TNCA_PHYSICAL_CHUNKS=1 $(TRITON_PYTHON) tools/r40/native_multicore_tests.py --cores 8
	@$(TRITON_PYTHON) tools/r40/native_multicore_features.py --cores 3
	@$(TRITON_PYTHON) tools/r40/native_multicore_features.py --cores 3 --pair

# Multi-controller functionality; HBM bandwidth/locality acceptance is separate.
.PHONY: r40-memory-ports
r40-memory-ports: | dirs
	@$(TRITON_PYTHON) tools/r40/memory_port_option_tests.py
	@$(PYTHON) tools/r40/memory_ports_tests.py
	@$(TRITON_PYTHON) tools/r40/native_memory_ports_tests.py --cores 3 --ports 2
	@$(TRITON_PYTHON) tools/r40/native_memory_ports_tests.py --cores 3 --ports 3

# Full-line AXI plumbing; board placement and performance remain separate gates.
.PHONY: r40-line-axi
r40-line-axi: | dirs
	@$(PYTHON) tools/r40/line_axi_tests.py
	@$(PYTHON) tools/r40/multicore_axi_tests.py
	@$(PYTHON) tools/r40/multicore_axi_tests.py --cores 3 --write-delay 300

# Versioned register path; shared board host binding driven by the AXI RTL.
.PHONY: r40-multicore-mmio r40-multicore-mmio-checks
r40-multicore-mmio-checks: | dirs
	@$(PYTHON) tools/r40/runner_source_tests.py
	@$(PYTHON) tools/r40/operating_clock_tests.py
	@$(PYTHON) tools/r40/storage_owner_tests.py
	@$(PYTHON) tools/r40/copy_range_tests.py
	@$(PYTHON) tools/r40/reserved_memory_tests.py
	@$(PYTHON) tools/r40/board_ablation_tests.py
	@$(PYTHON) tools/r40/board_ablation_acceptance_tests.py
	@$(PYTHON) tools/r40/device_conformance_acceptance_tests.py
	@$(PYTHON) tools/r40/residency_acceptance_tests.py
	@$(PYTHON) tools/r40/conditional_delivery_tests.py
	@$(PYTHON) tools/r40/multicore_board_acceptance_tests.py
	@$(PYTHON) tools/r40/hbm_probe_storage_tests.py
	@$(PYTHON) tools/r40/physical_grid_clock_tests.py
	@$(PYTHON) tools/r40/autotune_policy_tests.py
	@$(PYTHON) tools/r40/autotune_worker_tests.py
	@$(TRITON_PYTHON) tools/r40/code_cache_epoch_tests.py
	@$(TRITON_PYTHON) tools/r40/bucket_decode_tests.py
	@$(PYTHON) tools/r40/code_link_tests.py
	@$(PYTHON) tools/r40/mesh_pipeline_tests.py
	@$(PYTHON) tools/r40/multicore_mmio_synth_tests.py
	@$(PYTHON) tools/r40/placed_core_audit_tests.py
	@$(PYTHON) tools/r40/multicore_mmio_option_tests.py

# Keep the complete local entry point; CI schedules these independent lanes separately.
r40-multicore-mmio: r40-multicore-mmio-checks
	@$(TRITON_PYTHON) tools/r40/multicore_mmio_tests.py --cores 1 --ports 1
	@$(TRITON_PYTHON) tools/r40/multicore_mmio_tests.py --cores 3 --ports 3
	@$(TRITON_PYTHON) tools/r40/multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top
	@$(TRITON_PYTHON) tools/r40/native_memory_ports_tests.py --backend fpga-multicore-sim --cores 3 --ports 3 --pipeline 2 --board-top
	@$(TRITON_PYTHON) tools/r40/native_multicore_features.py --backend fpga-multicore-sim --cores 3 --pipeline 2 --board-top
	@$(TRITON_PYTHON) tools/r40/native_multicore_features.py --backend fpga-multicore-sim --cores 3 --pair --pipeline 2 --board-top

# Formal package entry points: functionality alone never signs off a package.
# The evidence judge also requires physical placement, board measurements and
# every performance gate from the frozen plan. Missing evidence exits nonzero.
.PHONY: r40-multicore r40-nuca r40-pipeline r40-pipeline-checks r40-cache-maintenance r40-multicore-dma
r40-multicore: r40-core-parameters r40-multicore-shell \
  r40-physical-grid-transport r40-multicore-backend \
  r40-multicore-native-smoke r40-physical-chunks r40-multicore-mmio
	@$(PYTHON) tools/r40/acceptance.py --wp R40-05

r40-cache-maintenance:
	@$(PYTHON) tools/r40/cache_maintenance_tests.py --global-grid

r40-multicore-dma:
	@$(PYTHON) tools/r40/multicore_mmio_option_tests.py
	@$(PYTHON) tools/r40/dma_handoff_guard_tests.py
	@mkdir -p $(BUILD)/r40
	@for enabled in 0 1; do $(IVERILOG) $(IVFLAGS) -s tb_mmio_cache_handoff \
	  -Ptb_mmio_cache_handoff.ENABLE=$$enabled -o $(BUILD)/r40/mmio_handoff$$enabled.vvp \
	  $(RTL)/fpga/tnca_multicore_mmio.v $(TB)/fpga/tb_mmio_cache_handoff.v || exit $$?; \
	  $(VVP) $(BUILD)/r40/mmio_handoff$$enabled.vvp || exit $$?; done
	@$(PYTHON) tools/r40/multicore_dma_tests.py --cores 1 --ports 1 --pipeline 0
	@$(PYTHON) tools/r40/multicore_dma_tests.py --cores 3 --ports 3 --pipeline 2 --board-top
	@CUDA_VISIBLE_DEVICES= $(TORCH_PYTHON) tools/r40/native_memory_ports_tests.py \
	  --backend fpga-multicore-sim --cores 3 --ports 3 --pipeline 2 --board-top --dma
	@CUDA_VISIBLE_DEVICES= $(TORCH_PYTHON) tools/r40/recovery_tests.py --multicore-dma

r40-nuca: r40-coherent-host r40-memory-ports r40-line-axi r40-cache-maintenance r40-multicore-dma
	@$(PYTHON) tools/r40/acceptance.py --wp R40-06

# Ring tests are mechanism diagnostics. Standard Torch overlap, frozen
# long-K/matrix A/B and policy measurements must supply the package evidence.
r40-pipeline-checks: $(R2_DATA_STAMP) r40-barrier-epoch
	@$(PYTHON) tools/r40/engine_overlap_tests.py
	@$(TRITON_PYTHON) tools/r40/pipeline_stage_options_tests.py
	@$(TRITON_PYTHON) -c "from tnca import device; device.prepare('core-vrf-sync-v5')"
	@for stages in 2 3; do CUDA_VISIBLE_DEVICES= TNCA_RCTX_STAGES=$$stages \
	  $(TRITON_PYTHON) tools/r39/rctx_roles_tests.py || exit $$?; \
	  r40_trace_dir=$$(mktemp -d $(BUILD)/r40/pipeline$$stages.XXXXXX) || exit $$?; \
	  CUDA_VISIBLE_DEVICES= $(TORCH_PYTHON) tools/r40/native_pipeline_tests.py \
	    --stages $$stages --trace-dir "$$r40_trace_dir" --require-overlap || exit $$?; done

r40-pipeline: r40-pipeline-checks
	@$(PYTHON) tools/r40/acceptance.py --wp R40-07

# Functional tuning checks; frozen holdout/model evidence is still required.
.PHONY: r40-autotune-checks r40-autotune
r40-autotune-checks:
	@$(PYTHON) tools/r40/autotune_policy_tests.py
	@$(PYTHON) tools/r40/autotune_worker_tests.py
	@$(PYTHON) tools/r40/physical_grid_clock_tests.py
	@CUDA_VISIBLE_DEVICES= $(TORCH_PYTHON) tools/r40/autotune_native_tests.py

r40-autotune: r40-autotune-checks
	@$(PYTHON) tools/r40/acceptance.py --wp R40-09

.PHONY: r40-attention-checks r40-attention
R40_ATTENTION_PROFILE ?= core-vrf-sync-v5/hbm
r40-attention-checks: $(R2_DATA_STAMP) | dirs
	@r40_attention_dir=$$(mktemp -d $(BUILD)/r40/attention.XXXXXX) || exit $$?; \
	  CUDA_VISIBLE_DEVICES= $(TORCH_PYTHON) tools/r40/attention_checks.py \
	    --profile $(R40_ATTENTION_PROFILE) --out "$$r40_attention_dir/results"

r40-attention: r40-attention-checks
	@$(PYTHON) tools/r40/acceptance.py --wp R40-08

# R41 Stage 2: integrity-protected manual judgments; nonzero until all pass.
.PHONY: r41-stage2-acceptance r41-acceptance
r41-stage2-acceptance:
	$(PYTHON) tools/performance_harness/stage2_acceptance.py --root $(CURDIR) --json $(CURDIR)/sim/build/r41/stage2-acceptance.json
r41-acceptance:
	$(PYTHON) tools/performance_harness/stage2_acceptance.py --root $(CURDIR) --overall --json $(CURDIR)/sim/build/r41/acceptance.json
