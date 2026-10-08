#!/usr/bin/env python3
"""
gen_target.py -- generate the TNCA target description the compiler consumes.

compiler_overview.md sec 15, milestone C0 requires a target configuration
covering at least: RCTX count, VLEN, physical/logical V register limits,
accumulator count, DMA rank/dtype/layout capabilities, MMA throughput and
command constraints, barrier count and count width, source-release capability,
and SFU/atomics/fence capabilities.

Those facts already exist -- as RTL parameters and as the ISA schema. Typing
them a second time into the compiler would recreate exactly the drift sec 11.3
forbids for opcodes, so this script reads them out of the design instead:

  rtl/core/tnca_core.v      parameters -> sizes and latencies
  rtl/tnca_top.v            parameters -> mesh and cache geometry
  isa/tnca-isa-v0.1.json    -> capability names, trap codes, register limits

Anything genuinely not derivable is listed in DESIGN_FACTS below with the
section of the ISA or architecture document that fixes it, so a reader can
check it rather than trust it.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CORE_V = os.path.join(ROOT, 'rtl', 'core', 'tnca_core.v')
TOP_V = os.path.join(ROOT, 'rtl', 'tnca_top.v')
SCHEMA = os.path.join(ROOT, 'isa', 'tnca-isa-v0.1.json')
OUT = os.path.join(ROOT, 'compiler', 'targets', 'tnca-sim-v0.1.json')
PROFILES = os.path.join(ROOT, 'spec', 'r1_profiles.json')

# Additional elaborations of the same core. The parameters here are exactly
# what the corresponding bench passes with -P, so a variant target describes a
# machine that can actually be built rather than one that was imagined.
# compiler_overview.md sec 7.6 is the reason these exist: the compiler must
# map one graph onto different Role Context counts, which it cannot be shown
# to do against a single target.
# The profiles come from spec/r1_profiles.json, which is also what the
# elaboration parameters and the test commands come from. A target file that
# describes a module's *defaults* rather than an *elaborated instance* is how
# the audit's F03 happened: the four-tile target advertised an SFU that
# rtl/tile/tnca_tile.v had switched off.
with open(PROFILES) as _fh:
    PROFILE_SPEC = json.load(_fh)

PARAM_RE = re.compile(r"^\s*parameter\s+(\w+)\s*=\s*([0-9]+)\s*,?", re.M)


def params(path):
    """The default parameter values of the first module in `path`."""
    return {m.group(1): int(m.group(2)) for m in PARAM_RE.finditer(open(path).read())}


# Facts the RTL does not carry as a number, each with its normative source.
DESIGN_FACTS = {
    'tensor_max_rank': (5, 'rtl/core/tnca_core.v TNCA_SPEC_MAX_RANK; the '
                           'transfer engine implements rank 1-5 (ISA sec 9E), '
                           'signed coordinates, zero/NaN/fault padding'),
    'mma_shape': ({'m': 16, 'n': 16, 'k': 16},
                  'ISA sec 9C: m.mma.async is a 16x16x16 step'),
    'mma_accum_order': ('increasing_k',
                        'ISA sec 9C.1: RNE_f32(exact(A)*exact(B) + C), '
                        'increasing k, bit-reproducible'),
    'gather_index_scaling': ('element',
                             'ISA sec 9.1: lane i => Sa + sext(Vb[i]) * sz'),
    'descriptor_bytes': (256, 'ISA sec 9D.3'),
    'descriptor_semantic_bytes': (128,
                                  'ISA sec 9D.3: bytes 0-127 are semantics, '
                                  '128-255 are ignorable hints'),
}


def build_target(core, top, isa, name, description, profile=None):

    # Capability bits, from the ISA names and this INSTANCE's parameters --
    # `core` has already had the profile's elaboration overrides applied, so
    # `bool(core['HAS_SFU'])` is what the tiles are actually built with.
    # ISA sec 11 makes a clear bit a promise of ILLEGAL_INSTRUCTION, so a bit
    # set here that the hardware does not honour is a lie the compiler acts on.
    caps = {
        'vector_fp32': True,
        'sfu': bool(core['HAS_SFU']),
        'gather_scatter': True,
        'atomics_device': bool(core['HAS_ATOMICS']),
        'atomics_system': bool(core['HAS_SYS_ATOMICS']),
        'matrix_f16': True,
        'matrix_bf16': True,
        'matrix_fp8': True,
        'tensor_dma': True,
        # `t.st.async` (ISA sec 9E func 0x08): window -> memory, rank 1-5,
        # out-of-bounds elements not written. The engine reads the window
        # through its own VRF read port (tnca_tensor vr_*). Every engine has
        # it, so it is baseline rather than a CAPABILITY0 bit.
        'tensor_store': True,
        # ISA sec 9E's mod[5] asks the engine for a transposed write.
        # R38 S1-07 implemented it in tnca_tensor.v behind the core's
        # HAS_TENSOR_XPOSE parameter; an engine built without it raises
        # ILLEGAL_INSTRUCTION for mod[5] (it used to transfer the tile
        # untransposed). So the capability is this instance's parameter,
        # and on a profile without it asking remains a compile error.
        'tensor_transpose': bool(core.get('HAS_TENSOR_XPOSE', 0)),
        'mbarrier': True,
        'packed_f16': False,
        'cache_hints': False,
        # `fence` is ATOM 0x20 and has no CAPABILITY0 bit, which makes
        # it baseline rather than absent -- the same thing `s.add` is.
        # It was False here with a comment saying no encoding existed,
        # while tools/asm/tnca_isa.py held the encoding, tnca_core.v
        # decoded it, t14_caps.s and t22_atomic.s executed it, and the
        # generated support matrix recorded it as machine-verified. The
        # ATOM family gate is the one that applies.
        'fence': True,
    }

    target = {
        'schema_version': 1,
        'name': name,
        'description': description,
        'note': ('Generated file -- do not edit. Run `make target`. Derived '
                 'from rtl/core/tnca_core.v, rtl/tnca_top.v and '
                 'isa/tnca-isa-v0.1.json; see compiler_overview.md sec 15 C0.'),
        'isa_version': isa['isa_version'],

        # ---- execution contexts ----
        'num_rctx': core['NUM_RCTX'],
        'num_cores': top['NX'] * top['NY'],
        'mesh': {'x': top['NX'], 'y': top['NY']},

        # ---- vector shape ----
        'vlen': core['VLEN'],
        'elen_bits': core['ELEN'],

        # ---- register limits ----
        # logical: what an instruction field can name for one RCTX.
        # physical: what the machine actually has. They differ, and the
        # difference is what makes a shared RF window possible at all.
        'registers': {
            'vreg_logical_per_rctx': core['VREGS_PER_RCTX'],
            'vreg_physical_total': core['PHYS_VREGS'],
            'vreg_encodable': isa['register_files']['vreg']['count'],
            'sreg_per_rctx': isa['register_files']['sreg']['count'],
            'preg_per_rctx': isa['register_files']['preg']['count'],
        },

        # ---- shared resources: allocated for the whole program, not per role
        # (compiler_overview.md sec 10.1) ----
        'accumulators': {
            'count': core['ACC_N'],
            'rows': core['ACC_ROWS'],
            'element': 'f32',
        },
        'barriers': {
            'count': core['BAR_N'],
            'index_width': core['BAR_W'],
            # sec 13.1: arrivals and operations are counted separately.
            'counts': 'arrivals_and_operations_separately',
            'operation_unit': 'one_complete_async_command',
        },
        'descriptors': {'count': core['DESC_N']},
        'windows': {'count': core['WIN_N']},

        # ---- asynchronous engines ----
        'dma': {
            'max_rank': DESIGN_FACTS['tensor_max_rank'][0],
            'dtypes': ['f16', 'bf16'],
            'descriptor_bytes': DESIGN_FACTS['descriptor_bytes'][0],
            'descriptor_semantic_bytes':
                DESIGN_FACTS['descriptor_semantic_bytes'][0],
            'padding_modes': ['zero', 'clamp', 'fault'],
            'destination': 'rf_window',
            'note': ('A window names vector registers and nothing else '
                     '(ISA sec 9D.2); it is not a scratchpad and has no byte '
                     'pointer, so a transfer target must be a register range.'),
        },
        'mma': {
            'shape': DESIGN_FACTS['mma_shape'][0],
            'dtypes': isa['mma_dtype'],
            'k_steps': isa['mma_k'],
            'accumulate': DESIGN_FACTS['mma_accum_order'][0],
            'fused': True,
            'bit_reproducible': True,
            # sec 10.2: an async lifetime ends at a completion event, and the
            # source registers are NOT reusable at the same moment as the
            # destination. v0.1 has one completion, covering both.
            'source_release': 'on_command_completion',
            'separate_source_completion': False,
        },

        # ---- capability gates (compiler_overview.md sec 12) ----
        'capabilities': caps,

        # ---- issue and latency, for the cost model of sec 10.4 ----
        'latency': {
            'fp_result_cycles': core['FP_LAT'],
            'mul_result_cycles': core['MUL_LAT'],
        },
        'issue': {
            'width': 1,
            'note': 'Single issue; sec 10.4 accumulates cost per shared unit.',
        },

        # ---- memory ----
        'memory': {
            'line_bytes': top['LINE_BYTES'],
            'nuca': {
                'sets': top['SETS'], 'ways': top['WAYS'],
                'mshrs': top['MSHR_N'], 'targets': top['TARGET_N'],
                'placement': 'static_home',
                'note': ('home = H(physical line address); no replication and '
                         'no migration, so there is no inter-slice coherence '
                         'protocol to model.'),
            },
            'gather_index_scaling': DESIGN_FACTS['gather_index_scaling'][0],
            # Each lane's index is one 32-bit element, sign-extended then
            # scaled, so the reach is +/- 2^31 elements from the scalar base.
            # sec 12: a gather whose offsets exceed that needs rebasing,
            # grouping or a per-lane scalar fallback -- not a silent wrap.
            'gather_index_bits': 32,
            'gather_index_signed': True,
            'addressable_scratchpad': False,
        },

        # sec 12: "uniform FP32 标量 -- 不假定 Scalar ALU 有浮点能力". It does
        # not: all 36 SALU encodings are integer. A uniform FP32 value moves
        # through the vector domain or as a bit pattern in a scalar register,
        # and a lowering that assumes otherwise emits nothing that exists.
        'scalar': {
            'width_bits': 64,
            'has_floating_point': False,
            'note': ('SALU is integer only; move uniform FP through the '
                     'vector domain or carry it as a bit pattern.'),
        },

        'trap_codes': isa['trap_codes'],
        'design_fact_sources': {k: v[1] for k, v in DESIGN_FACTS.items()},
    }
    return target


def main():
    core = params(CORE_V)
    top = params(TOP_V)
    with open(SCHEMA) as f:
        isa = json.load(f)

    outputs = []
    for pname, spec in PROFILE_SPEC['profiles'].items():
        inst = dict(core)
        inst.update(spec['elaboration'])
        t = build_target(inst, top, isa, os.path.splitext(spec['target_file'])[0],
                         spec['description'], profile=pname)
        # The instance's own shape, not the top module's defaults. A
        # single-core profile that reported four cores is the same class of
        # error as an SFU that is not there.
        t['profile'] = pname
        t['num_cores'] = spec['num_cores']
        t['topology'] = spec['topology']
        t['elaboration'] = spec['elaboration']
        # The bench is deliberately NOT copied here. Phase 3.6 section
        # 3.1: a compile target describes the ISA, the ABI, the
        # capabilities and the resources, and may not be equated with a
        # simulator executable. It used to carry
        # `sim/build/tb_core_vrf.vvp`, which the compiler never read and
        # which made the target look like a binding to one program.
        if spec['num_cores'] == 1:
            t['memory'].pop('mesh', None)
        outputs.append((os.path.join(os.path.dirname(OUT), spec['target_file']), t))

    check = '--check' in sys.argv
    stale = []
    for path, target in outputs:
        text = json.dumps(target, indent=2) + '\n'
        if check:
            cur = open(path).read() if os.path.exists(path) else None
            if cur != text:
                stale.append(os.path.relpath(path, ROOT))
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').write(text)
        print(f"wrote {os.path.relpath(path, ROOT)}: "
              f"{target['num_rctx']} RCTX, "
              f"{target['registers']['vreg_logical_per_rctx']} logical vregs "
              f"each, {target['registers']['vreg_physical_total']} physical")
    if check:
        if stale:
            print(f"FAIL: stale: {', '.join(stale)}; run `make target`")
            return 1
        print("PASS: target descriptions match the RTL parameters")
    return 0


if __name__ == '__main__':
    sys.exit(main())
