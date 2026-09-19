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

PARAM_RE = re.compile(r"^\s*parameter\s+(\w+)\s*=\s*([0-9]+)\s*,?", re.M)


def params(path):
    """The default parameter values of the first module in `path`."""
    return {m.group(1): int(m.group(2)) for m in PARAM_RE.finditer(open(path).read())}


# Facts the RTL does not carry as a number, each with its normative source.
DESIGN_FACTS = {
    'tensor_max_rank': (2, 'rtl/core/tnca_core.v TNCA_SPEC_MAX_RANK; the '
                           'transfer engine implements rank 1-2 only'),
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


def main():
    core = params(CORE_V)
    top = params(TOP_V)
    with open(SCHEMA) as f:
        isa = json.load(f)

    # Capability bits, from the ISA names and this build's parameters.
    # A bit is set only where the decoder actually executes the encoding;
    # ISA sec 11 makes a clear bit a promise of ILLEGAL_INSTRUCTION.
    caps = {
        'vector_fp32': True,
        'sfu': bool(core['HAS_SFU']),
        'gather_scatter': True,
        'atomics_device': False,
        'atomics_system': False,
        'matrix_f16': True,
        'matrix_bf16': True,
        'tensor_dma': True,
        'mbarrier': True,
        'packed_f16': False,
        'cache_hints': False,
        # Not a CAPABILITY0 bit: no fence encoding exists in v0.1 at all.
        'fence': False,
    }

    target = {
        'schema_version': 1,
        'name': 'tnca-sim-v0.1',
        'description': ('The configuration tb/top/tb_tnca_top.v elaborates: '
                        'four tiles in a 2x2 mesh, two Role Contexts per core.'),
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
            'addressable_scratchpad': False,
        },

        'trap_codes': isa['trap_codes'],
        'design_fact_sources': {k: v[1] for k, v in DESIGN_FACTS.items()},
    }

    text = json.dumps(target, indent=2) + '\n'
    if '--check' in sys.argv:
        cur = open(OUT).read() if os.path.exists(OUT) else None
        if cur != text:
            print(f"FAIL: {os.path.relpath(OUT, ROOT)} is stale; run `make target`")
            return 1
        print(f"PASS: {os.path.relpath(OUT, ROOT)} matches the RTL parameters")
        return 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    print(f"  {target['num_rctx']} RCTX x {target['num_cores']} cores, "
          f"VLEN={target['vlen']}, "
          f"{target['registers']['vreg_physical_total']} physical vregs, "
          f"{target['accumulators']['count']} accumulators, "
          f"{target['barriers']['count']} barriers")
    off = [k for k, v in caps.items() if not v]
    print(f"  capabilities off: {', '.join(off)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
