#!/usr/bin/env python3
"""
export_schema.py -- emit the canonical machine-readable TNCA ISA schema.

doc/compiler_overview.md sec 11.3 requires ONE source of opcode numbers, feeding
the assembler/disassembler tables, the compiler's opcode definitions, encoding
legality, later TableGen input, and conformance tests -- explicitly not a third
hand-maintained set of numbers for the compiler.

The authoring form stays tools/asm/tnca_isa.py, which the assembler, the
disassembler and the architectural model already read and which the RTL test
suite exercises. This script serialises it into isa/tnca-isa-v0.1.json, the
form every non-Python consumer generates from. Regenerating is idempotent:
`make isa` rewrites the JSON and fails if anything downstream drifts.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))

import tnca_isa as T  # noqa: E402

SCHEMA_VERSION = 1
OUT = os.path.join(ROOT, 'isa', 'tnca-isa-v0.1.json')

# Which instruction-word field each operand kind occupies, and what it means.
# The compiler needs this to emit operands without re-deriving the encoding.
OPERAND_KINDS = {
    'sd':   {'field': 'd', 'class': 'sreg',  'role': 'def'},
    'sa':   {'field': 'a', 'class': 'sreg',  'role': 'use'},
    'sb':   {'field': 'b', 'class': 'sreg',  'role': 'use'},
    'sc':   {'field': 'c', 'class': 'sreg',  'role': 'use'},
    'vd':   {'field': 'd', 'class': 'vreg',  'role': 'def'},
    'va':   {'field': 'a', 'class': 'vreg',  'role': 'use'},
    'vb':   {'field': 'b', 'class': 'vreg',  'role': 'use'},
    'vc':   {'field': 'c', 'class': 'vreg',  'role': 'use'},
    'pd':   {'field': 'd', 'class': 'preg',  'role': 'def'},
    'pa':   {'field': 'a', 'class': 'preg',  'role': 'use'},
    'pb':   {'field': 'b', 'class': 'preg',  'role': 'use'},
    'ad':   {'field': 'd', 'class': 'areg',  'role': 'def'},
    'aa':   {'field': 'a', 'class': 'areg',  'role': 'use'},
    'bd':   {'field': 'd', 'class': 'breg',  'role': 'def'},
    'ba':   {'field': 'a', 'class': 'breg',  'role': 'use'},
    'dd':   {'field': 'd', 'class': 'dreg',  'role': 'def'},
    'da':   {'field': 'a', 'class': 'dreg',  'role': 'use'},
    'wd':   {'field': 'd', 'class': 'wreg',  'role': 'def'},
    'wa':   {'field': 'a', 'class': 'wreg',  'role': 'use'},
    'i16s': {'field': 'c', 'class': 'imm',   'role': 'use', 'bits': 16, 'signed': True},
    'i16z': {'field': 'c', 'class': 'imm',   'role': 'use', 'bits': 16, 'signed': False},
    'i24':  {'field': 'c', 'class': 'imm',   'role': 'use', 'bits': 24, 'signed': False},
    'lbl16': {'field': 'c', 'class': 'label', 'role': 'use', 'bits': 16, 'signed': True},
    'lbl24': {'field': 'c', 'class': 'label', 'role': 'use', 'bits': 24, 'signed': True},
    'spec': {'field': 'mod', 'class': 'special', 'role': 'use'},
    'lane': {'field': 'mod', 'class': 'lane', 'role': 'use'},
    # Reduction destination: scalar, or vector when the '.b' broadcast suffix
    # sets mod[0]. The class depends on the suffix, so the compiler must pick
    # the register file after it has chosen the form.
    'rdst': {'field': 'd', 'class': 'sreg|vreg', 'role': 'def',
             'note': "vreg when the '.b' broadcast suffix is present, else sreg"},
    'cnt2': {'field': 'mod', 'class': 'count', 'role': 'use'},
    'winspec': {'field': 'mod', 'class': 'window', 'role': 'use'},
    'mem_si': {'field': 'a+c', 'class': 'mem', 'role': 'use',
               'note': 'scalar base in a, signed 16-bit displacement in c'},
    'mem_sv': {'field': 'a+b', 'class': 'mem', 'role': 'use',
               'note': 'scalar base in a, per-lane vector offsets in b'},
    'mem_s': {'field': 'a', 'class': 'mem', 'role': 'use',
              'note': 'scalar address, no displacement'},
}


def check_operand_kinds():
    """Every kind the instruction table uses must be described above."""
    used = set()
    for _mnem, (_maj, _f, ops, _sfx) in T.ISA.items():
        used.update(ops)
    unknown = sorted(used - set(OPERAND_KINDS))
    if unknown:
        raise SystemExit(f"export_schema: undescribed operand kinds: {unknown}")
    unused = sorted(set(OPERAND_KINDS) - used)
    return unused


def build():
    unused = check_operand_kinds()

    instructions = []
    for mnem in sorted(T.ISA):
        maj_name, func, ops, sfx = T.ISA[mnem]
        instructions.append({
            'mnemonic': mnem,
            'major': maj_name,
            'major_code': T.MAJ[maj_name],
            'func': func,
            'operands': list(ops),
            'suffix_class': sfx,
            'maskable': T.MAJ[maj_name] in T.MASKABLE_MAJORS,
        })

    # An opcode is (major, func). Two mnemonics sharing one is an ISA bug.
    seen = {}
    for ins in instructions:
        key = (ins['major_code'], ins['func'])
        if key in seen:
            raise SystemExit(
                f"export_schema: opcode collision {key[0]:#04x}/{key[1]:#04x} "
                f"between {seen[key]!r} and {ins['mnemonic']!r}")
        seen[key] = ins['mnemonic']

    return {
        'schema_version': SCHEMA_VERSION,
        'isa_version': '0.1',
        'normative_source': 'doc/isa/TNCA-ISA-v0.1.md',
        'generated_from': 'tools/asm/tnca_isa.py',
        'note': ('Generated file -- do not edit. Run `make isa` after changing '
                 'tools/asm/tnca_isa.py. See compiler_overview.md sec 11.3.'),
        'word': {
            'width': 64,
            'fields': {
                'major': {'shift': T.SH_MAJOR, 'width': T.W_MAJOR},
                'func':  {'shift': T.SH_FUNC,  'width': T.W_FUNC},
                'd':     {'shift': T.SH_D,     'width': T.W_D},
                'a':     {'shift': T.SH_A,     'width': T.W_A},
                'b':     {'shift': T.SH_B,     'width': T.W_B},
                'c':     {'shift': T.SH_C,     'width': T.W_C},
                'pred':  {'shift': T.SH_PRED,  'width': T.W_PRED},
                'x':     {'shift': T.SH_X,     'width': 1},
                'pn':    {'shift': T.SH_PN,    'width': 1},
                'mod':   {'shift': T.SH_MOD,   'width': T.W_MOD},
            },
        },
        'register_files': {
            'sreg': {'count': T.SREG_N, 'per': 'rctx', 'width_bits': 64},
            'preg': {'count': T.PREG_N, 'per': 'rctx', 'width_bits': T.VLEN},
            'vreg': {'count': T.VREG_N, 'per': 'core', 'lanes': T.VLEN,
                     'lane_width_bits': 32},
            'areg': {'count': T.AREG_N, 'per': 'core'},
            'breg': {'count': T.BREG_N, 'per': 'core'},
            'dreg': {'count': T.DREG_N, 'per': 'core'},
            'wreg': {'count': T.WREG_N, 'per': 'core'},
        },
        'vlen': T.VLEN,
        'majors': dict(sorted(T.MAJ.items(), key=lambda kv: kv[1])),
        'maskable_majors': sorted(T.MASKABLE_MAJORS),
        'condition_codes': T.CC,
        'condition_codes_int': sorted(T.CC_INT),
        'condition_codes_fp': sorted(T.CC_FP),
        'special_registers': T.SPECIAL,
        'access_sizes': T.SIZE,
        'cache_hints': T.CACHE_HINT,
        'fp_minmax_modes': T.FMINMAX_MODE,
        'rounding_modes': T.RND,
        'trap_codes': T.TRAP,
        'mma_k': T.MMA_K,
        'mma_dtype': T.MMA_DT,
        'operand_kinds': OPERAND_KINDS,
        'operand_kinds_unused': unused,
        'instructions': instructions,
    }


def main():
    schema = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = json.dumps(schema, indent=2, sort_keys=False) + '\n'

    check_only = '--check' in sys.argv
    if check_only:
        if not os.path.exists(OUT):
            print(f"FAIL: {OUT} does not exist; run `make isa`")
            return 1
        if open(OUT).read() != text:
            print(f"FAIL: {OUT} is stale; run `make isa`")
            return 1
        print(f"PASS: {os.path.relpath(OUT, ROOT)} is up to date "
              f"({len(schema['instructions'])} instructions)")
        return 0

    open(OUT, 'w').write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: "
          f"{len(schema['instructions'])} instructions, "
          f"{len(schema['majors'])} majors, "
          f"{len(schema['special_registers'])} special registers")
    if schema['operand_kinds_unused']:
        print(f"  note: described but unused operand kinds: "
              f"{schema['operand_kinds_unused']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
