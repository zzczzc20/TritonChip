#!/usr/bin/env python3
"""
gen.py -- generate every downstream ISA consumer from the canonical schema.

compiler_overview.md sec 11.3 lists what must come out of one source:
assembler/disassembler tables, compiler opcode definitions, encoding legality,
later TableGen input, and conformance tests. The Python tooling already reads
tools/asm/tnca_isa.py directly; this script produces the rest, so the compiler
never carries a hand-typed opcode number.

  gen.py cpp   -> compiler/include/tnca/ISA/TNCAOpcodes.inc
  gen.py td    -> compiler/include/tnca/ISA/TNCAInstrInfo.td
  gen.py md    -> doc/isa/opcode-map.md
  gen.py all   -> all of the above
  gen.py --check all   regenerate into memory and diff; nonzero if stale
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SCHEMA = os.path.join(ROOT, 'isa', 'tnca-isa-v0.1.json')

BANNER = ("TNCA ISA v0.1 -- GENERATED FILE, DO NOT EDIT.\n"
          "Source: isa/tnca-isa-v0.1.json (itself generated from\n"
          "tools/asm/tnca_isa.py). Regenerate with `make isa`.\n"
          "Normative text: doc/isa/TNCA-ISA-v0.1.md.")


def load():
    with open(SCHEMA) as f:
        return json.load(f)


def ident(mnem):
    """'vf.reduce.add' -> 'VF_REDUCE_ADD'."""
    return mnem.replace('.', '_').upper()


def td_ident(mnem):
    """'vf.reduce.add' -> 'VF_REDUCE_ADD' (TableGen record names)."""
    return ident(mnem)


# --------------------------------------------------------------------------
# C++ : an x-macro table, so a consumer picks its own struct layout.
# --------------------------------------------------------------------------
def gen_cpp(s):
    o = []
    for line in BANNER.split('\n'):
        o.append(f"// {line}")
    o.append("")
    o.append("// Usage:")
    o.append("//   #define TNCA_INST(ID, MNEMONIC, MAJOR, FUNC, MASKABLE) ...")
    o.append("//   #include \"tnca/ISA/TNCAOpcodes.inc\"")
    o.append("//   #undef TNCA_INST")
    o.append("")

    o.append("#ifdef TNCA_MAJOR")
    for name, code in s['majors'].items():
        o.append(f"TNCA_MAJOR({name}, {code:#04x})")
    o.append("#endif")
    o.append("")

    o.append("#ifdef TNCA_INST")
    for ins in s['instructions']:
        o.append("TNCA_INST({id}, \"{mnem}\", {maj:#04x}, {func:#04x}, {mask})".format(
            id=ident(ins['mnemonic']), mnem=ins['mnemonic'],
            maj=ins['major_code'], func=ins['func'],
            mask='true' if ins['maskable'] else 'false'))
    o.append("#endif")
    o.append("")

    o.append("#ifdef TNCA_SPECIAL")
    for name, code in sorted(s['special_registers'].items(), key=lambda kv: kv[1]):
        o.append(f"TNCA_SPECIAL({ident(name)}, \"{name}\", {code:#05x})")
    o.append("#endif")
    o.append("")

    o.append("#ifdef TNCA_TRAP")
    for name, code in sorted(s['trap_codes'].items(), key=lambda kv: kv[1]):
        o.append(f"TNCA_TRAP({name.upper()}, \"{name}\", {code:#04x})")
    o.append("#endif")
    o.append("")

    o.append("#ifdef TNCA_FIELD")
    for name, f in s['word']['fields'].items():
        o.append(f"TNCA_FIELD({name}, {f['shift']}, {f['width']})")
    o.append("#endif")
    o.append("")
    return '\n'.join(o)


# --------------------------------------------------------------------------
# TableGen : input for the thin LLVM target of sec 11.2. Deliberately minimal
# -- it fixes the encoding, and leaves patterns and register classes to the
# hand-written .td files that will sit beside it.
# --------------------------------------------------------------------------
def gen_td(s):
    w = s['word']['fields']
    o = []
    for line in BANNER.split('\n'):
        o.append(f"// {line}")
    o.append("")
    o.append("// This file carries encodings only. Selection patterns, register")
    o.append("// classes and scheduling belong in hand-written .td beside it;")
    o.append("// sec 11.2 keeps the LLVM target thin and off the critical path.")
    o.append("")
    o.append("class TNCAInst<bits<8> major, bits<8> func, string mnem>")
    o.append("    : Instruction {")
    o.append("  field bits<64> Inst;")
    o.append("  let Namespace = \"TNCA\";")
    o.append("  let Size = 8;")
    o.append(f"  let Inst{{{w['major']['shift'] + w['major']['width'] - 1}-"
             f"{w['major']['shift']}}} = major;")
    o.append(f"  let Inst{{{w['func']['shift'] + w['func']['width'] - 1}-"
             f"{w['func']['shift']}}} = func;")
    o.append("  let AsmString = mnem;")
    o.append("  bits<8> Major = major;")
    o.append("  bits<8> Func  = func;")
    o.append("  bit Maskable = 0;")
    o.append("}")
    o.append("")
    for ins in s['instructions']:
        mask = "  let Maskable = 1;\n" if ins['maskable'] else ""
        o.append("def TNCA_{id} : TNCAInst<{maj}, {func}, \"{mnem}\"> {{\n"
                 "{mask}}}".format(id=td_ident(ins['mnemonic']),
                                   maj=ins['major_code'], func=ins['func'],
                                   mnem=ins['mnemonic'], mask=mask))
    o.append("")
    return '\n'.join(o)


# --------------------------------------------------------------------------
# Markdown : the opcode map, so the prose ISA document and the tables cannot
# silently disagree about which (major, func) pairs are taken.
# --------------------------------------------------------------------------
def gen_md(s):
    by_major = {}
    for ins in s['instructions']:
        by_major.setdefault(ins['major'], []).append(ins)

    o = ["# TNCA ISA v0.1 -- opcode map", ""]
    o.append("<!-- " + BANNER.replace('\n', ' ') + " -->")
    o.append("")
    o.append(f"{len(s['instructions'])} instructions across "
             f"{len(by_major)} allocated majors. "
             f"Normative semantics live in `{s['normative_source']}`; this file "
             "only records which encodings are taken.")
    o.append("")
    for name, code in s['majors'].items():
        rows = sorted(by_major.get(name, []), key=lambda i: i['func'])
        o.append(f"## `{name}` (major `{code:#04x}`)")
        o.append("")
        if not rows:
            o.append("Allocated, no instructions defined yet.")
            o.append("")
            continue
        o.append("| func | mnemonic | operands | suffixes | maskable |")
        o.append("|---|---|---|---|---|")
        for ins in rows:
            ops = ', '.join(f"`{x}`" for x in ins['operands']) or '--'
            sfx = f"`{ins['suffix_class']}`" if ins['suffix_class'] else '--'
            o.append(f"| `{ins['func']:#04x}` | `{ins['mnemonic']}` | {ops} | "
                     f"{sfx} | {'yes' if ins['maskable'] else 'no'} |")
        o.append("")
    return '\n'.join(o)


TARGETS = {
    'cpp': ('compiler/include/tnca/ISA/TNCAOpcodes.inc', gen_cpp),
    'td':  ('compiler/include/tnca/ISA/TNCAInstrInfo.td', gen_td),
    'md':  ('doc/isa/opcode-map.md', gen_md),
}


def main(argv):
    check = '--check' in argv
    argv = [a for a in argv if a != '--check']
    which = argv[1:] if len(argv) > 1 else ['all']
    if which == ['all']:
        which = list(TARGETS)

    schema = load()
    stale = []
    for key in which:
        if key not in TARGETS:
            print(f"unknown target {key!r}; known: {', '.join(TARGETS)}")
            return 2
        rel, fn = TARGETS[key]
        path = os.path.join(ROOT, rel)
        text = fn(schema)
        if check:
            cur = open(path).read() if os.path.exists(path) else None
            if cur != text:
                stale.append(rel)
            else:
                print(f"  up to date: {rel}")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').write(text)
            print(f"  wrote {rel} ({len(text.splitlines())} lines)")

    if check and stale:
        print("FAIL: stale generated files: " + ', '.join(stale))
        print("      run `make isa`")
        return 1
    if check:
        print("PASS: all generated ISA files match the schema")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
