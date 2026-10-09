#!/usr/bin/env python3
"""
Generate doc/support_matrix.md -- doc/phase1_tasks_and_acceptance.md R1-01.

    "将'ISA 定义、profile 实现、编译器可生成、整机验证'分为四列，另列
     dtype/K/shape/stride/control-flow 限制。"

Four columns, and the reason there are four is that they disagree. An
instruction can be in the ISA and in no profile; in a profile and beyond the
compiler; emitted by the compiler and never executed on the whole machine.
Collapsing those into one word -- "supported" -- is what the audit found, and
what this file exists to stop.

So none of the columns is written by hand. Each is DERIVED:

  ISA defined          isa/tnca-isa-v0.1.json, the normative export
  profile implemented  the capability the instruction needs, against each
                       profile's generated target description
  compiler can emit    the mnemonic literals the lowering actually passes to
                       the assembler builder
  machine verified     the mnemonic appearing in a program `make top` runs on
                       the four-tile machine

`--check` regenerates into memory and fails if the committed file has drifted,
so the matrix cannot quietly stop describing the machine.

Every mnemonic must be classified by CAPABILITY_OF below. An unclassified one
is an error rather than a default, because a default would silently answer
"available everywhere" for the next instruction somebody adds.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ISA = os.path.join(ROOT, 'isa', 'tnca-isa-v0.1.json')
PROFILES = os.path.join(ROOT, 'spec', 'r1_profiles.json')
TARGETS = os.path.join(ROOT, 'compiler', 'targets')
OUT = 'doc/support_matrix.md'

LOWERING_FILES = ['codegen.py', 'gemm.py', 'pipeline.py', 'partition.py']
TOP_PROGS = ['t10_mesh', 't24_amo_mesh']

# What a profile must have to execute an instruction. Keyed by ISA major,
# because that is the ISA's own grouping and it does not drift when a
# mnemonic is renamed; the per-mnemonic table below is for the handful of
# cases where one major spans two capabilities.
#
# `None` means every profile: the baseline integer, vector, memory, control,
# predicate and conversion set that a TNCA core is by definition.
CAPABILITY_BY_MAJOR = {
    'SALU': None, 'VINT': None, 'VFP': None, 'PRED': None, 'RED': None,
    'SHUF': None, 'CTRL': None, 'SYS': None, 'CVT': None, 'LSU': None,
    'SFU': 'sfu',
    'ATOM': 'atomics_device',
    'MBAR': 'mbarrier',
    'TDESC': 'tensor_dma',
    'TXFER': 'tensor_dma',
    'MATRIX': 'matrix_f16',
    'ACC': 'matrix_f16',
}

CAPABILITY_BY_MNEMONIC = {
    'v.ld.gather': 'gather_scatter',
    'v.st.scatter': 'gather_scatter',
    # Encodable, and no engine can source it. See doc/support_matrix.md's
    # "Not delivered" section and tb/core/progs/t26_stfault.s.
    't.st.async': 'tensor_store',
}

# The limits the contract asks to be listed separately, each with the place
# that enforces it rather than a claim that it is enforced.
LIMITS = [
    ('dtype (matrix)', 'f16, bf16, e5m2, e4m3',
     'compiler/python/tnca_cc/footprints.py `MMA_DTYPE_BITS`; an unknown '
     'dtype is a Diagnostic, checked by FP-05'),
    ('dtype (transfer)', 'f16, bf16, f32',
     'the element-size suffix in `_lower_transfer`; anything else has no '
     'suffix and cannot be emitted'),
    ('dtype (convert)', 'f32, bf16, f16, e5m2, e4m3',
     '`vf.cvt.<src>.<dst>` mod[5:0], ISA sec 8.4; t21_cvt runs all '
     'twenty-five source/destination pairs, sixteen lanes each'),
    ('K step', 'k16, k32, k64',
     '`MMA_K_STEPS`; the footprint is sized from the INSTRUCTION, which is '
     'audit F01 and case FP-02/FP-04'),
    ('output tile', 'at most the accumulator shape (16x16)',
     '`plan_gemm` refuses a larger tile; a bigger output needs several '
     'accumulators and an outer loop, which is not planned'),
    ('transfer rank', '1 or 2',
     'ISA sec 9E; rank > 2 raises DESCRIPTOR_FAULT in the engine and is a '
     'compile error in `_lower_transfer`. Rank 3-5 is NOT delivered'),
    ('transfer stride', 'row stride and base both 8-byte aligned; '
     'innermost stride = element size',
     'checked in rtl/core/tnca_tensor.v before the transfer starts; '
     't17_descfault is the negative'),
    ('window write granularity', 'whole 8-byte units',
     'rtl/core/tnca_tensor.v refuses a descriptor whose row is not a whole '
     'number of 8-byte units, so every unit the engine writes is a full one. '
     'The write port itself carries a byte keep-mask for a partial tail unit; '
     'with the geometry check in place that path cannot be reached, and a '
     'byte-granular transfer is NOT delivered'),
    ('control flow', 'a single block per region; loops via `b.loop`',
     'multi-block CFG, nested regions and events through block arguments '
     'are refused by name -- IR-02. This is a frozen boundary, not a '
     'removed capability'),
    ('async lifetime', 'one completion event per command; sources release '
     'on command completion',
     'ISA v0.1 `source_release`; the compiler takes the bound rather than '
     'the looser one sec 10.2 permits hardware'),
]

# Per-async-operation contract. R1-01 asks for read/write footprints, the
# moment the resource is held, the release event, error-completion behaviour,
# barrier generation and predicated-off behaviour -- for ALL of them.
ASYNC_OPS = [
    dict(
        mnemonic='t.ld.async',
        reads='the coordinate run: `rank` consecutive scalars from Sb '
              '(ISA sec 9E), plus the descriptor snapshot',
        writes='the whole destination window, every register of it',
        held='the window, from issue',
        release='the completion event; the coordinate registers are free one '
                'cycle after issue, because rtl/core/tnca_tensor.v:357 '
                'latches c0/c1 on cmd_valid and never reads the file again '
                '(case OWN-04)',
        error='the barrier is signalled with its error bit set, so the '
              'waiter is released rather than hanging; the window is not '
              'published',
        generation='counts one operation on the armed phase of its barrier',
        predicated='a fully masked-off transfer performs no access and still '
                   'completes exactly once (ISA sec 9A.3), so the static '
                   'operation count stays correct (case OWN-09)'),
    dict(
        mnemonic='t.st.async',
        reads='the source window -- LATE, after the issuing instruction has '
              'retired (sec 13.4) -- and the coordinate run',
        writes='memory only; no register file',
        held='the source window, from issue',
        release='the completion event. Reading the window meanwhile is '
                'legal; overwriting it is not (case OWN-03)',
        error='not executable on any R1 profile: the engine has no vector '
              'read port and raises ILLEGAL_INSTRUCTION with the barrier '
              'error bit set (t26_stfault). The compiler refuses to emit it',
        generation='would count one operation; the capability '
                   '`tensor_store` is false everywhere',
        predicated='n/a on this profile set'),
    dict(
        mnemonic='m.mma.async',
        reads='both fragments in full: fragment_registers(m,k) and '
              'fragment_registers(k,n) from the INSTRUCTION\'s K and dtype, '
              'not the target default (audit F01)',
        writes='the destination accumulator, for the whole command',
        held='both source windows and the accumulator, from issue',
        release='the completion event. With mod[4] the source-free event '
                'arrives one barrier phase earlier; on ISA v0.1 the compiler '
                'takes command completion for both',
        error='the barrier carries the error bit; the accumulator is left '
              'undefined and must not be read',
        generation='counts one operation, or two with the source-free event, '
                   'on the armed phase',
        predicated='the same rule: completes once, whatever the mask'),
    dict(
        mnemonic='t.prefetch',
        reads='the coordinate run',
        writes='nothing architectural',
        held='nothing',
        release='none: it raises no completion and counts on no barrier',
        error='none; a failed prefetch is not observable',
        generation='n/a -- it must never be counted in an mb.arm',
        predicated='performs no access when masked off, like any other'),
]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def capability_of(ins):
    m = ins['mnemonic']
    if m in CAPABILITY_BY_MNEMONIC:
        return CAPABILITY_BY_MNEMONIC[m]
    if ins['major'] in CAPABILITY_BY_MAJOR:
        return CAPABILITY_BY_MAJOR[ins['major']]
    return 'UNCLASSIFIED'


def emitted_mnemonics():
    """Every mnemonic the lowering can hand to the builder.

    Two sources, because the lowering has two shapes. Most calls pass a
    literal; the arithmetic ones look up a table (`_SBIN`, `_VBIN_FP`, ...)
    whose VALUES are the mnemonics, and a scan that only read literals
    reported every scalar and vector ALU instruction as unemitted -- which
    would have made this column understate the compiler rather than describe
    it. So the tables are read by importing the module.

    An f-string like f"v.ld.{size}" contributes its literal prefix `v.ld.`,
    which is recorded without the trailing dot: the suffix is a mode of the
    same instruction, not a different one.
    """
    found = set()
    base = os.path.join(ROOT, 'compiler', 'python', 'tnca_cc')
    pat = re.compile(r"""_?emit\(\s*f?['"]([a-z][a-z0-9._]*)""")
    direct = re.compile(r"""directive\(\s*f?["']\s*\.?([a-z][a-z0-9._]*)""")
    for f in LOWERING_FILES:
        p = os.path.join(base, f)
        if not os.path.exists(p):
            continue
        text = open(p).read()
        found |= set(pat.findall(text))
        found |= set(direct.findall(text))
        # `mnem = 't.st.async' if store else 't.ld.async'` -- the transfer
        # lowering picks its mnemonic before emitting it, so the literal is
        # on an assignment rather than at the call.
        for line in text.splitlines():
            if re.match(r"\s*\w*mnem\w*\s*=", line):
                found |= set(re.findall(r"['\"]([a-z][a-z0-9._]{2,})['\"]",
                                        line))
    found = {m.rstrip('.') for m in found if m.rstrip('.')}

    sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
    try:
        from tnca_cc import codegen as CG
        for name in dir(CG):
            val = getattr(CG, name)
            if isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, str) and re.fullmatch(
                            r'[a-z][a-z0-9._]*', v) and '.' in v:
                        found.add(v)
    except Exception as e:
        raise SystemExit(f"could not read the lowering tables: {e}")
    return found


def machine_mnemonics():
    """Mnemonics appearing in the programs `make top` runs."""
    found = set()
    pat = re.compile(r'^\s{2,}(\.?[a-z][a-z0-9._]*)')
    for prog in TOP_PROGS:
        p = os.path.join(ROOT, 'tb', 'core', 'progs', f'{prog}.s')
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.split('//')[0]
            m = pat.match(line)
            if m:
                found.add(m.group(1).lstrip('.'))
    return found


def matches(mnemonic, seen):
    """Does `seen` contain this mnemonic, allowing for suffixes?

    The tables name `t.ld.async` and the emitter writes `t.ld.async.h`, so an
    exact-match column would report every transfer as unemitted.
    """
    if mnemonic in seen:
        return True
    return any(s == mnemonic or s.startswith(mnemonic + '.') for s in seen)


def render():
    isa = load(ISA)
    spec = load(PROFILES)
    profiles = sorted(spec['profiles'])
    targets = {}
    for name in profiles:
        with open(os.path.join(TARGETS,
                               spec['profiles'][name]['target_file'])) as fh:
            targets[name] = json.load(fh)

    emitted = emitted_mnemonics()
    machine = machine_mnemonics()

    rows = []
    unclassified = []
    for ins in sorted(isa['instructions'], key=lambda i: (i['major_code'],
                                                          i['func'])):
        m = ins['mnemonic']
        cap = capability_of(ins)
        if cap == 'UNCLASSIFIED':
            unclassified.append(f"{m} (major {ins['major']})")
            continue
        per_profile = []
        for name in profiles:
            if cap is None:
                per_profile.append('yes')
            else:
                per_profile.append(
                    'yes' if targets[name]['capabilities'].get(cap) else 'no')
        # Three states, not two. A lowering can exist and be refused on
        # every profile -- `t.st.async` is exactly that -- and calling it
        # either "yes" or "no" says something false.
        if not matches(m, emitted):
            emit_state = 'no'
        elif cap and not any(p == 'yes' for p in per_profile):
            emit_state = 'gated'
        else:
            emit_state = 'yes'
        rows.append(dict(mnemonic=m, major=ins['major'], cap=cap,
                         profiles=per_profile, emitted=emit_state,
                         machine=matches(m, machine)))
    if unclassified:
        raise SystemExit(
            "these majors are not classified in CAPABILITY_BY_MAJOR, and a "
            "default would answer 'available everywhere' for them:\n  "
            + '\n  '.join(unclassified))

    out = []
    a = out.append
    a('# TNCA support matrix')
    a('')
    a('<!-- GENERATED by tools/isa/gen_support_matrix.py -- do not edit. -->')
    a('<!-- Run `make isa` to regenerate; `make r1-check-generated` fails if '
      'this file has drifted from its sources. -->')
    a('')
    a(f"ISA version {isa['isa_version']}. "
      f"Contract: `doc/phase1_tasks_and_acceptance.md` R1-01.")
    a('')
    a('Four columns, because they disagree. An instruction can be in the ISA')
    a('and in no profile; in a profile and beyond the compiler; emitted by the')
    a('compiler and never executed on the whole machine. One word --')
    a('"supported" -- hides all three differences, which is what the audit')
    a('found. None of these columns is written by hand: each is derived from')
    a('the normative ISA export, the generated target descriptions, the')
    a('lowering itself, and the programs `make top` runs.')
    a('')
    a('"machine verified" means the mnemonic is executed by a program running')
    a('on the four-tile machine (`make top`: '
      + ', '.join(f"`{p}`" for p in TOP_PROGS) + ').')
    a('A `no` there is not a defect -- most instructions are verified on the')
    a('single-core bench, which `make core` runs under three memory models --')
    a('it is a statement about which column of evidence exists.')
    a('')
    a('## Instructions')
    a('')
    a('| mnemonic | major | capability | '
      + ' | '.join(profiles) + ' | compiler can emit | machine verified |')
    a('|---|---|---|' + '---|' * (len(profiles) + 2))
    for r in rows:
        a(f"| `{r['mnemonic']}` | {r['major']} | "
          f"{r['cap'] or '(baseline)'} | "
          + ' | '.join(r['profiles'])
          + f" | {r['emitted']}"
          + f" | {'yes' if r['machine'] else 'no'} |")
    a('')
    n_emit = sum(1 for r in rows if r['emitted'] == 'yes')
    n_gated = sum(1 for r in rows if r['emitted'] == 'gated')
    n_mach = sum(1 for r in rows if r['machine'])
    a(f"{len(rows)} instructions: {n_emit} the compiler emits, {n_gated} it "
      f"has a lowering for and refuses on every R1 profile, {n_mach} executed "
      f"on the whole machine.")
    a('')
    a('`gated` in the emit column means the lowering exists and the capability')
    a('is false everywhere, so asking for it is a compile error rather than a')
    a('runtime trap. Calling that either "yes" or "no" would say something')
    a('false about it.')
    a('')
    a('## Limits')
    a('')
    a('| what | limit | where it is enforced |')
    a('|---|---|---|')
    for what, limit, where in LIMITS:
        a(f"| {what} | {limit} | {where} |")
    a('')
    a('## Asynchronous operations')
    a('')
    a('For each: what the engine reads and writes beyond the registers the')
    a('encoding names, when the resource is held, what releases it, what an')
    a('error completion does, how it counts against a barrier generation, and')
    a('what happens when it is predicated off. These are the facts')
    a('`compiler_overview.md` sec 13.4 says must be in the compiler model,')
    a('because a liveness analysis built from the operand fields sees two')
    a('registers where the engine touches thirty-two.')
    a('')
    for op in ASYNC_OPS:
        a(f"### `{op['mnemonic']}`")
        a('')
        a(f"- **reads** {op['reads']}")
        a(f"- **writes** {op['writes']}")
        a(f"- **held** {op['held']}")
        a(f"- **released by** {op['release']}")
        a(f"- **on error** {op['error']}")
        a(f"- **barrier generation** {op['generation']}")
        a(f"- **predicated off** {op['predicated']}")
        a('')
    a('## Not delivered')
    a('')
    a('Listed because R1-01 forbids marking an undelivered requirement as not')
    a('applicable. Each of these is a FAIL or NOT RUN, never an absence.')
    a('')
    a('| item | status | where it is tracked |')
    a('|---|---|---|')
    a('| transfer rank 3-5 | NOT DELIVERED | ISA sec 9E allows up to 5; the '
      'engine executes 1 and 2 and raises DESCRIPTOR_FAULT above that |')
    a('| byte-granular window writes | NOT DELIVERED | a descriptor whose '
      'row is not a whole number of 8-byte units raises DESCRIPTOR_FAULT |')
    a('| `t.st.async` execution | NOT DELIVERED | encodable, faults with '
      'ILLEGAL_INSTRUCTION; capability `tensor_store` is false |')
    a('| transposed transfer (`t.ld.async` mod[5]) | PROFILE-GATED | '
      'implemented behind HAS_TENSOR_XPOSE (profile core-xpose-v1, '
      'capability `tensor_transpose`): 16-bit elements, 4/8/16 source rows '
      '(ISA sec 9E; tb/core/progs/t30_xpose.s). Without it the encoding '
      'raises ILLEGAL_INSTRUCTION (t31_xpose_illegal.s) |')
    a('| general CFG in the MLIR verifier | FROZEN BOUNDARY | refused by '
      'name (IR-02), not accepted unchecked |')
    a('| `tnca_vector_fpu` synthesis | NOT COMPLETED | see '
      '`doc/synthesis.md`; yosys did not finish |')
    a('')
    return '\n'.join(out) + '\n'


def main(argv):
    text = render()
    path = os.path.join(ROOT, OUT)
    if '--check' in argv:
        cur = open(path).read() if os.path.exists(path) else None
        if cur != text:
            print(f"FAIL: {OUT} is stale; run `make isa`")
            return 1
        print(f"  up to date: {OUT}")
        return 0
    with open(path, 'w') as fh:
        fh.write(text)
    print(f"  wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
