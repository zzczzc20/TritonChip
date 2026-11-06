"""
tnca_isa.py -- machine-readable TNCA ISA v0.1 encoding tables.

Normative source: doc/isa/TNCA-ISA-v0.1.md. This module is the single place the
Python tooling (assembler, disassembler, architectural model) reads encodings
from, so the three cannot drift apart independently.
"""

# ---------------------------------------------------------------------------
# Instruction word layout (ISA sec 2)
# ---------------------------------------------------------------------------
SH_MAJOR, W_MAJOR = 56, 8
SH_FUNC,  W_FUNC  = 48, 8
SH_D,     W_D     = 40, 8
SH_A,     W_A     = 32, 8
SH_B,     W_B     = 24, 8
SH_C,     W_C     = 16, 8
SH_PRED,  W_PRED  = 12, 4
SH_X              = 11
SH_PN             = 10
SH_MOD,   W_MOD   = 0, 10

VLEN = 16
SREG_N = 64
PREG_N = 16
VREG_N = 256
AREG_N = 16      # accumulator slots
BREG_N = 32      # mbarriers
DREG_N = 16      # tensor descriptor handles
WREG_N = 32      # RF window handles

# ---------------------------------------------------------------------------
# Major opcodes (ISA sec 3)
# ---------------------------------------------------------------------------
MAJ = {
    'SYS': 0x00, 'SALU': 0x01, 'CTRL': 0x02,
    'VINT': 0x10, 'VFP': 0x11, 'CVT': 0x12, 'PRED': 0x13,
    'SHUF': 0x14, 'RED': 0x15, 'SFU': 0x16,
    'LSU': 0x20, 'ATOM': 0x21,
    'TDESC': 0x30, 'TXFER': 0x31, 'THINT': 0x32,
    'MATRIX': 0x40, 'ACC': 0x41, 'MBAR': 0x50, 'ROLE': 0x51, 'DEBUG': 0x60,
}
MAJ_NAME = {v: k for k, v in MAJ.items()}

# ---------------------------------------------------------------------------
# Condition codes (ISA sec 4)
# ---------------------------------------------------------------------------
CC = {
    'eq': 0x0, 'ne': 0x1, 'lt': 0x2, 'le': 0x3, 'gt': 0x4, 'ge': 0x5,
    'ltu': 0x6, 'leu': 0x7, 'gtu': 0x8, 'geu': 0x9, 'o': 0xa, 'uo': 0xb,
}
CC_NAME = {v: k for k, v in CC.items()}
CC_INT = {'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'ltu', 'leu', 'gtu', 'geu'}
CC_FP = {'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'o', 'uo'}

# ---------------------------------------------------------------------------
# Special registers (ISA sec 5.1)
# ---------------------------------------------------------------------------
SPECIAL = {
    'pid.x': 0x000, 'pid.y': 0x001, 'pid.z': 0x002,
    'npid.x': 0x004, 'npid.y': 0x005, 'npid.z': 0x006,
    'role_id': 0x008, 'num_roles': 0x009,
    'core_id': 0x00a, 'num_cores': 0x00b, 'num_lanes': 0x00c,
    'num_vregs': 0x00d, 'tensor_max_rank': 0x00e, 'num_acc': 0x00f,
    'clock': 0x010, 'instret': 0x011,
    'num_barriers': 0x012, 'num_windows': 0x013,
    'num_descriptors': 0x014, 'num_phys_vregs': 0x015,
    'arg_base': 0x020,
    'isa_version': 0x030, 'cap0': 0x031,
    'trap_pc': 0x040, 'trap_code': 0x041,
}
SPECIAL_NAME = {v: k for k, v in SPECIAL.items()}

# ---------------------------------------------------------------------------
# Access sizes / cache hints (ISA sec 9.1)
# ---------------------------------------------------------------------------
SIZE = {'b': 0, 'h': 1, 'w': 2, 'd': 3}
CACHE_HINT = {'default': 0, 'stream': 1, 'hipri': 2}

# Floating-point min/max NaN modes (ISA sec 8.3)
FMINMAX_MODE = {'ieee': 0, 'prop': 1, 'fast': 2}

# FP->int rounding modes (ISA sec 8.4)
RND = {'rtz': 0, 'rne': 1, 'rmi': 2, 'rpi': 3}

# Trap codes (ISA sec 10)
TRAP = {
    'NONE': 0x00, 'ILLEGAL_INSTRUCTION': 0x01, 'MISALIGNED_PC': 0x02,
    'ACCESS_FAULT': 0x03, 'DESCRIPTOR_FAULT': 0x04, 'BARRIER_FAULT': 0x05,
    'RESOURCE_EXHAUSTED': 0x06, 'TRAP_USER': 0x40,
}
TRAP_NAME = {v: k for k, v in TRAP.items()}

# ---------------------------------------------------------------------------
# Operand kinds used by the instruction table.
#
#   sd/sa/sb/sc  scalar register in field d/a/b/c
#   vd/va/vb/vc  vector register
#   pd/pa/pb     predicate register
#   i16s/i16z    16-bit immediate in {b,c}, sign- or zero-extended
#   i24          24-bit immediate in {a,b,c}
#   lbl16/lbl24  branch label, encoded as a word displacement
#   spec         special-register name, encoded into mod
#   lane         lane index literal, encoded into mod[3:0]
# ---------------------------------------------------------------------------

# name -> (major, func, operand-kind tuple, suffix-class)
#
# suffix-class is one of:
#   None      no suffixes
#   'cc_int'  required integer condition code, into mod[3:0]
#   'cc_fp'   required FP condition code, into mod[3:0]
#   'ls'      load/store: size (required) + optional sign + optional cache hint
#   'fmm'     optional FP min/max mode, into mod[1:0] (default ieee)
#   'rnd'     optional FP->int rounding mode, into mod[2:0] (default rtz)
#   'lsl'     optional "lsl #n" trailing operand, into mod[1:0]
#   'redb'    optional '.b' broadcast form, into mod[0]
#   'ftz'     optional '.ftz', into mod[8]

ISA = {
    # ---- system (ISA sec 5) -------------------------------------------------
    'nop':          ('SYS',  0x00, (),                  None),
    's.rdspecial':  ('SYS',  0x01, ('sd', 'spec'),      None),
    's.wrspecial':  ('SYS',  0x02, ('spec', 'sa'),      None),

    # ---- scalar integer (ISA sec 6) ----------------------------------------
    's.mov':        ('SALU', 0x00, ('sd', 'sa'),              None),
    's.add':        ('SALU', 0x01, ('sd', 'sa', 'sb'),        None),
    's.sub':        ('SALU', 0x02, ('sd', 'sa', 'sb'),        None),
    's.mul':        ('SALU', 0x03, ('sd', 'sa', 'sb'),        None),
    's.mulh.s':     ('SALU', 0x04, ('sd', 'sa', 'sb'),        None),
    's.mulh.u':     ('SALU', 0x05, ('sd', 'sa', 'sb'),        None),
    's.and':        ('SALU', 0x08, ('sd', 'sa', 'sb'),        None),
    's.or':         ('SALU', 0x09, ('sd', 'sa', 'sb'),        None),
    's.xor':        ('SALU', 0x0a, ('sd', 'sa', 'sb'),        None),
    's.andn':       ('SALU', 0x0b, ('sd', 'sa', 'sb'),        None),
    's.shl':        ('SALU', 0x0c, ('sd', 'sa', 'sb'),        None),
    's.lshr':       ('SALU', 0x0d, ('sd', 'sa', 'sb'),        None),
    's.ashr':       ('SALU', 0x0e, ('sd', 'sa', 'sb'),        None),
    's.min.s':      ('SALU', 0x10, ('sd', 'sa', 'sb'),        None),
    's.max.s':      ('SALU', 0x11, ('sd', 'sa', 'sb'),        None),
    's.min.u':      ('SALU', 0x12, ('sd', 'sa', 'sb'),        None),
    's.max.u':      ('SALU', 0x13, ('sd', 'sa', 'sb'),        None),
    's.clz':        ('SALU', 0x14, ('sd', 'sa'),              None),
    's.popc':       ('SALU', 0x15, ('sd', 'sa'),              None),
    's.sext.w':     ('SALU', 0x16, ('sd', 'sa'),              None),
    's.zext.w':     ('SALU', 0x17, ('sd', 'sa'),              None),
    's.cmp':        ('SALU', 0x18, ('sd', 'sa', 'sb'),        'cc_int'),
    's.sel':        ('SALU', 0x19, ('sd', 'sa', 'sb', 'sc'),  None),
    's.addi':       ('SALU', 0x20, ('sd', 'sa', 'i16s'),      None),
    's.subi':       ('SALU', 0x21, ('sd', 'sa', 'i16s'),      None),
    's.muli':       ('SALU', 0x22, ('sd', 'sa', 'i16s'),      None),
    's.andi':       ('SALU', 0x28, ('sd', 'sa', 'i16z'),      None),
    's.ori':        ('SALU', 0x29, ('sd', 'sa', 'i16z'),      None),
    's.xori':       ('SALU', 0x2a, ('sd', 'sa', 'i16z'),      None),
    's.shli':       ('SALU', 0x2c, ('sd', 'sa', 'i16z'),      None),
    's.lshri':      ('SALU', 0x2d, ('sd', 'sa', 'i16z'),      None),
    's.ashri':      ('SALU', 0x2e, ('sd', 'sa', 'i16z'),      None),
    's.cmpi':       ('SALU', 0x38, ('sd', 'sa', 'i16s'),      'cc_int'),
    's.movz':       ('SALU', 0x40, ('sd', 'i16z'),            'lsl'),
    's.movk':       ('SALU', 0x41, ('sd', 'i16z'),            'lsl'),
    's.movn':       ('SALU', 0x42, ('sd', 'i16z'),            'lsl'),

    # ---- control flow (ISA sec 7) ------------------------------------------
    'br':           ('CTRL', 0x00, ('lbl24',),          None),
    'br.nz':        ('CTRL', 0x01, ('sa', 'lbl16'),     None),
    'br.z':         ('CTRL', 0x02, ('sa', 'lbl16'),     None),
    'br.r':         ('CTRL', 0x04, ('sa',),             None),
    'call':         ('CTRL', 0x08, ('sd', 'lbl24'),     None),
    'ret':          ('CTRL', 0x09, ('sa',),             None),
    'yield':        ('CTRL', 0x10, (),                  None),
    'trap':         ('CTRL', 0x18, ('i24',),            None),
    'exit':         ('CTRL', 0x1f, (),                  None),

    # ---- vector integer (ISA sec 8.1) --------------------------------------
    'v.mov':        ('VINT', 0x00, ('vd', 'va'),              None),
    'v.add':        ('VINT', 0x01, ('vd', 'va', 'vb'),        None),
    'v.sub':        ('VINT', 0x02, ('vd', 'va', 'vb'),        None),
    'v.mul':        ('VINT', 0x03, ('vd', 'va', 'vb'),        None),
    'v.mulh.s':     ('VINT', 0x04, ('vd', 'va', 'vb'),        None),
    'v.mulh.u':     ('VINT', 0x05, ('vd', 'va', 'vb'),        None),
    'v.and':        ('VINT', 0x08, ('vd', 'va', 'vb'),        None),
    'v.or':         ('VINT', 0x09, ('vd', 'va', 'vb'),        None),
    'v.xor':        ('VINT', 0x0a, ('vd', 'va', 'vb'),        None),
    'v.andn':       ('VINT', 0x0b, ('vd', 'va', 'vb'),        None),
    'v.shl':        ('VINT', 0x0c, ('vd', 'va', 'vb'),        None),
    'v.lshr':       ('VINT', 0x0d, ('vd', 'va', 'vb'),        None),
    'v.ashr':       ('VINT', 0x0e, ('vd', 'va', 'vb'),        None),
    'v.min.s':      ('VINT', 0x10, ('vd', 'va', 'vb'),        None),
    'v.max.s':      ('VINT', 0x11, ('vd', 'va', 'vb'),        None),
    'v.min.u':      ('VINT', 0x12, ('vd', 'va', 'vb'),        None),
    'v.max.u':      ('VINT', 0x13, ('vd', 'va', 'vb'),        None),
    'v.clz':        ('VINT', 0x14, ('vd', 'va'),              None),
    'v.popc':       ('VINT', 0x15, ('vd', 'va'),              None),
    'v.cmp':        ('VINT', 0x18, ('pd', 'va', 'vb'),        'cc_int'),
    'v.sel':        ('VINT', 0x19, ('vd', 'va', 'vb'),        None),
    'v.splat':      ('VINT', 0x1a, ('vd', 'sa'),              None),
    'v.iota':       ('VINT', 0x1b, ('vd',),                   None),
    'v.addi':       ('VINT', 0x20, ('vd', 'va', 'i16s'),      None),
    'v.subi':       ('VINT', 0x21, ('vd', 'va', 'i16s'),      None),
    'v.muli':       ('VINT', 0x22, ('vd', 'va', 'i16s'),      None),
    'v.andi':       ('VINT', 0x28, ('vd', 'va', 'i16z'),      None),
    'v.ori':        ('VINT', 0x29, ('vd', 'va', 'i16z'),      None),
    'v.xori':       ('VINT', 0x2a, ('vd', 'va', 'i16z'),      None),
    'v.shli':       ('VINT', 0x2c, ('vd', 'va', 'i16z'),      None),
    'v.lshri':      ('VINT', 0x2d, ('vd', 'va', 'i16z'),      None),
    'v.ashri':      ('VINT', 0x2e, ('vd', 'va', 'i16z'),      None),
    'v.cmpi':       ('VINT', 0x38, ('pd', 'va', 'i16s'),      'cc_int'),
    'v.movi':       ('VINT', 0x40, ('vd', 'i16s'),            None),

    # ---- vector FP (ISA sec 8.2) -------------------------------------------
    'vf.add':       ('VFP',  0x01, ('vd', 'va', 'vb'),        'ftz'),
    'vf.sub':       ('VFP',  0x02, ('vd', 'va', 'vb'),        'ftz'),
    'vf.mul':       ('VFP',  0x03, ('vd', 'va', 'vb'),        'ftz'),
    'vf.fma':       ('VFP',  0x06, ('vd', 'va', 'vb', 'vc'),  'ftz'),
    'vf.fms':       ('VFP',  0x07, ('vd', 'va', 'vb', 'vc'),  'ftz'),
    'vf.min':       ('VFP',  0x10, ('vd', 'va', 'vb'),        'fmm'),
    'vf.max':       ('VFP',  0x11, ('vd', 'va', 'vb'),        'fmm'),
    'vf.neg':       ('VFP',  0x14, ('vd', 'va'),              None),
    'vf.abs':       ('VFP',  0x15, ('vd', 'va'),              None),
    'vf.cmp':       ('VFP',  0x18, ('pd', 'va', 'vb'),        'cc_fp'),

    # ---- convert (ISA sec 8.4) ---------------------------------------------
    'vf.cvt.s32.f32': ('CVT', 0x00, ('vd', 'va'),             None),
    'vf.cvt.u32.f32': ('CVT', 0x01, ('vd', 'va'),             None),
    'vf.cvt.f32.s32': ('CVT', 0x02, ('vd', 'va'),             'rnd'),
    'vf.cvt.f32.u32': ('CVT', 0x03, ('vd', 'va'),             'rnd'),
    # vf.cvt.<src>.<dst> for any pair of the five float formats. One mnemonic
    # with two format suffixes rather than twenty-five table entries; the
    # exact-match rule in resolve_mnemonic keeps the four int conversions
    # above resolving to themselves.
    'vf.cvt':         ('CVT', 0x04, ('vd', 'va'),             'fmt2'),

    # ---- predicate (ISA sec 8.5) -------------------------------------------
    'p.mov':        ('PRED', 0x00, ('pd', 'pa'),        None),
    'p.and':        ('PRED', 0x01, ('pd', 'pa', 'pb'),  None),
    'p.or':         ('PRED', 0x02, ('pd', 'pa', 'pb'),  None),
    'p.xor':        ('PRED', 0x03, ('pd', 'pa', 'pb'),  None),
    'p.not':        ('PRED', 0x04, ('pd', 'pa'),        None),
    'p.andn':       ('PRED', 0x05, ('pd', 'pa', 'pb'),  None),
    'p.any':        ('PRED', 0x10, ('sd', 'pa'),        None),
    'p.all':        ('PRED', 0x11, ('sd', 'pa'),        None),
    'p.ballot':     ('PRED', 0x12, ('sd', 'pa'),        None),
    'p.frombits':   ('PRED', 0x13, ('pd', 'sa'),        None),
    'p.popc':       ('PRED', 0x14, ('sd', 'pa'),        None),

    # ---- shuffle (ISA sec 8.6) ---------------------------------------------
    'v.shuffle.xor':  ('SHUF', 0x00, ('vd', 'va', 'sb'),  None),
    'v.shuffle.up':   ('SHUF', 0x01, ('vd', 'va', 'sb'),  None),
    'v.shuffle.down': ('SHUF', 0x02, ('vd', 'va', 'sb'),  None),
    'v.permute':      ('SHUF', 0x03, ('vd', 'va', 'vb'),  None),
    'v.rotate':       ('SHUF', 0x04, ('vd', 'va', 'sb'),  None),
    'v.extract':      ('SHUF', 0x08, ('sd', 'va', 'lane'), None),
    'v.insert':       ('SHUF', 0x09, ('vd', 'sa', 'lane'), None),

    # ---- reduction (ISA sec 8.7) -------------------------------------------
    'v.reduce.add':   ('RED', 0x00, ('rdst', 'va'), 'redb'),
    'v.reduce.max.s': ('RED', 0x01, ('rdst', 'va'), 'redb'),
    'v.reduce.min.s': ('RED', 0x02, ('rdst', 'va'), 'redb'),
    'v.reduce.and':   ('RED', 0x03, ('rdst', 'va'), 'redb'),
    'v.reduce.or':    ('RED', 0x04, ('rdst', 'va'), 'redb'),
    'v.reduce.xor':   ('RED', 0x05, ('rdst', 'va'), 'redb'),
    'v.reduce.max.u': ('RED', 0x06, ('rdst', 'va'), 'redb'),
    'v.reduce.min.u': ('RED', 0x07, ('rdst', 'va'), 'redb'),
    'vf.reduce.add':  ('RED', 0x08, ('rdst', 'va'), 'redb'),
    'vf.reduce.max':  ('RED', 0x09, ('rdst', 'va'), 'redb'),
    'vf.reduce.min':  ('RED', 0x0a, ('rdst', 'va'), 'redb'),

    # ---- SFU (ISA sec 8.8) -------------------------------------------------
    'v.exp2.approx':  ('SFU', 0x00, ('vd', 'va'), None),
    'v.rcp.approx':   ('SFU', 0x01, ('vd', 'va'), None),
    'v.rsqrt.approx': ('SFU', 0x02, ('vd', 'va'), None),
    'v.log2.approx':  ('SFU', 0x03, ('vd', 'va'), None),

    # ---- mbarrier (ISA sec 9A) ---------------------------------------------
    'mb.init':     ('MBAR', 0x00, ('bd',),                        None),
    'mb.arm':      ('MBAR', 0x01, ('sd', 'ba', 'cnt2'),           None),
    'mb.arrive':   ('MBAR', 0x02, ('ba',),                        None),
    'mb.test':     ('MBAR', 0x03, ('sd', 'sa'),                   None),
    'mb.wait':     ('MBAR', 0x04, ('sa',),                        None),
    'mb.inval':    ('MBAR', 0x05, ('ba',),                        None),

    # ---- accumulator (ISA sec 9B) ------------------------------------------
    'a.vread':     ('ACC', 0x00, ('vd', 'aa', 'sb'),              None),
    'a.vwrite':    ('ACC', 0x01, ('ad', 'sa', 'vb'),              None),
    'a.mul.row':   ('ACC', 0x02, ('ad', 'sa', 'sb'),              None),
    'a.fma.row':   ('ACC', 0x03, ('ad', 'sa', 'vb', 'sc'),        None),
    'a.zero':      ('ACC', 0x08, ('ad',),                         None),

    # ---- matrix (ISA sec 9C) -----------------------------------------------
    'm.zero':      ('MATRIX', 0x00, ('ad',),                      None),
    'm.mma.async': ('MATRIX', 0x01, ('ad', 'va', 'vb', 'sc'),     'mma'),
    # ISA sec 9C.4 (approved 2026-10-29): grouped dot-product accumulation
    'm.mma.dp.async': ('MATRIX', 0x02, ('ad', 'va', 'vb', 'sc'),  'mmadp'),

    # ---- tensor descriptor / window (ISA sec 9D) ---------------------------
    'td.load':     ('TDESC', 0x00, ('dd', 'mem_s'),               None),
    'td.release':  ('TDESC', 0x02, ('dd',),                       None),
    'w.def':       ('TDESC', 0x10, ('wd', 'winspec'),             None),
    'w.release':   ('TDESC', 0x11, ('wd',),                       None),

    # ---- tensor transfer (ISA sec 9E) --------------------------------------
    't.ld.async':  ('TXFER', 0x00, ('wd', 'da', 'sb', 'sc'),      'txf'),
    't.st.async':  ('TXFER', 0x08, ('wd', 'da', 'sb', 'sc'),      'txf'),
    't.prefetch':  ('TXFER', 0x10, ('da', 'sb'),                  'txf'),

    # ---- atomic / fence (ISA sec 9.2) --------------------------------------
    # The size suffix is mandatory and only .w and .d exist; order and scope
    # default to relaxed and core, which are the weakest and therefore the
    # ones a program must ask to strengthen.
    'atom.add':      ('ATOM', 0x00, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.and':      ('ATOM', 0x01, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.or':       ('ATOM', 0x02, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.xor':      ('ATOM', 0x03, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.xchg':     ('ATOM', 0x04, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.min.s':    ('ATOM', 0x05, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.max.s':    ('ATOM', 0x06, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.min.u':    ('ATOM', 0x07, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.max.u':    ('ATOM', 0x08, ('sd', 'mem_s', 'sb'),  'amo'),
    'atom.cas':      ('ATOM', 0x10, ('sd', 'mem_s', 'sb', 'sc'), 'amo'),
    'atom.ld':       ('ATOM', 0x18, ('sd', 'mem_s'),        'amo'),
    'atom.st':       ('ATOM', 0x19, ('mem_s', 'sb'),        'amo'),
    'fence':         ('ATOM', 0x20, (),                      'amo'),

    # ---- load/store (ISA sec 9.1) ------------------------------------------
    's.ld':          ('LSU', 0x00, ('sd', 'mem_si'),          'ls'),
    's.st':          ('LSU', 0x08, ('mem_si', 'sd'),          'ls'),
    'v.ld':          ('LSU', 0x10, ('vd', 'mem_si'),          'ls'),
    'v.st':          ('LSU', 0x18, ('mem_si', 'vd'),          'ls'),
    'v.ld.gather':   ('LSU', 0x20, ('vd', 'mem_sv'),          'ls'),
    'v.st.scatter':  ('LSU', 0x28, ('mem_sv', 'vd'),          'ls'),
}

# Instructions that may carry a lane-mask prefix (ISA sec 2.2).
MASKABLE_MAJORS = {MAJ['VINT'], MAJ['VFP'], MAJ['CVT'], MAJ['SHUF'],
                   MAJ['RED'], MAJ['SFU'], MAJ['LSU'], MAJ['ATOM'],
                   MAJ['TXFER'], MAJ['ACC']}

# Float format codes (ISA sec 8.4). The same numbering as xls/tnca_formats.x
# and tools/model/floatfmt.py, so the assembler, the generated RTL and the
# exact oracle cannot disagree about what a 3 means.
FMT      = {'f32': 0, 'bf16': 1, 'f16': 2, 'e5m2': 3, 'e4m3': 4}
FMT_NAME = {v: k for k, v in FMT.items()}

# Atomic modifiers (ISA sec 9.2).
AMO_ORDER = {'relaxed': 0, 'acquire': 1, 'release': 2, 'acq_rel': 3}
AMO_SCOPE = {'core': 0, 'device': 1, 'system': 2}
AMO_ORDER_NAME = {v: k for k, v in AMO_ORDER.items()}
AMO_SCOPE_NAME = {v: k for k, v in AMO_SCOPE.items()}

# Matrix K and input type suffixes (ISA sec 9C). The dtype field is the same
# five-format vocabulary minus binary32, which a fragment cannot be.
MMA_K    = {'k16': 0, 'k32': 1, 'k64': 2}
MMA_DT   = {'f16': 0, 'bf16': 1, 'e5m2': 2, 'e4m3': 3}


def encode(major, func, d=0, a=0, b=0, c=0, pred=0, pn=0, mod=0, x=0):
    """Pack fields into the 64-bit instruction word."""
    for name, val, width in (('d', d, 8), ('a', a, 8), ('b', b, 8), ('c', c, 8),
                             ('pred', pred, 4), ('mod', mod, 10),
                             ('major', major, 8), ('func', func, 8)):
        if not (0 <= val < (1 << width)):
            raise ValueError(f"field {name}={val} does not fit in {width} bits")
    return ((major << SH_MAJOR) | (func << SH_FUNC) | (d << SH_D) | (a << SH_A) |
            (b << SH_B) | (c << SH_C) | (pred << SH_PRED) | (x << SH_X) |
            (pn << SH_PN) | (mod << SH_MOD))


def decode_fields(word):
    """Unpack a 64-bit instruction word into its named fields."""
    return {
        'major': (word >> SH_MAJOR) & 0xff,
        'func':  (word >> SH_FUNC) & 0xff,
        'd':     (word >> SH_D) & 0xff,
        'a':     (word >> SH_A) & 0xff,
        'b':     (word >> SH_B) & 0xff,
        'c':     (word >> SH_C) & 0xff,
        'pred':  (word >> SH_PRED) & 0xf,
        'x':     (word >> SH_X) & 1,
        'pn':    (word >> SH_PN) & 1,
        'mod':   (word >> SH_MOD) & 0x3ff,
        'imm16': (word >> SH_C) & 0xffff,
        'imm24': (word >> SH_C) & 0xffffff,
    }
