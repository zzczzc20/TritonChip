"""
The ISA, as the compiler sees it.

compiler_overview.md sec 11.3: one machine-readable source feeds the
assembler/disassembler tables, the compiler's opcode definitions, encoding
legality, later TableGen input and the conformance tests -- and explicitly not
a third hand-maintained set of numbers for the compiler.

So this module reads isa/tnca-isa-v0.1.json, which tools/isa/export_schema.py
generates from tools/asm/tnca_isa.py, which the assembler and the RTL tests
already exercise. tools/isa/check_consistency.py keeps the hand-written Verilog
defines honest against the same tables.
"""
import json
import os

from ._datapath import ROOT, find

SCHEMA_PATH = find('isa', 'tnca-isa-v0.1.json')


class Instruction:
    """One encodable instruction."""

    __slots__ = ('mnemonic', 'major', 'major_code', 'func', 'operands',
                 'suffix_class', 'maskable')

    def __init__(self, d):
        self.mnemonic = d['mnemonic']
        self.major = d['major']
        self.major_code = d['major_code']
        self.func = d['func']
        self.operands = tuple(d['operands'])
        self.suffix_class = d['suffix_class']
        self.maskable = d['maskable']

    @property
    def opcode(self):
        return (self.major_code, self.func)

    def __repr__(self):
        return (f"<Instruction {self.mnemonic} "
                f"{self.major_code:#04x}/{self.func:#04x}>")


class _ISA:
    def __init__(self, path=SCHEMA_PATH):
        with open(path) as f:
            self._d = json.load(f)
        self._by_mnemonic = {i['mnemonic']: Instruction(i)
                             for i in self._d['instructions']}
        self._by_opcode = {i.opcode: i for i in self._by_mnemonic.values()}

    # ---- lookup -----------------------------------------------------------
    def __getitem__(self, mnemonic):
        try:
            return self._by_mnemonic[mnemonic]
        except KeyError:
            raise KeyError(f"no TNCA instruction {mnemonic!r}") from None

    def __contains__(self, mnemonic):
        return mnemonic in self._by_mnemonic

    def __iter__(self):
        return iter(self._by_mnemonic.values())

    def __len__(self):
        return len(self._by_mnemonic)

    def by_opcode(self, major, func):
        return self._by_opcode.get((major, func))

    def in_major(self, major_name):
        return [i for i in self._by_mnemonic.values() if i.major == major_name]

    # ---- encoding facts ---------------------------------------------------
    @property
    def version(self):
        return self._d['isa_version']

    @property
    def fields(self):
        return dict(self._d['word']['fields'])

    @property
    def majors(self):
        return dict(self._d['majors'])

    @property
    def special_registers(self):
        return dict(self._d['special_registers'])

    @property
    def trap_codes(self):
        return dict(self._d['trap_codes'])

    @property
    def condition_codes(self):
        return dict(self._d['condition_codes'])

    def operand_kind(self, kind):
        return dict(self._d['operand_kinds'][kind])

    # The capability each encoding depends on. A mnemonic absent here needs
    # none. Keyed by major because CAPABILITY0 gates whole encoding groups
    # (ISA sec 11), not individual functions.
    _MAJOR_CAPABILITY = {
        'SFU': 'sfu',
        'ATOM': 'atomics_device',
        'TXFER': 'tensor_dma',
        'THINT': 'tensor_dma',
        'TDESC': 'tensor_dma',
        'MBAR': 'mbarrier',
        'VFP': 'vector_fp32',
    }

    def required_capability(self, mnemonic):
        """The CAPABILITY0 feature `mnemonic` needs, or None.

        The compiler consults this before emitting, so a missing feature
        becomes a diagnostic at compile time instead of an
        ILLEGAL_INSTRUCTION trap at run time (sec 12).
        """
        ins = self[mnemonic]
        if mnemonic in ('v.ld.gather', 'v.st.scatter'):
            return 'gather_scatter'
        return self._MAJOR_CAPABILITY.get(ins.major)


ISA = _ISA()
