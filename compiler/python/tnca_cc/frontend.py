"""
Triton frontend: TTIR -> the TNCA IR.

compiler_overview.md sec 3.1 puts the Triton frontend in the first category --
直接复用，尽量不修改 -- so nothing here reimplements Python tracing, tensor
semantics or TTIR construction. Triton builds the module; this walks it.

The walk uses Triton's own traversal API rather than parsing the printed IR,
so operand identity comes from the compiler rather than from matching SSA
names in text.

Two rules govern what is *not* here:

  sec 11.1  资源不够可以先明确报错，不能为了快速"支持"而偷偷破坏异步生命周期.
            An operation with no lowering raises, naming the operation, rather
            than being skipped.

  sec 12    the lowering boundaries: a tensor wider than one register, a
            64-bit tensor pointer, gather offsets past signed 32-bit, an
            absent SFU, absent atomics. Each is a diagnostic, never a quiet
            substitution.
"""
import re

from .diagnostics import Diagnostic
from .ir import Builder, make_kernel

# TTIR arithmetic -> our binary op names.
_INT_BIN = {'arith.addi': 'add', 'arith.subi': 'sub', 'arith.muli': 'mul',
            'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor',
            'arith.shli': 'shl', 'arith.maxsi': 'max', 'arith.minsi': 'min'}
_FP_BIN = {'arith.addf': 'add', 'arith.subf': 'sub', 'arith.mulf': 'mul',
           'arith.maximumf': 'max', 'arith.minimumf': 'min'}

# Predicates as MLIR spells them in arith.cmpi/cmpf attributes.
_CMP_I = {0: 'eq', 1: 'ne', 2: 'lt', 3: 'le', 4: 'gt', 5: 'ge',
          6: 'ltu', 7: 'leu', 8: 'gtu', 9: 'geu'}
_CMP_F = {1: 'eq', 2: 'gt', 3: 'ge', 4: 'lt', 5: 'le', 6: 'ne'}

_TENSOR_RE = re.compile(r'tensor<(\d+)x(.+)>')
_PTR_RE = re.compile(r'!tt\.ptr<(.+?)>')

_DTYPE_BYTES = {'f32': 4, 'i32': 4, 'f16': 2, 'bf16': 2, 'i16': 2,
                'i8': 1, 'i64': 8, 'f64': 8}
_SIZE_SUFFIX = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}


def _parse_type(text):
    """(shape, element type) for a TTIR type string."""
    m = _TENSOR_RE.fullmatch(text.strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, text.strip()


def _pointee(text):
    m = _PTR_RE.fullmatch(text.strip())
    return m.group(1).strip() if m else None


class Unsupported(Diagnostic):
    """A TTIR construct with no TNCA lowering yet."""


class TTIRImporter:
    """Walk a TTIR module and build the TNCA IR for it."""

    def __init__(self, target, name='kernel'):
        self.target = target
        self.kernel = make_kernel(name)
        self.b = Builder(self.kernel, self.kernel.role(0).region)
        self.name = name
        self._v = {}            # TTIR value id -> our Value
        self._const = {}        # TTIR value id -> python constant
        self._ptr_base = {}     # TTIR value id -> (arg name, element type)
        self._offsets = {}      # TTIR value id -> our offset Value (elements)
        self._args = {}         # arg index -> (name, our Value)
        self._pending_const = {}  # TTIR value id -> constant not yet emitted
        self._scalar_cache = {}   # constant -> our Value, so BLOCK is emitted once
        self._byte_offsets = {}   # (offset value, shift) -> byte offset Value
        self.arg_names = []

    # ---- entry point ------------------------------------------------------
    def run(self, module, func_name, arg_values):
        """`arg_values` maps parameter name -> a literal the kernel is
        compiled for (an address for a pointer, an integer for an i32)."""
        # MLIR's get_function returns a null handle for a name that is not in
        # the module, and the next call on it segfaults the interpreter with no
        # message at all. A wrong kernel name is an ordinary mistake -- the
        # Triton function is `add_kernel` and someone writes `add` -- and it
        # has to produce a diagnostic, not a crash with nothing to read.
        if not module.has_function(func_name):
            raise Diagnostic(
                f"no function {func_name!r} in the module",
                because='the name must match the @triton.jit function exactly',
                remedy='pass the kernel function name, e.g. "add_kernel"',
                source='compiler_overview.md sec 11.1')
        fn = module.get_function(func_name)
        self._bind_args(fn, arg_values)

        ops = []
        module.walk(lambda op: ops.append(op))
        self._float_constants(module, ops)

        for op in ops:
            name = op.get_name()
            if name in ('tt.func', 'builtin.module', 'tt.return'):
                continue
            handler = getattr(self, '_op_' + name.replace('.', '_'), None)
            if handler is None:
                raise Unsupported(
                    f"no TNCA lowering for TTIR operation {name!r}",
                    because='every operation must lower explicitly; skipping '
                            'one silently changes the program',
                    remedy='add a lowering, or reject the kernel',
                    source='compiler_overview.md sec 11.1, sec 12')
            handler(op)

        self.b.exit()
        return self.kernel, self.b

    # ---- arguments --------------------------------------------------------
    def _bind_args(self, fn, arg_values):
        for i in range(fn.get_num_args()):
            arg = fn.args(i)
            ty = str(arg.get_type())
            name = list(arg_values)[i] if i < len(arg_values) else f"arg{i}"
            self.arg_names.append(name)
            value = arg_values[name]
            v = self.b.const(value)
            self._v[arg.id()] = v
            self._args[i] = (name, v)
            if _pointee(ty):
                self._ptr_base[arg.id()] = (name, _pointee(ty))

    # ---- constants --------------------------------------------------------
    def _float_constants(self, module, ops):
        """Pair float constants with their printed values.

        get_constant_value() returns integers but not floats, so the printed
        module supplies those. The pairing is checked against the integers it
        can read: if the two disagree the order is wrong and we refuse rather
        than guess.
        """
        text = module.str_nodebug() if hasattr(module, 'str_nodebug') \
            else module.str()
        printed = re.findall(r'=\s*arith\.constant\s+(\S+)\s*:', text)
        walked = [op for op in ops if op.get_name() == 'arith.constant']
        if len(printed) != len(walked):
            self._printed_consts = None
            return
        self._printed_consts = {}
        for op, literal in zip(walked, printed):
            self._printed_consts[op.get_result(0).id()] = literal
            iv = op.get_constant_value()
            if iv is not None:
                m = re.fullmatch(r'(-?\d+)', literal)
                if m and int(m.group(1)) != iv:
                    raise Diagnostic(
                        'constant order from the printed module does not '
                        'match the walk; refusing to guess float values',
                        source='compiler_overview.md sec 11.1')

    @staticmethod
    def _literal_value(literal, elem):
        lit = literal.strip()
        m = re.fullmatch(r'dense<(.+)>', lit)
        if m:
            lit = m.group(1)
        if elem.startswith('f'):
            import struct
            return struct.unpack('<I', struct.pack('<f', float(lit)))[0]
        return int(float(lit)) if '.' in lit or 'e' in lit.lower() else int(lit)

    def _op_arith_constant(self, op):
        rid = op.get_result(0).id()
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        iv = op.get_constant_value()
        if iv is None:
            printed = (self._printed_consts or {}).get(rid)
            if printed is None:
                raise Unsupported(
                    f"cannot read the value of an {elem} constant",
                    source='compiler_overview.md sec 11.1')
            iv = self._literal_value(printed, elem)
        self._const[rid] = iv
        # Materialised lazily. A tt.load's `other` operand is a constant that
        # becomes an attribute of our load rather than a value, and emitting
        # it eagerly would leave a dead `.li` and a dead register in every
        # masked access.
        if shape is None:
            self._pending_const[rid] = ('scalar', iv, elem)
        else:
            self._check_block(shape)
            self._pending_const[rid] = ('vector', iv, elem)

    # ---- shape rules ------------------------------------------------------
    def _check_block(self, shape):
        vlen = self.target.vlen
        if shape % vlen:
            raise Unsupported(
                f"a block of {shape} is not a multiple of VLEN={vlen}",
                because='a partial register would need a lane mask on every '
                        'operation, not just the boundary ones',
                remedy=f"use a multiple of {vlen}",
                source='compiler_overview.md sec 12')
        if shape != vlen:
            raise Unsupported(
                f"a block of {shape} spans {shape // vlen} vector registers",
                because='sec 12 wants a wide tensor split across registers '
                        'with its index mapping intact; that is not built yet',
                remedy=f"use BLOCK={vlen} for now",
                source='compiler_overview.md sec 12')

    # ---- TTIR operations --------------------------------------------------
    def _op_tt_get_program_id(self, op):
        axis = op.get_int_attr('axis') or 0
        self._v[op.get_result(0).id()] = self.b.program_id(axis)

    def _op_tt_make_range(self, op):
        shape, _ = _parse_type(str(op.get_result(0).get_type()))
        self._check_block(shape)
        v = self.b.arange()
        self._v[op.get_result(0).id()] = v
        self._offsets[op.get_result(0).id()] = ('arange', None)

    def _op_tt_splat(self, op):
        src = op.get_operand(0)
        rid = op.get_result(0).id()
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        self._check_block(shape)
        if _pointee(elem):
            # A splat pointer is a uniform base. sec 12: recognise a uniform
            # base with a bounded offset rather than materialising 16 pointers.
            if src.id() not in self._ptr_base:
                raise Unsupported('splat of a non-argument pointer')
            self._ptr_base[rid] = self._ptr_base[src.id()]
            return
        self._v[rid] = self.b.splat(self._value(src), dtype=elem)
        self._offsets[rid] = ('uniform', src.id())

    def _op_tt_addptr(self, op):
        base, offs = op.get_operand(0), op.get_operand(1)
        rid = op.get_result(0).id()
        if base.id() not in self._ptr_base:
            raise Unsupported(
                'a pointer that is not a uniform base plus an offset',
                because='sec 12 asks for uniform base + bounded offset to be '
                        'recognised; a general tensor of pointers would need '
                        'per-lane addressing',
                source='compiler_overview.md sec 12')
        self._ptr_base[rid] = self._ptr_base[base.id()]
        self._offsets[rid] = ('ptr', offs.id())

    def _op_arith_addi(self, op):
        self._binary(op, _INT_BIN['arith.addi'])

    def _op_arith_subi(self, op):
        self._binary(op, 'sub')

    def _op_arith_muli(self, op):
        self._binary(op, 'mul')

    def _op_arith_andi(self, op):
        self._binary(op, 'and')

    def _op_arith_ori(self, op):
        self._binary(op, 'or')

    def _op_arith_xori(self, op):
        self._binary(op, 'xor')

    def _op_arith_addf(self, op):
        self._binary(op, 'add')

    def _op_arith_subf(self, op):
        self._binary(op, 'sub')

    def _op_arith_mulf(self, op):
        self._binary(op, 'mul')

    def _op_arith_maximumf(self, op):
        self._binary(op, 'max')

    def _op_arith_minimumf(self, op):
        self._binary(op, 'min')

    def _binary(self, op, kind):
        a, b = op.get_operand(0), op.get_operand(1)
        rid = op.get_result(0).id()
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        if shape is None:
            self._v[rid] = self.b.sbin(kind, self._value(a), self._value(b))
        else:
            self._check_block(shape)
            if elem.startswith('f'):
                self.target.require('vector_fp32', f"emit a {elem} {kind}")
            self._v[rid] = self.b.vbin(kind, self._value(a), self._value(b),
                                       dtype=elem)
            # Track `pid*BLOCK + arange` so a contiguous access is recognised.
            self._note_affine(rid, a, b, kind)

    def _note_affine(self, rid, a, b, kind):
        if kind != 'add':
            return
        ka = self._offsets.get(a.id(), (None, None))[0]
        kb = self._offsets.get(b.id(), (None, None))[0]
        if {ka, kb} == {'uniform', 'arange'}:
            uni = a if ka == 'uniform' else b
            self._offsets[rid] = ('contiguous', self._offsets[uni.id()][1])

    def _op_arith_cmpi(self, op):
        self._compare(op, _CMP_I, 'i')

    def _op_arith_cmpf(self, op):
        self._compare(op, _CMP_F, 'f')

    def _compare(self, op, table, which):
        a, b = op.get_operand(0), op.get_operand(1)
        pred = op.get_int_attr('predicate')
        cc = table.get(pred)
        if cc is None:
            raise Unsupported(f"comparison predicate {pred} on {which}")
        self._v[op.get_result(0).id()] = self.b.vcmp(cc, self._value(a),
                                                     self._value(b))

    # ---- memory -----------------------------------------------------------
    def _address(self, ptr_value):
        """A scalar byte address for a recognised uniform-base access.

        Returns (scalar base register value, size suffix, element type, kind)
        where kind is 'contiguous' or 'indexed'.
        """
        pid = ptr_value.id()
        name, elem = self._ptr_base[pid]
        esize = _DTYPE_BYTES.get(elem)
        if esize is None:
            raise Unsupported(f"element type {elem!r}")
        kind, src = self._offsets.get(pid, (None, None))
        if kind != 'ptr':
            raise Unsupported('an access whose offset was not tracked')
        okind, ouni = self._offsets.get(src, (None, None))

        base = self._arg_value(name)
        if okind == 'contiguous':
            # base + uniform*esize, lane i at +i*esize: exactly what a plain
            # vector load does (ISA sec 9.1). No gather needed.
            shift = {1: 0, 2: 1, 4: 2, 8: 3}[esize]
            # Every pointer in a kernel shares the same block offset, so this
            # is computed once rather than once per access. Not a general CSE
            # -- sec 11.1 says not to build one here -- just not emitting the
            # same shift three times for three pointers.
            key = (ouni, shift)
            byte = self._byte_offsets.get(key)
            if byte is None:
                byte = self.b.sbini(
                    'shl', self._v[ouni], shift,
                    comment=f"element offset -> bytes (x{esize})")
                self._byte_offsets[key] = byte
            addr = self.b.sbin('add', base, byte, comment=f"&{name}[block]")
            return addr, _SIZE_SUFFIX[esize], elem, 'contiguous', None
        if okind == 'arange':
            return base, _SIZE_SUFFIX[esize], elem, 'contiguous', None
        # Anything else is per-lane. ISA sec 9.1 scales the index by the
        # access size, so the offset tensor is already in elements.
        return base, _SIZE_SUFFIX[esize], elem, 'indexed', self._v[src]

    def _arg_value(self, name):
        for _, (n, v) in self._args.items():
            if n == name:
                return v
        raise Diagnostic(f"no kernel argument {name!r}")

    def _op_tt_load(self, op):
        ptr = op.get_operand(0)
        mask = self._value(op.get_operand(1)) if op.get_num_operands() > 1 \
            else None
        other = 0
        if op.get_num_operands() > 2:
            other = self._const.get(op.get_operand(2).id(), 0)
        addr, size, elem, kind, index = self._address(ptr)
        rid = op.get_result(0).id()
        if kind == 'contiguous':
            self._v[rid] = self.b.load(addr, mask=mask, other=other,
                                       dtype=elem, size=size)
        else:
            self.target.require('gather_scatter', 'emit v.ld.gather')
            self._v[rid] = self.b.gather(addr, index, mask=mask, other=other,
                                         dtype=elem, size=size)

    def _op_tt_store(self, op):
        ptr, value = op.get_operand(0), op.get_operand(1)
        mask = self._value(op.get_operand(2)) if op.get_num_operands() > 2 \
            else None
        addr, size, elem, kind, index = self._address(ptr)
        if kind == 'contiguous':
            self.b.store(addr, self._value(value), mask=mask, size=size)
        else:
            self.target.require('gather_scatter', 'emit v.st.scatter')
            self.b.scatter(addr, index, self._value(value), mask=mask,
                           size=size)

    # ---- helpers ----------------------------------------------------------
    def _value(self, ttir_value):
        vid = ttir_value.id()
        v = self._v.get(vid)
        if v is not None:
            return v
        pending = self._pending_const.get(vid)
        if pending is not None:
            kind, iv, elem = pending
            if kind == 'scalar':
                # One register per distinct value: BLOCK appears once per use
                # in TTIR, and three `.li`s of 16 is three registers held for
                # nothing.
                v = self._scalar_cache.get(iv)
                if v is None:
                    v = self.b.const(iv)
                    self._scalar_cache[iv] = v
            else:
                v = self.b.vmovi(iv, dtype=elem)
            self._v[vid] = v
            return v
        raise Unsupported(
            f"a value produced by an operation that was not lowered "
            f"(type {ttir_value.get_type()})")
