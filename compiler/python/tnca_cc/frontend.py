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
import struct

from . import mathlib
from .diagnostics import Diagnostic
from .ir import Builder, VECTOR, make_kernel
from .tiles import (Affine2D, Tile, TileError, TENSORMAP_BYTES,
                    check_geometry, tensormap_words)

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

_TENSOR_RE = re.compile(r'tensor<((?:\d+x)+)(.+)>')
_PTR_RE = re.compile(r'!tt\.ptr<(.+?)>')

_DTYPE_BYTES = {'f32': 4, 'i32': 4, 'f16': 2, 'bf16': 2, 'i16': 2,
                'i8': 1, 'i64': 8, 'f64': 8}
_SIZE_SUFFIX = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}


def _parse_type(text):
    r"""(shape, element type) for a TTIR type string.

    A 1-D tensor reports its length as an int, which is what every
    caller here wants and what the whole importer was written against.
    A tensor of rank 2 or more reports its dimensions as a tuple, so
    that `_check_block` can refuse it by name.

    The distinction is not cosmetic. The pattern used to be
    `tensor<(\d+)x(.+)>`, and `tensor<16x16xf16>` matched it -- as a
    16-element tensor of element type "16xf16". Nothing downstream
    looked at the element type closely enough to notice, so a 2-D block
    was silently compiled as the first row of itself.
    """
    m = _TENSOR_RE.fullmatch(text.strip())
    if m:
        dims = tuple(int(d) for d in m.group(1).rstrip('x').split('x'))
        return (dims[0] if len(dims) == 1 else dims), m.group(2).strip()
    return None, text.strip()


def _pointee(text):
    m = _PTR_RE.fullmatch(text.strip())
    return m.group(1).strip() if m else None


class _Descriptor:
    """A TensorMap this compiler built, and what it says.

    The slot alone is not enough: an accumulator drains to memory
    through the vector unit rather than through a transfer, and doing
    that needs the base and the row stride that went INTO the map, not
    a handle to the snapshot of it.
    """

    __slots__ = ('value', 'base', 'byte_strides', 'block', 'elem')

    def __init__(self, value, *, base, byte_strides, block, elem):
        self.value = value
        self.base = base
        self.byte_strides = list(byte_strides)
        self.block = list(block)
        self.elem = elem


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
        # TTIR pointer id -> the offsets added to it, oldest first. The
        # offset table alone cannot say which pointer an offset belonged
        # to, and reading only the last link is what made `p + row*N +
        # offs` load from `p + offs`.
        self._ptr_chain = {}
        self._offsets = {}  # TTIR value id -> our offset Value (elements)
        self._bodies = {}   # region id -> the operations inside that region
        self._args = {}         # arg index -> (name, our Value)
        self._pending_const = {}  # TTIR value id -> constant not yet emitted
        self._scalar_cache = {}   # constant -> our Value, so BLOCK is emitted once
        self._byte_offsets = {}   # (offset value, shift) -> byte offset Value
        self._fconst_cache = {}   # (f32 bit pattern, type) -> our Value
        self._affine = {}   # TTIR value id -> Affine2D, for a 2-D index
        self._ptr2d = {}    # TTIR value id -> (arg name, elem, Affine2D)
        self._tiles = {}    # TTIR value id -> Tile
        self._descs = {}    # TTIR value id -> our descriptor Value
        self._scratch = None      # the descriptor scratch base, read once
        self.scratch_maps = 0     # how many TensorMaps this kernel builds
        self.spill_bytes = 0      # scratch beyond them, for tile repacks
        self.arg_names = []
        self.arg_abi_types = []   # parallel to arg_names, in dynamic mode
        self.dynamic_args = False  # set by run(); see _load_argument
        self._fn = None           # the tt.func being lowered

    # ---- entry point ------------------------------------------------------
    def run(self, module, func_name, arg_values, *, param_names=None,
            dynamic_args=False):
        """Lower ONE function of a TTIR module.

        `arg_values` maps parameter name -> a literal the kernel is compiled
        for (an address for a pointer, an integer for an i32). `param_names`
        is the function's DECLARED parameter order -- from the @triton.jit
        function's `arg_names`, minus the constexprs, which do not become
        MLIR arguments. Without it the binding falls back to the order the
        caller happened to build the dict in, which is the audit's F07: two
        dicts with the same contents and different insertion order compiled
        to two different kernels, silently.

        `dynamic_args` switches parameter binding from "materialise the
        literal" to "load the slot from the argument block". The literals are
        still required and still type-checked, because the compiler needs to
        know the SHAPE of each argument -- but in dynamic mode no literal
        reaches the code, which is what lets one artifact run with different
        addresses and scalars (R2-06 ABI-02).
        """
        self.dynamic_args = bool(dynamic_args)
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
        self._fn = fn
        self._bind_args(fn, func_name, arg_values, param_names, module)

        # ONLY this function's operations.
        #
        # `module.walk` visits every operation in the module. On a
        # single-function module that is the same set; on a module with a
        # second `tt.func` -- a @triton.jit helper called with noinline=True,
        # say -- it is not, and the extra operations were lowered into this
        # kernel as though they belonged to it. The walk does not report which
        # function it is in, so membership is decided by region identity:
        # climb from the operation's block out to the function's own region.
        ops = []
        module.walk(lambda op: ops.append(op))
        ops = [op for op in ops if self._in_function(op, fn)]
        if not ops:
            raise Diagnostic(
                f"function {func_name!r} has no operations to lower",
                because='an empty body means the walk found nothing inside '
                        'this function, not that the kernel is trivial',
                source='compiler_overview.md sec 11.1')
        self._float_constants(module, ops)

        ops, self._bodies = self._split_regions(ops)

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

    @staticmethod
    def _constant_value(op):
        """`op.get_constant_value()`, where the binding has it.

        It does not exist before Triton 3.8, and calling it raised
        AttributeError -- an interpreter error where the importer already had
        a working fallback: the printed module, which is where float
        constants come from anyway. The locked version has the method; this
        keeps an older one from failing in a way that looks like a compiler
        bug rather than a missing binding.
        """
        fn = getattr(op, 'get_constant_value', None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            return None

    # ---- function membership ----------------------------------------------
    @staticmethod
    def _in_function(op, fn):
        """Is `op` inside `fn`'s body?

        Region identity, climbing outwards. An operation in an `scf.for` body
        is two regions deep and still belongs to the function; an operation in
        a different `tt.func` never reaches this region at all.
        """
        try:
            target = fn.get_region(0).id()
            region = op.get_block().get_parent()
        except Exception:
            # A binding without the accessors. Refusing would be wrong -- the
            # single-function case was always correct -- but claiming the
            # filter ran would be worse, so say which one happened.
            return True
        seen = 0
        while region is not None and seen < 64:
            if region.id() == target:
                return True
            try:
                region = region.get_parent_region()
            except Exception:
                return False
            seen += 1
        return False

    @staticmethod
    def _split_regions(ops):
        """Separate the function body from the bodies of region operations.

        `module.walk` does not stop at a region boundary: the combinator
        inside a `tt.reduce` arrives in the same flat list as the reduce
        that owns it. Dispatched there it would be lowered twice -- once
        as part of the reduction, and once as a stray scalar
        `arith.maxnumf` over two block arguments that were never bound to
        anything, which is a wrong answer rather than an error.

        Returns (top-level ops, {region id: [ops inside it]}). The bodies
        stay in walk order, which is program order, because the operation
        that owns them has to read them as a sequence.
        """
        owned = set()
        for op in ops:
            if op.get_name() in ('tt.func', 'builtin.module'):
                continue
            for i in range(op.get_num_regions()):
                owned.add(op.get_region(i).id())
        if not owned:
            return ops, {}

        top, bodies = [], {}
        for op in ops:
            rid = None
            try:
                region = op.get_block().get_parent()
            except Exception:                             # noqa: BLE001
                region = None
            seen = 0
            while region is not None and seen < 64:
                if region.id() in owned:
                    rid = region.id()
                    break
                try:
                    region = region.get_parent_region()
                except Exception:                         # noqa: BLE001
                    break
                seen += 1
            if rid is None:
                top.append(op)
            else:
                bodies.setdefault(rid, []).append(op)
        return top, bodies

    # ---- reductions -------------------------------------------------------
    # The combinator inside a `tt.reduce` region, as the reduction it is.
    # `maximumf`/`minimumf` are deliberately absent: they propagate NaN,
    # and ISA sec 8.7 does not say what `vf.reduce.max` does with one.
    # Mapping them onto it anyway would be a guess about the hardware
    # written into the compiler.
    _COMBINATOR = {
        'arith.addf': 'add', 'arith.addi': 'add',
        'arith.maxnumf': 'max', 'arith.minnumf': 'min',
        'arith.maxsi': 'max', 'arith.minsi': 'min',
        'arith.maxui': 'maxu', 'arith.minui': 'minu',
        'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor',
    }

    def _op_tt_reduce(self, op):
        """A whole-vector reduction to a scalar.

        The masked lanes need no handling here. Triton has already put
        `other=` into the loaded vector, and ISA sec 8.7 makes an
        inactive lane contribute the operation's identity -- which is the
        same value. Adding a second mask would be harmless for `add` and
        wrong for nothing, but it would also be untrue: the mask this
        reduction would want is the load's, and by this point the load
        has consumed it.
        """
        if op.get_num_operands() != 1 or op.get_num_results() != 1:
            raise Unsupported(
                f"a {op.get_num_operands()}-input reduction",
                because='a multi-input `tt.reduce` carries an index '
                        'alongside the value -- argmax and argmin -- and '
                        'ISA sec 8.7 reduces one vector to one scalar',
                remedy='reduce the value and the index separately, or add '
                       'an argmax lowering',
                source='doc/isa/TNCA-ISA-v0.1.md sec 8.7')

        axis = op.get_int_attr('axis')
        src = op.get_operand(0)
        shape, elem = _parse_type(str(src.get_type()))
        if isinstance(shape, tuple):
            if axis >= len(shape):
                raise Unsupported(
                    f"a reduction along axis {axis} of a {len(shape)}-D "
                    f"tile")
            _, rtype = _parse_type(str(op.get_result(0).get_type()))
            self._op_tt_reduce_2d(op, self._combinator(op), rtype or elem)
            return
        if shape is None:
            raise Unsupported('a reduction over a scalar')
        rank = len(shape) if isinstance(shape, tuple) else 1
        if rank != 1 or axis != 0:
            raise Unsupported(
                f"a reduction over axis {axis} of a {rank}-D tensor",
                because='only the lane axis of a 1-D block reduces to a '
                        'scalar in one instruction',
                source='doc/isa/TNCA-ISA-v0.1.md sec 8.7')

        kind = self._combinator(op)
        _, rtype = _parse_type(str(op.get_result(0).get_type()))
        rtype = rtype or elem
        self._v[op.get_result(0).id()] = self.b.reduce(
            kind, self._value(src), dtype=rtype,
            broadcast=rtype.startswith('f'))

    def _combinator(self, op):
        """Which reduction the region body computes.

        Read out of the region rather than assumed: `tt.reduce` is a
        generic fold and the body is the only thing that says whether it
        is a sum, a maximum or something this backend has no instruction
        for. Guessing from the operator name would be guessing from the
        Python that produced it, which the IR no longer contains.
        """
        body = self._bodies.get(op.get_region(0).id(), [])
        body = [b for b in body if b.get_name() != 'tt.reduce.return']
        rets = self._bodies.get(op.get_region(0).id(), [])
        rets = [b for b in rets if b.get_name() == 'tt.reduce.return']

        if len(body) != 1 or len(rets) != 1:
            raise Unsupported(
                f"a reduction whose body is {len(body)} operations",
                because='a single ISA reduction computes a single '
                        'associative combinator; a longer body is a fold '
                        'that has to be written out as a loop',
                source='doc/isa/TNCA-ISA-v0.1.md sec 8.7')

        inner, ret = body[0], rets[0]
        name = inner.get_name()
        kind = self._COMBINATOR.get(name)
        if kind is None:
            raise Unsupported(
                f"a reduction combined with {name!r}",
                because='ISA sec 8.7 has add, max, min, and, or and xor, '
                        'and a combinator outside that set is not one '
                        'instruction',
                remedy='use tl.sum, tl.max or tl.min',
                source='doc/isa/TNCA-ISA-v0.1.md sec 8.7')

        # The body must fold its two block arguments and return that.
        # Anything else -- a captured outer value, a returned constant --
        # is a different function that happens to contain the right
        # operator, and lowering it as this reduction would be wrong.
        if inner.get_num_operands() != 2:
            raise Unsupported(f"a reduction combined with a "
                              f"{inner.get_num_operands()}-operand {name!r}")
        a, b = inner.get_operand(0).id(), inner.get_operand(1).id()
        if a == b or a in self._v or b in self._v:
            raise Unsupported(
                'a reduction whose body captures a value from outside it',
                because='the body must fold its own two arguments; one that '
                        'reads an outer value is not the fold this lowers',
                source='doc/isa/TNCA-ISA-v0.1.md sec 8.7')
        if (ret.get_num_operands() != 1
                or ret.get_operand(0).id() != inner.get_result(0).id()):
            raise Unsupported(
                'a reduction that returns something other than its '
                'combinator')
        return kind

    # ---- arguments --------------------------------------------------------
    def _bind_args(self, fn, func_name, arg_values, param_names, module):
        """Bind by the DECLARED signature, never by dict insertion order."""
        n = fn.get_num_args()

        if param_names is None:
            # Insertion order is not a signature. It is accepted only where it
            # cannot be wrong: one parameter, or none.
            if len(arg_values) > 1:
                raise Diagnostic(
                    f"no declared parameter order for {sorted(arg_values)}",
                    because='binding by dict insertion order made two calls '
                            'with identical arguments compile to different '
                            'kernels depending on the order the caller built '
                            'the dict in',
                    remedy='pass param_names: the @triton.jit function\'s '
                           'arg_names, minus the constexprs',
                    source='compiler_overview.md sec 11.1')
            param_names = list(arg_values)

        param_names = list(param_names)
        if len(param_names) != n:
            raise Diagnostic(
                f"{func_name!r} takes {n} argument(s), but "
                f"{len(param_names)} parameter name(s) were given: "
                f"{param_names}",
                because='the MLIR signature is the authority on how many '
                        'values the kernel takes, and a constexpr is not '
                        'among them',
                remedy='drop the constexprs from param_names',
                source='compiler_overview.md sec 11.1')

        missing = [p for p in param_names if p not in arg_values]
        extra = [k for k in arg_values if k not in param_names]
        if missing or extra:
            raise Diagnostic(
                f"argument mismatch: missing {missing}, unexpected {extra}",
                because='every declared parameter needs a value and nothing '
                        'else may be passed; an unexpected name is a typo '
                        'that would otherwise bind nothing and be ignored',
                source='compiler_overview.md sec 11.1')

        try:
            sig = module.get_function_signature(fn)
        except Exception:
            sig = [None] * n

        # The argument block's base, read once. Only in dynamic mode: a
        # kernel compiled for literals never touches ARG_BASE, and emitting a
        # read it does not use would be a difference between the two modes
        # that shows up in the code rather than in what the code does.
        arg_base = self.b.special('arg_base') if self.dynamic_args else None

        for i in range(n):
            arg = fn.args(i)
            ty = str(arg.get_type())
            name = param_names[i]
            value = arg_values[name]
            self._check_arg_type(name, ty, sig[i] if i < len(sig) else None,
                                 value, dynamic=self.dynamic_args)
            self.arg_names.append(name)
            if self.dynamic_args:
                v = self._load_argument(arg_base, i, name, ty)
            else:
                v = self.b.const(value)
            self._v[arg.id()] = v
            self._args[i] = (name, v)
            if _pointee(ty):
                self._ptr_base[arg.id()] = (name, _pointee(ty))

    # MLIR type -> (ABI type, the scalar load that reads its slot).
    #
    # ISA sec 9.1 mod[3] sign-extends an integer load, and that bit is the
    # whole reason `i32` reads as `.sw`: the ABI puts a 32-bit value in the
    # LOW four bytes with the high four ZERO, so a 64-bit load of -5 would
    # produce 4294967291. Reading the slot at the declared width, with the
    # declared signedness, is what makes the two halves of the contract --
    # the encoder and the kernel -- agree.
    _ARG_LOAD = {
        'ptr': ('device_ptr64', 'd'),
        'i64': ('i64', 'd'),
        'i32': ('i32', 'sw'),
        'f32': ('f32', 'w'),
    }

    def _load_argument(self, arg_base, index, name, ty):
        """Emit the load of one argument slot.

        doc/phase2_tasks_and_acceptance.md R2-06: the kernel loads dynamic
        parameters from the block at ARG_BASE. Embedding an address or a
        scalar in the code, or patching an instruction per launch, is
        forbidden -- and R2M-01 exists to catch a compiler that goes back to
        doing it.
        """
        kind = 'ptr' if _pointee(ty) else str(ty).strip()
        if kind not in self._ARG_LOAD:
            raise Unsupported(
                f"parameter {name!r} has type {ty!r}, which the v1 argument "
                "ABI does not carry",
                because='spec/r2_abi_v1.json lists the six types a slot may '
                        'hold',
                remedy='pass it through a buffer element',
                source='doc/phase2_tasks_and_acceptance.md R2-06')
        abi_type, size = self._ARG_LOAD[kind]
        self.arg_abi_types.append(abi_type)
        return self.b.sload(arg_base, offset=8 * index, size=size,
                            comment=f"{name}: {abi_type} at arg_base+"
                                    f"{8 * index}")

    @staticmethod
    def _check_arg_type(name, ty, sig_ty, value, *, dynamic=False):
        """The value must fit the declared parameter type.

        A float where the signature says i32 is not a narrowing the compiler
        may perform quietly: the kernel would be compiled for a different
        number from the one the caller passed.

        What "fits" means depends on the mode, and this is the one place the
        two differ. Compiling FOR a literal bakes the value into the code, so
        it has to be an address or an integer -- there is no instruction that
        materialises a float operand from an immediate here. Compiling for a
        DYNAMIC argument bakes in nothing; the value is a shape declaration,
        and an f32 parameter is exactly where a Python float belongs. Refusing
        it there rejected a kernel the frontend had already accepted, and said
        so by naming the type rather than the mode.
        """
        declared = (sig_ty or ty or '').strip()
        is_float = declared in ('f32', 'fp32', 'float32')

        if dynamic and is_float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Diagnostic(
                    f"argument {name!r} is {type(value).__name__} {value!r}, "
                    f"and the kernel declares it {declared}",
                    because='an f32 parameter takes a number',
                    source='doc/phase2_tasks_and_acceptance.md R2-06')
            return

        if isinstance(value, bool) or not isinstance(value, int):
            raise Diagnostic(
                f"argument {name!r} is {type(value).__name__} {value!r}, "
                f"and the kernel declares it {sig_ty or ty}",
                because=('a kernel compiled for literal arguments takes an '
                         'address or an integer, never a float or an object'
                         if not dynamic else
                         'this parameter is not a float in the signature'),
                remedy=('pass an integer address for a pointer and an '
                        'integer for an i32' if not dynamic else
                        'check the signature: only an f32 parameter takes a '
                        'float'),
                source='compiler_overview.md sec 11.1')
        if _pointee(ty) and value < 0:
            raise Diagnostic(
                f"argument {name!r} is a pointer and was given {value}",
                because='a negative address is not one',
                source='compiler_overview.md sec 11.1')

    # ---- constants --------------------------------------------------------
    def _float_constants(self, module, ops):
        """Pair float constants with their printed values.

        get_constant_value() returns integers but not floats, so the printed
        module supplies those. The pairing is checked against the integers it
        can read: if the two disagree the order is wrong and we refuse rather
        than guess.
        """
        # THIS function's text, not the module's. `ops` is already filtered
        # to the target function, so pairing it against a module-wide list of
        # printed constants would line up the wrong ones as soon as a second
        # tt.func existed -- and the check below would then reject a perfectly
        # good kernel for "constant order does not match".
        fn = self._fn
        if fn is not None and hasattr(fn, 'str_nodebug'):
            text = fn.str_nodebug()
        elif hasattr(module, 'str_nodebug'):
            text = module.str_nodebug()
        else:
            text = module.str()
        printed = re.findall(r'=\s*arith\.constant\s+(\S+)\s*:', text)
        walked = [op for op in ops if op.get_name() == 'arith.constant']
        if len(printed) != len(walked):
            self._printed_consts = None
            return
        self._printed_consts = {}
        for op, literal in zip(walked, printed):
            self._printed_consts[op.get_result(0).id()] = literal
            iv = self._constant_value(op)
            if iv is not None:
                m = re.fullmatch(r'(-?\d+)', literal)
                if m and int(m.group(1)) != iv:
                    raise Diagnostic(
                        'constant order from the printed module does not '
                        'match the walk; refusing to guess float values',
                        source='compiler_overview.md sec 11.1')

    @staticmethod
    def _literal_value(literal, elem):
        """One printed MLIR literal -> the bits the hardware will hold.

        MLIR prints a float whose decimal form would not round-trip as a raw
        hexadecimal bit pattern: NaN is `0x7FC00000`, positive infinity is
        `0x7F800000`. Those went through `float(lit)`, which raises
        ValueError -- an interpreter traceback where the contract asks for a
        diagnostic, and the sort of thing that later gets "fixed" by a
        fallback that reads the value as something else.

        Two rules here. A hexadecimal pattern is taken as bits, exactly, so
        the sign of a zero and the payload of a NaN survive; and a literal
        this cannot read is refused by name rather than approximated.
        """
        lit = literal.strip()
        m = re.fullmatch(r'dense<(.+)>', lit)
        if m:
            lit = m.group(1)
        lit = lit.strip()

        is_hex = re.fullmatch(r'[+-]?0[xX][0-9a-fA-F]+', lit) is not None

        if elem.startswith('f'):
            if is_hex:
                # A raw bit pattern. `struct` must not see it: reinterpreting
                # 0x7FC00000 as the integer 2143289344 and rounding it to f32
                # gives a large finite number, which is exactly the "read as
                # another value" the contract forbids.
                bits = int(lit, 16)
                if not 0 <= bits <= 0xFFFFFFFF:
                    raise Diagnostic(
                        f"float constant {lit} does not fit 32 bits",
                        because='the vector element type is 32 bits wide',
                        source='compiler_overview.md sec 11.1')
                return bits
            try:
                # float() accepts 'inf', '-inf' and 'nan' as MLIR spells them,
                # and -0.0 keeps its sign bit through struct.pack.
                value = float(lit)
            except ValueError:
                raise Diagnostic(
                    f"cannot read the float literal {lit!r}",
                    because='the printed module is the only source of float '
                            'constant values, and a literal that does not '
                            'parse has no value to use',
                    remedy='report the literal form so the frontend can '
                           'learn it; it must not be guessed',
                    source='compiler_overview.md sec 11.1')
            return struct.unpack('<I', struct.pack('<f', value))[0]

        if is_hex:
            return int(lit, 16)
        try:
            if '.' in lit or 'e' in lit.lower():
                return int(float(lit))
            return int(lit)
        except ValueError:
            raise Diagnostic(
                f"cannot read the integer literal {lit!r}",
                source='compiler_overview.md sec 11.1')

    def _op_arith_constant(self, op):
        rid = op.get_result(0).id()
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        iv = self._constant_value(op)
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
        if isinstance(shape, tuple):
            # A 2-D constant is not a vector register's worth of
            # anything. It is either an index term -- `dense<16>`, the
            # row stride in every `A[m, k]` -- or the zero accumulator
            # `tt.dot` is given. Both are read by whatever consumes
            # them; neither is materialised here, and `_value` refuses
            # to invent a 1-D register for one.
            self._pending_const[rid] = ('tile', iv, elem)
            if not elem.startswith('f'):
                self._affine[rid] = Affine2D(const=iv)
            return
        if shape is None and not elem.startswith('f'):
            self._pending_const[rid] = ('scalar', iv, elem)
        else:
            # A float constant goes in a vector register even when TTIR
            # calls it a scalar: `s.li` would put it in a register no
            # floating-point instruction can read.
            if shape is not None:
                self._check_block(shape)
            self._pending_const[rid] = ('vector', iv, elem)

    # ---- shape rules ------------------------------------------------------
    def _check_block(self, shape):
        vlen = self.target.vlen
        if isinstance(shape, tuple):
            raise Unsupported(
                f"a {len(shape)}-D block of {'x'.join(map(str, shape))}",
                because='a vector register is one dimension; a 2-D tile '
                        'needs the register-blocking that sec 12 describes '
                        'and this importer does not build yet',
                remedy='use a 1-D block, or a tl.dot whose operands the '
                       'matrix lowering handles',
                source='compiler_overview.md sec 12')
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

    def _op_tt_get_num_programs(self, op):
        axis = op.get_int_attr('axis') or 0
        self._v[op.get_result(0).id()] = self.b.num_programs(axis)

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
        if isinstance(shape, tuple):
            # Stretching one value over a tile. For a pointer that is
            # the start of a tile access; for an integer it is a
            # constant term in the index. Neither materialises anything
            # -- there is no 2-D register to put it in.
            if _pointee(elem):
                if src.id() not in self._ptr_base:
                    raise Unsupported('splat of a non-argument pointer')
                name, pelem = self._ptr_base[src.id()]
                self._ptr2d[rid] = (name, pelem, Affine2D())
                return
            if elem.startswith('f'):
                # `sm_scale` over the whole tile. Broadcast once; every
                # row names the same register.
                v = self._broadcast_scalar(self._value(src), elem)
                self._tiles[rid] = Tile('lazy', lambda r: v, shape[0],
                                        shape[1], elem)
                return
            k = self._scalar_const(src.id())
            if k is not None:
                self._affine[rid] = Affine2D(const=k)
                return
            if src.id() in self._v and self._v[src.id()].kind != VECTOR:
                self._affine[rid] = Affine2D(dynamic=[self._v[src.id()]])
                return
            raise Unsupported(
                'a 2-D splat of a value that is neither a pointer nor a '
                'scalar index',
                because='a tile has no register to be broadcast into; only '
                        'a term of the index it is addressed by')
        self._check_block(shape)
        if _pointee(elem):
            # A splat pointer is a uniform base. sec 12: recognise a uniform
            # base with a bounded offset rather than materialising 16 pointers.
            if src.id() not in self._ptr_base:
                raise Unsupported('splat of a non-argument pointer')
            self._ptr_base[rid] = self._ptr_base[src.id()]
            self._ptr_chain[rid] = list(self._ptr_chain.get(src.id(), []))
            return
        v = self._value(src)
        if v.kind == VECTOR:
            # Already broadcast to every lane -- a reduced float, or a
            # float constant. Splatting it again would emit a `v.splat`
            # that reads a vector register, which is not what that
            # instruction does.
            self._v[rid] = v
        else:
            self._v[rid] = self.b.splat(v, dtype=elem)
        self._offsets[rid] = ('uniform', src.id())

    def _op_tt_addptr(self, op):
        base, offs = op.get_operand(0), op.get_operand(1)
        rid = op.get_result(0).id()
        if self._try_addptr_2d(rid, base, offs):
            return
        if base.id() not in self._ptr_base:
            raise Unsupported(
                'a pointer that is not a uniform base plus an offset',
                because='sec 12 asks for uniform base + bounded offset to be '
                        'recognised; a general tensor of pointers would need '
                        'per-lane addressing',
                source='compiler_overview.md sec 12')
        self._ptr_base[rid] = self._ptr_base[base.id()]
        self._offsets[rid] = ('ptr', offs.id())
        # The offsets this pointer accumulated, oldest first. Kept
        # explicitly because the offset table alone cannot say which
        # pointer an offset belonged to.
        #
        # The TTIR values, not their ids: an offset can be a constant
        # that has not been materialised yet, and only the value can ask
        # for it to be. Keeping ids here meant a chain containing a
        # constant vector -- `x[row*N + col + dx]`, every unrolled
        # convolution tap -- died in a KeyError instead of loading.
        self._ptr_chain[rid] = self._ptr_chain.get(base.id(), []) + [offs]

    def _op_arith_addi(self, op):
        if self._try_affine(op, 'add'):
            return
        self._binary(op, _INT_BIN['arith.addi'])

    def _op_arith_subi(self, op):
        if self._try_affine(op, 'sub'):
            return
        self._binary(op, 'sub')

    def _op_arith_muli(self, op):
        if self._try_affine(op, 'mul'):
            return
        self._binary(op, 'mul')

    def _op_arith_andi(self, op):
        self._binary(op, 'and')

    def _op_arith_ori(self, op):
        self._binary(op, 'or')

    def _op_arith_shrsi(self, op):
        self._binary(op, 'ashr')

    def _op_arith_shrui(self, op):
        self._binary(op, 'lshr')

    def _op_arith_shli(self, op):
        self._binary(op, 'shl')

    def _op_arith_maxsi(self, op):
        self._binary(op, 'max')

    def _op_arith_minsi(self, op):
        self._binary(op, 'min')

    def _op_arith_xori(self, op):
        self._binary(op, 'xor')

    def _op_arith_addf(self, op):
        self._binary(op, 'add')

    def _op_arith_subf(self, op):
        self._binary(op, 'sub')

    def _op_arith_mulf(self, op):
        self._binary(op, 'mul')

    # MLIR has two maxima and they are not the same function. ISA sec 8.3
    # has both: `.prop` propagates a NaN operand, `.ieee` ignores it.
    # `tl.maximum` lowers to `arith.maximumf`, and `tl.max`/softmax to
    # `arith.maxnumf` through the reduction -- so getting these the wrong
    # way round shows up as a NaN that should have been eaten, or a
    # number where a NaN was owed.
    def _op_arith_maximumf(self, op):
        self._binary(op, 'max', fmm='prop')

    def _op_arith_minimumf(self, op):
        self._binary(op, 'min', fmm='prop')

    def _op_arith_maxnumf(self, op):
        self._binary(op, 'max', fmm='ieee')

    def _op_arith_minnumf(self, op):
        self._binary(op, 'min', fmm='ieee')

    def _op_arith_negf(self, op):
        self._unary_fp(op, 'neg')

    def _op_math_absf(self, op):
        self._unary_fp(op, 'abs')

    def _unary_fp(self, op, kind):
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: self.b.vun(
                kind, xs[0], dtype=e))
            return
        _, elem, _bcast = self._float_shape(op)
        self.target.require('vector_fp32', f"emit a {elem} {kind}")
        self._v[op.get_result(0).id()] = self.b.vun(
            kind, self._vec(op.get_operand(0), elem), dtype=elem)

    # ---- two-dimensional blocks -------------------------------------------
    #
    # tiles.py explains why these do not lower one operation at a time:
    # `tt.expand_dims`, `tt.broadcast` and the `arith.muli` between them
    # have no meaning on this machine individually. What they add up to
    # does -- an affine index over (row, col) -- and that is what a
    # TensorMap describes. So they are tracked, not lowered, and the
    # `tt.load` at the end of the chain becomes a descriptor and a DMA.

    def _affine_of(self, value):
        """The affine index this TTIR value denotes, or None."""
        vid = value.id()
        if vid in self._affine:
            return self._affine[vid]
        k = self._scalar_const(vid)
        if k is not None:
            return Affine2D(const=k)
        pending = self._pending_const.get(vid)
        if pending is not None and pending[0] == 'vector':
            # A dense splat of one integer, which is a constant offset in
            # every lane. The bit pattern IS the value for an integer.
            _, iv, elem = pending
            if not elem.startswith('f'):
                return Affine2D(const=int(iv))
        return None

    def _op_tt_expand_dims(self, op):
        """`x[:, None]` or `x[None, :]`: which axis an iota runs along.

        This is the only place the row/column distinction enters. A
        `tt.make_range` is just 0..n; `expand_dims` is what says whether
        it indexes down the tile or across it.
        """
        src = op.get_operand(0)
        axis = op.get_int_attr('axis')
        tile = self._tiles.get(src.id())
        if tile is not None and tile.kind == 'col':
            # Already one value per row. `peak[:, None]` says exactly
            # that, so it is a rename.
            self._tiles[op.get_result(0).id()] = tile
            return
        kind, _ = self._offsets.get(src.id(), (None, None))
        if kind != 'arange' and src.id() in self._v:
            # A 1-D value of DATA, not an index. `peak[:, None]` makes
            # one value per row out of it and `q[None, :]` makes one row
            # repeated; the first needs the value's lanes taken apart,
            # which no instruction does, and the second does not.
            if axis == 0:
                shape, elem = _parse_type(
                    str(op.get_result(0).get_type()))
                v = self._value(src)
                self._tiles[op.get_result(0).id()] = Tile(
                    'lazy', lambda r: v, shape[0], shape[1], elem)
                return
            # `x[:, None]`: lane r of one register becomes the whole
            # of row r. ISA sec 8 has no instruction that takes a
            # vector apart by lane, so it goes out to memory and comes
            # back as sixteen scalars -- the same repack, the other way
            # round, as _col_to_vector.
            _, selem = _parse_type(str(src.get_type()))
            self._tiles[op.get_result(0).id()] = self._vector_to_col(
                self._value(src), selem)
            return
        if kind != 'arange':
            inner = self._affine_of(src)
            if inner is None:
                raise Unsupported(
                    'tt.expand_dims of something that is not a range',
                    because='the importer recognises a 2-D index as an '
                            'affine function of the row and the column, '
                            'and only a range establishes which one an '
                            'axis is',
                    source='doc/isa/TNCA-ISA-v0.1.md sec 9D.3')
            self._affine[op.get_result(0).id()] = inner
            return
        # axis=1 makes the range vary down the rows; axis=0, across.
        self._affine[op.get_result(0).id()] = (
            Affine2D(row_stride=1) if axis == 1 else Affine2D(col_stride=1))

    def _op_tt_broadcast(self, op):
        """Stretching a 16x1 or 1x16 to 16x16 changes no index."""
        src = op.get_operand(0)
        rid = op.get_result(0).id()
        tile = self._tiles.get(src.id())
        if tile is not None:
            shape, elem = _parse_type(str(op.get_result(0).get_type()))
            if tile.kind == 'col':
                # One value per row, stretched across the row.
                get = self._tile_row_getter(tile, elem)
                self._tiles[rid] = Tile('lazy', get, shape[0], shape[1],
                                        elem)
                return
            if tile.rows == shape[0]:
                self._tiles[rid] = Tile(tile.kind, tile.handle, shape[0],
                                        shape[1], tile.dtype)
                return
            if tile.rows == 1:
                # `q[None, :]` stretched down the tile: every row is the
                # same vector, so one register is named sixteen times
                # rather than copied sixteen times.
                get = self._tile_row_getter(tile)
                self._tiles[rid] = Tile('lazy', lambda r: get(0), shape[0],
                                        shape[1], tile.dtype)
                return
            raise Unsupported(
                f"a broadcast from {tile.rows} rows to {shape[0]}",
                because='only a single row stretches down a tile without '
                        'moving data')
        aff = self._affine_of(src)
        if aff is not None:
            self._affine[rid] = aff
            return
        if src.id() in self._ptr2d:
            self._ptr2d[rid] = self._ptr2d[src.id()]
            return
        raise Unsupported(
            'tt.broadcast of a value that is neither an index nor a pointer',
            because='a broadcast of DATA would have to replicate a vector '
                    'register across a tile, which is a real data movement '
                    'and not the index bookkeeping this recognises')

    def _try_affine(self, op, kind):
        """Track an integer operation on a 2-D index. True if handled."""
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        if not isinstance(shape, tuple) and op.get_result(0).id() \
                not in self._affine:
            # 1-D and scalars keep their existing lowering; only a 2-D
            # result, or an operand already being tracked, comes here.
            if not any(self._affine_of(op.get_operand(i)) is not None
                       and isinstance(
                           _parse_type(str(op.get_operand(i).get_type()))[0],
                           tuple)
                       for i in range(op.get_num_operands())):
                return False
        a = self._affine_of(op.get_operand(0))
        b = self._affine_of(op.get_operand(1))
        if a is None or b is None:
            raise Unsupported(
                f"a 2-D index built with {op.get_name()!r} out of something "
                f"this does not track",
                because='a tile is described by an affine index over the '
                        'row and the column (ISA sec 9D.3), and an operand '
                        'that is not one has no place in a descriptor')
        rid = op.get_result(0).id()
        if kind == 'add':
            self._affine[rid] = a.plus(b)
            return True
        if kind == 'mul':
            # One side must be a constant: an index that multiplies two
            # varying terms is not affine, and no descriptor describes it.
            for x, y in ((a, b), (b, a)):
                if (y.row_stride, y.col_stride, y.dynamic) == (0, 0, []):
                    self._affine[rid] = x.scaled(y.const)
                    return True
            raise Unsupported(
                'a 2-D index that multiplies two varying terms',
                because='a TensorMap gives each axis one stride (ISA sec '
                        '9D.3); a product of two indices has no stride to be')
        raise Unsupported(f"a 2-D index built with {op.get_name()!r}")

    def _try_addptr_2d(self, rid, base, offs):
        """Accumulate a 2-D pointer's index. True if this was one.

        A 2-D pointer arrives as `splat(p)` broadcast to the tile shape,
        with each `tt.addptr` contributing one affine term. Nothing is
        emitted: what the chain adds up to becomes a descriptor when the
        load at the end of it asks.
        """
        aff = self._affine_of(offs)
        if aff is None:
            return False
        prev = self._ptr2d.get(base.id())
        if prev is None:
            spec = self._ptr_base.get(base.id())
            if spec is None:
                return False
            shape, _ = _parse_type(str(base.get_type()))
            result_2d = isinstance(
                _parse_type(str(offs.get_type()))[0], tuple)
            if not result_2d and not isinstance(shape, tuple):
                return False
            name, elem = spec
            prev = (name, elem, Affine2D())
        name, elem, acc = prev
        self._ptr2d[rid] = (name, elem, acc.plus(aff))
        return True

    def _add_dynamic(self, base, aff, esize):
        """Fold the index's runtime terms into a byte address.

        Each carries its own coefficient in elements, so the scale and
        the conversion to bytes are one multiply: `s * K` in a split-k
        matmul becomes `s * (K * sizeof)`.
        """
        for v, coeff in aff.dynamic:
            step = coeff * esize
            term = v if step == 1 else self.b.sbini(
                'mul', v, step, comment=f"a runtime tile origin, x{step}")
            base = self.b.sbin('add', base, term,
                               comment='the tile origin')
        return base

    def _tile_descriptor(self, name, elem, aff, block, what):
        """A TensorMap for the tile this pointer arithmetic describes.

        The constant part of the index folds into the descriptor's base
        rather than becoming a coordinate, so the transfer runs at
        coordinate [0, 0] and the extents describe exactly the tile. A
        coordinate would have had to be divided by the row stride to be
        recovered, and a division that is only correct when the offset
        happens to be a multiple of it is not one to build on.
        """
        if aff.col_stride != 1:
            raise Unsupported(
                f"a tile whose columns are {aff.col_stride} elements apart",
                because='ISA sec 9E.2 refuses a descriptor whose innermost '
                        'stride is not one element: the transfer engine '
                        'copies runs, it does not gather per element',
                remedy='index the last axis contiguously',
                source='doc/isa/TNCA-ISA-v0.1.md sec 9E.2')
        if aff.row_stride <= 0:
            raise Unsupported(
                f"a tile with a row stride of {aff.row_stride}",
                because='a TensorMap addresses rows forward from its base')

        esize = self._ELEM_BYTES[elem]
        base = self._arg_value(name)
        if aff.const:
            base = self.b.sbini(
                'add', base, aff.const * esize,
                comment=f"the tile's origin, folded into the descriptor "
                        f"base ({aff.const} elements)")
        base = self._add_dynamic(base, aff, esize)
        return self._build_descriptor(
            base=base, elem=elem, extents=list(block),
            byte_strides=[aff.row_stride * esize, esize],
            block_shape=list(block), what=what)

    # ---- descriptors ------------------------------------------------------

    _ELEM_BYTES = {'f16': 2, 'bf16': 2, 'f32': 4, 'i8': 1, 'i32': 4}

    def _scratch_base(self):
        """Where this kernel builds its TensorMaps.

        `td.load Dd, [Sa]` snapshots a 256-byte object FROM MEMORY (ISA
        sec 9D.1); there is no instruction that builds a descriptor out
        of registers. A kernel whose base pointer arrives as an argument
        therefore has to write its own map somewhere before it can load
        it.

        That somewhere is one slot past the declared arguments. The
        frozen offsets in spec/r2_abi_v1.json are `8 * index` in
        signature order, so appending leaves every declared argument
        exactly where it was; the compiler reports `scratch_bytes` in
        the manifest and the launch path supplies the buffer.
        """
        if self._scratch is None:
            if not self.dynamic_args:
                raise Unsupported(
                    'a descriptor in a kernel compiled for literal '
                    'arguments',
                    because='the TensorMap has to be written to memory, and '
                            'the scratch buffer arrives through the argument '
                            'block that this mode does not read',
                    remedy='compile with dynamic arguments')
            self._scratch = self.b.sload(
                self.b.special('arg_base'), offset=8 * len(self.arg_names),
                size='d',
                comment='descriptor scratch: one slot past the declared '
                        'arguments (ABI v1 offsets are unchanged)')
        return self._scratch

    def _spill_region(self, nbytes):
        """Reserve scratch for a repack, and return its address.

        Repacking through memory is not a workaround here, it is the
        mechanism the architecture leaves: there is no scratchpad (sec
        5.2), the ISA reserves `v.pack.f16` and `v.unpack.f16` (sec
        8.4), and nothing in sec 8 takes a vector apart by lane. When a
        value has to change shape, memory is where it changes.

        Rounded to a whole TensorMap so the descriptors that follow
        stay 8-byte aligned, which ISA sec 9E.2 requires of a base.
        """
        offset = self.scratch_maps * TENSORMAP_BYTES + self.spill_bytes
        self.spill_bytes += (nbytes + 7) & ~7
        return self.b.sbini('add', self._scratch_base(), offset,
                            comment=f"{nbytes} bytes of repack scratch")

    def _rows_to_window(self, rows, dtype, out_dtype, cols, nrows=None):
        """Sixteen vector registers into one transfer-engine fragment.

        A per-lane store of `cols * sizeof` bytes writes one row
        contiguously, and sixteen of those in a row ARE the fragment
        that ISA sec 9C.2 describes -- a flat row-major byte array.
        So the repack is a store, a descriptor over what was stored,
        and one transfer back.

        The conversion happens on the way out, per lane, which is how
        this reaches a packed FP16 fragment without `v.pack.f16`.
        """
        esize = self._ELEM_BYTES[out_dtype]
        nrows = nrows or (self.target.vlen if callable(rows)
                          else len(rows))
        addr = self._spill_region(nrows * cols * esize)
        self.b.comment(f"repack {nrows}x{cols} {out_dtype} through memory: "
                       f"there is no v.pack (ISA sec 8.4)")
        row_bytes = cols * esize
        for r in range(nrows):
            v = rows(r) if callable(rows) else rows[r]
            if out_dtype != dtype:
                v = self.b.vcvt(dtype, out_dtype, v,
                                comment=f"row {r}: {dtype} -> {out_dtype}")
            at = addr if r == 0 else self.b.sbini('add', addr, r * row_bytes,
                                                  comment=f"row {r}")
            self.b.store(at, v, size=_SIZE_SUFFIX[esize])
        desc = self._build_descriptor(
            base=addr, elem=out_dtype, extents=[nrows, cols],
            byte_strides=[row_bytes, esize], block_shape=[nrows, cols],
            what='a repacked tile')
        zero = self.b.const(0)
        return self._dma_in(desc.value, [zero, zero], [nrows, cols],
                            out_dtype, 'a repacked tile')

    def _vector_to_col(self, v, elem):
        """One vector register into one value per row.

        The mirror of `_col_to_vector`, and through memory for the same
        reason: lane r has to become row r, and no instruction reads a
        single lane.
        """
        esize = self._ELEM_BYTES[elem]
        vlen = self.target.vlen
        addr = self._spill_region(vlen * esize)
        self.b.comment('one vector into one value per row: no instruction '
                       'reads a single lane (ISA sec 8)')
        self.b.store(addr, v, size=_SIZE_SUFFIX[esize])
        out = []
        for r in range(vlen):
            at = addr if r == 0 else self.b.sbini('add', addr, r * esize,
                                                  comment=f"row {r}")
            out.append(self.b.sload(at, size=_SIZE_SUFFIX[esize],
                                    dtype=elem))
        return Tile('col', tuple(out), vlen, 1, elem)

    def _col_to_vector(self, tile, elem):
        """One value per row, gathered into one vector register.

        The values are in sixteen separate registers and have to end up
        in sixteen lanes of one. Nothing in ISA sec 8 writes a single
        lane, so they go out to memory and come back as a vector load.
        """
        esize = self._ELEM_BYTES[elem]
        addr = self._spill_region(len(tile.handle) * esize)
        self.b.comment('one value per row into one vector: no instruction '
                       'writes a single lane (ISA sec 8)')
        for r, v in enumerate(tile.handle):
            at = addr if r == 0 else self.b.sbini('add', addr, r * esize,
                                                  comment=f"row {r}")
            if v.kind == VECTOR:
                # A broadcast reduction result: lane 0 is the value.
                self.b.store(at, v, size=_SIZE_SUFFIX[esize],
                             mask=self.b.vcmp('eq', self.b.arange(),
                                              self.b.splat(self.b.const(0))))
            else:
                self.b.sstore(at, v, size=_SIZE_SUFFIX[esize])
        return self.b.load(addr, dtype=elem, size=_SIZE_SUFFIX[esize])

    def _build_descriptor(self, *, base, elem, extents, byte_strides,
                          block_shape, what):
        """Write a TensorMap into scratch and snapshot it.

        Every word is written, including the zero ones. The scratch
        buffer's previous contents are not this kernel's to assume, and
        a descriptor half-inherited from the last launch is a wrong
        address rather than a fault.
        """
        try:
            check_geometry(byte_strides=byte_strides,
                           block_shape=block_shape,
                           elem_bytes=self._ELEM_BYTES[elem])
        except TileError as e:
            raise Unsupported(str(e), source='doc/isa/TNCA-ISA-v0.1.md '
                                             'sec 9E.2')

        slot = self.scratch_maps
        self.scratch_maps += 1
        addr = self.b.sbini('add', self._scratch_base(),
                            slot * TENSORMAP_BYTES,
                            comment=f"TensorMap {slot}: {what}")

        words = tensormap_words(base=0, extents=extents,
                                byte_strides=byte_strides,
                                block_shape=block_shape, dtype=elem)
        zero = self.b.const(0)
        for i, word in enumerate(words):
            if i == 0:
                # The base address is a register, not a literal: it came
                # from an argument slot.
                self.b.sstore(addr, base, offset=0, size='d')
                continue
            self.b.sstore(addr, zero if word == 0 else self.b.const(word),
                          offset=i * 8, size='d')
        return _Descriptor(self.b.descriptor(addr, name=f"d{slot}"),
                           base=base, byte_strides=byte_strides,
                           block=list(block_shape), elem=elem)

    def _op_tt_make_tensor_descriptor(self, op):
        """`tl.make_tensor_descriptor`, straight onto ISA sec 9D.

        This is the case the hardware was designed for, so it is also
        the shortest lowering in the file: the shape, the strides and
        the block shape are what a TensorMap holds, and the operation
        says all three.
        """
        ptr = op.get_operand(0)
        n = op.get_num_operands()
        rank = (n - 1) // 2
        if rank != 2:
            raise Unsupported(
                f"a rank-{rank} tensor descriptor",
                because='the tile path builds 16x16 fragments; a rank the '
                        'MMA does not consume has nowhere to go yet',
                source='doc/isa/TNCA-ISA-v0.1.md sec 9C.2')
        elem = _pointee(str(ptr.get_type()))
        if elem not in self._ELEM_BYTES:
            raise Unsupported(f"a tensor descriptor of {elem}")
        esize = self._ELEM_BYTES[elem]

        extents, strides = [], []
        for i in range(rank):
            k = self._scalar_const(op.get_operand(1 + i).id())
            if k is None:
                raise Unsupported(
                    'a tensor descriptor whose shape is a runtime value',
                    because='the extents are written into the TensorMap, '
                            'and only a constant can be written by the '
                            'store that builds it',
                    remedy='make the shape a tl.constexpr')
            extents.append(k)
        for i in range(rank):
            k = self._scalar_const(op.get_operand(1 + rank + i).id())
            if k is None:
                raise Unsupported(
                    'a tensor descriptor whose strides are runtime values')
            strides.append(k * esize)

        shape, _ = _parse_type(str(op.get_result(0).get_type()).replace(
            '!tt.tensordesc<', 'tensor<').rstrip('>') + '>')
        block = list(shape) if isinstance(shape, tuple) else [1, shape or 1]

        self._descs[op.get_result(0).id()] = self._build_descriptor(
            base=self._value(ptr), elem=elem, extents=extents,
            byte_strides=strides, block_shape=block,
            what='tl.make_tensor_descriptor')

    def _op_tt_descriptor_load(self, op):
        d = self._descs[op.get_operand(0).id()]
        coords = [self._value(op.get_operand(1 + i))
                  for i in range(op.get_num_operands() - 1)]
        self._tiles[op.get_result(0).id()] = self._dma_in(
            d.value, coords, d.block, d.elem, 'tt.descriptor_load')

    def _op_tt_descriptor_store(self, op):
        d = self._descs[op.get_operand(0).id()]
        # The stored value is the tensor operand. Taking the last one
        # got a coordinate, because MLIR prints `desc[i, j], value` and
        # orders the operands `desc, value, i, j`.
        value = None
        for i in range(1, op.get_num_operands()):
            cand = op.get_operand(i)
            if isinstance(_parse_type(str(cand.get_type()))[0], tuple):
                value = cand
                break
        if value is None:
            raise Unsupported('a tt.descriptor_store with no tile to store')
        tile = self._tile_of(value)
        if tile.kind == 'accumulator':
            # The result is in an accumulator and the destination is
            # described by a TensorMap whose base and stride this
            # compiler wrote. Draining through the vector unit reaches
            # it directly (ISA sec 9B); routing it back through a window
            # would be a transfer to reach an address already in hand.
            self._drain_tile(tile, d.base, d.byte_strides[0], d.elem)
            return
        coords = [self._value(op.get_operand(i))
                  for i in range(1, op.get_num_operands())
                  if op.get_operand(i).id() != value.id()]
        self._dma_out(d.value, coords, tile, d.elem, 'tt.descriptor_store')

    # ---- tiles in and out -------------------------------------------------

    def _window_for(self, rows, cols, elem):
        """A window big enough for one fragment of this shape.

        ISA sec 9C.2: a fragment is a flat row-major byte array, so the
        register count follows from the element width alone -- a 16x16
        FP16 fragment is eight registers and the same fragment in FP8 is
        four. There is no separate packing rule to get wrong.
        """
        from .footprints import fragment_registers
        bits = self._ELEM_BYTES[elem] * 8
        return self.b.window(fragment_registers(rows, cols, bits))

    def _dma_in(self, desc, coords, block, elem, what, transpose=False):
        """One asynchronous tile load, and the wait for it.

        Armed for exactly one operation. ISA sec 9E.1 step 8 signals a
        completion once per command however many transactions it expands
        into, which is what makes a static count checkable -- and the
        count being checkable is the only reason to arm it rather than
        poll.
        """
        window = self._window_for(block[0], block[1], elem)
        bar = self.b.barrier(name=f"fill{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.comment(f"{what}: one tile into a window (ISA sec 9E)")
        self.b.dma_load(window, desc.value if hasattr(desc, 'value')
                        else desc, coords, tok, dtype=elem,
                        transpose=transpose)
        self.b.barrier_wait(tok)
        return Tile('window', window, block[0], block[1], elem,
                    desc=(desc, coords))

    def _dma_out(self, desc, coords, tile, elem, what):
        if tile.kind != 'window':
            raise Unsupported(
                f"{what} from a tile that is not in a transfer window")
        bar = self.b.barrier(name=f"drain{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.comment(f"{what}: one tile out of a window (ISA sec 9E)")
        self.b.dma_store(window, desc, coords, tok, dtype=elem)
        self.b.barrier_wait(tok)

    def _drain_tile(self, tile, base, row_stride_bytes, dst_elem):
        """A 16x16 accumulator out to memory, a row at a time.

        ISA sec 9B exists for this. The accumulator is binary32 whatever
        the MMA's inputs were (sec 9C), and it is reachable from the
        vector unit -- so a row is `a.vread`, a convert if the
        destination is narrower, and a vector store. sec 5.4 is explicit
        that this round trip not having to go through memory is what
        makes a scratchpad-free design worth having.

        Converting on the way out is also what avoids the one thing this
        ISA cannot do: `arith.truncf` on a whole tile would need
        `v.pack.f16` to rebuild a packed fragment, and sec 8.4 reserves
        it. Per row, into a per-lane 16-bit store, there is nothing to
        pack.
        """
        esize = self._ELEM_BYTES[dst_elem]
        size = _SIZE_SUFFIX[esize]
        self.b.comment(f"tile epilogue: {tile.rows} rows through the vector "
                       f"unit, as {dst_elem} (ISA sec 9B)")
        get = self._tile_row_getter(tile)
        for r in range(tile.rows):
            v = get(r)
            if dst_elem != tile.dtype:
                v = self.b.vcvt(tile.dtype, dst_elem, v,
                                comment=f"row {r}: {tile.dtype} -> "
                                        f"{dst_elem} (ISA sec 8.4)")
            addr = base if r == 0 else self.b.sbini(
                'add', base, r * row_stride_bytes,
                comment=f"row {r}")
            self.b.store(addr, v, size=size)

    def _tile_of(self, value):
        t = self._tiles.get(value.id())
        if t is None:
            raise Unsupported(
                f"a 2-D value that is not a tile (type {value.get_type()})",
                because='a 16x16 block lives in a transfer-engine window or '
                        'an accumulator; a value that reached here in '
                        'neither was produced by an operation with no 2-D '
                        'lowering')
        return t

    def _op_tt_trans(self, op):
        """`tl.trans`, done by the transfer engine.

        ISA sec 9E's `mod[5]` asks the engine to write a tile
        transposed, so a transposed operand is the SAME transfer with
        one bit set -- the permutation happens while the data is in
        flight. There is no vector instruction that permutes a fragment,
        so the alternative would have been a round trip through memory
        for something the DMA does for nothing.

        It therefore only applies to a tile that came from a transfer.
        A tile computed in the vector unit has no transfer to set the
        bit on.
        """
        src = op.get_operand(0)
        tile = self._tile_of(src)
        order = [op.get_int_attr('order')] if False else None
        if tile.kind != 'window' or tile.desc is None:
            raise Unsupported(
                'tl.trans of a tile that was not loaded by the transfer '
                'engine',
                because='ISA sec 9E transposes a tile as it lands; nothing '
                        'in sec 8 permutes a fragment already in registers',
                remedy='transpose the operand as it is loaded')
        desc, coords = tile.desc
        self.b.comment('tl.trans: the same transfer, written transposed '
                       '(ISA sec 9E mod[5])')
        self._tiles[op.get_result(0).id()] = self._dma_in(
            desc, coords, [tile.cols, tile.rows], tile.dtype, 'tl.trans',
            transpose=True)

    def _op_tt_dot(self, op):
        """`tl.dot`, onto ISA sec 9C.

        The accumulator is binary32 whatever the inputs are, and sec
        9C.1 fixes the order and the rounding, so there is nothing to
        choose here: the two fragments are already in windows, in the
        layout the transfer engine wrote and the MMA reads, and the
        command is the whole lowering.
        """
        a = self._tile_of(op.get_operand(0))
        b = self._tile_of(op.get_operand(1))
        # ISA sec 9C reads A and B from vector registers in the fragment
        # layout. A value computed in the vector unit is not in it, and
        # sec 8.4 reserves the pack that would put it there -- so it
        # goes out to memory and comes back through the engine, which
        # is the repack path a scratchpad-free design leaves.
        a = self._as_fragment(a, 'the left operand of tl.dot')
        b = self._as_fragment(b, 'the right operand of tl.dot')
        if a.dtype != b.dtype:
            raise Unsupported(
                f"a tl.dot of {a.dtype} with {b.dtype}: ISA sec 9C's "
                f"mod[3:2] names ONE input format for both operands")

        acc = self.b.accumulator(name=f"acc{len(self.kernel.accumulators)}")
        # `tt.dot`'s third operand is the value accumulated into. Triton
        # passes an explicit zero tensor for a plain product, and an
        # accumulator carried across a loop otherwise; only the first is
        # expressible while a tile has no way to enter an accumulator.
        cin = op.get_operand(2)
        if self._tiles.get(cin.id()) is not None:
            raise Unsupported(
                'a tl.dot accumulating into a tile that is already live',
                because='ISA sec 9B can write an accumulator a row at a '
                        'time, but this importer has no path that puts a '
                        'computed tile back into one yet')
        if not self._is_zero_tensor(cin):
            raise Unsupported(
                'a tl.dot whose accumulator input is not zero',
                because='the MMA adds into its accumulator, so a non-zero '
                        'input has to be placed there first')
        self.b.acc_zero(acc)

        bar = self.b.barrier(name=f"mma{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.mma_async(acc, a.handle, b.handle, tok, dtype=a.dtype)
        # ISA sec 9C.3: the accumulator may not be read by the vector
        # unit before the command completes, and its sources may not be
        # overwritten. The wait is what makes both true.
        self.b.barrier_wait(tok)

        _, rtype = _parse_type(str(op.get_result(0).get_type()))
        self._tiles[op.get_result(0).id()] = Tile(
            'accumulator', acc, a.rows, b.cols, rtype)

    def _as_fragment(self, tile, what):
        """This tile in the layout `m.mma.async` reads."""
        if tile.kind == 'window':
            return tile
        if tile.dtype not in ('f16', 'bf16'):
            raise Unsupported(
                f"{what}: a computed {tile.dtype} tile",
                because="ISA sec 9C's mod[3:2] names FP16, BF16 and the two "
                        "FP8 encodings as input formats, and nothing else",
                remedy='convert it, which tl.dot on a narrower type does '
                       'anyway')
        return self._rows_to_window(self._tile_row_getter(tile), tile.dtype,
                                    tile.dtype if tile.dtype != 'f32'
                                    else tile.dtype, tile.cols,
                                    nrows=tile.rows)

    def _is_zero_tensor(self, value):
        pending = self._pending_const.get(value.id())
        if pending is None:
            return False
        return int(pending[1]) == 0

    # ---- a tile in the vector unit ----------------------------------------
    #
    # A third place a 2-D value can be, besides a transfer-engine window
    # and an accumulator: sixteen vector registers, one per row. That is
    # where elementwise arithmetic on a tile happens, because a vector
    # register is one row of one and there is nothing else it could be.
    #
    # Moving between the three is asymmetric, and the asymmetry is the
    # ISA's rather than this compiler's:
    #
    #   accumulator -> rows   `a.vread`, one instruction a row (sec 9B)
    #   rows -> accumulator   `a.vwrite`, likewise
    #   window -> rows        NOT POSSIBLE for a narrow type. Unpacking
    #                         a 16x16 FP16 fragment needs v.unpack.f16,
    #                         which sec 8.4 reserves.
    #   rows -> window        through memory: a per-lane 16-bit store
    #                         writes 32 contiguous bytes a row, and
    #                         sixteen of those ARE the fragment (9C.2).

    def _row_getter(self, value, rows, elem):
        """A function from row index to that row, in a vector register.

        Rows are pulled one at a time rather than all sixteen at once.
        That is not a micro-optimisation: there are 64 vector registers
        and 16 predicate ones, a fragment window costs eight, and a
        FlashAttention step has three tiles live. Materialising every
        row of every tile ran the allocator out.
        """
        tile = self._tiles.get(value.id())
        if tile is not None:
            return self._tile_row_getter(tile, elem)
        shape, selem = _parse_type(str(value.get_type()))
        if not isinstance(shape, tuple):
            v = self._vec(value, elem)
            return lambda r: v
        pending = self._pending_const.get(value.id())
        if pending is not None and pending[0] == 'tile':
            # A dense constant over the tile: the same value in every
            # lane of every row, so one register serves all sixteen.
            v = self.b.vmovi(int(pending[1]), dtype=pending[2])
            return lambda r: v
        raise Unsupported(
            f"a 2-D operand of type {value.get_type()} that is not a tile")

    def _broadcast_scalar(self, v, dtype):
        return v if v.kind == VECTOR else self.b.splat(v, dtype=dtype)

    def _tile_row_getter(self, tile, elem=None):
        if tile.kind == 'rows':
            return lambda r: tile.handle[r]
        if tile.kind == 'lazy':
            return tile.handle
        if tile.kind == 'col':
            # One value per row, the same across the row.
            return lambda r: self._broadcast_scalar(
                tile.handle[r], elem or tile.dtype)
        if tile.kind == 'accumulator':
            return lambda r: self.b.acc_read(tile.handle, self.b.const(r))
        if tile.kind == 'window' and self._ELEM_BYTES[tile.dtype] == 4 \
                and tile.cols == self.target.vlen:
            # A 32-bit fragment is one row per register (sec 9C.2: the
            # fragment is a flat row-major byte array, and sixteen words
            # is exactly a vector register). No unpacking, so the
            # window's registers ARE the rows, named not copied.
            return lambda r: self.b.window_reg(tile.handle, r,
                                               dtype=tile.dtype)
        raise Unsupported(
            f"elementwise arithmetic on a {tile.dtype} tile that is in a "
            f"transfer window",
            because='reading one element per lane out of a packed fragment '
                    'needs v.unpack.f16, which ISA sec 8.4 reserves',
            remedy='do the arithmetic on a tl.dot result, which is in an '
                   'accumulator')

    def _rows_of(self, tile):
        """Every row of a tile at once. Prefer `_tile_row_getter`."""
        get = self._tile_row_getter(tile)
        return tuple(get(r) for r in range(tile.rows))

    def _materialise(self, rows_fn, rows, cols, elem):
        """Where a computed tile goes once it exists.

        An accumulator, when the element is 32 bits wide. A0..A15 are
        16x16 of binary32 (ISA sec 9B) -- which is a tile, exactly, and
        is the storage this machine provides for one. Writing each row
        back as it is computed keeps one vector register live instead of
        sixteen, and `a.vread` brings a row back when something wants it.

        A mask cannot go there: a predicate is not a vector register. So
        an i1 tile stays a rule for computing a row, re-applied by
        whoever reads one. A comparison is a single instruction, so
        recomputing it costs less than holding sixteen of the sixteen
        predicate registers.
        """
        if elem == 'i1':
            return Tile('lazy', rows_fn, rows, cols, elem)
        if self._ELEM_BYTES.get(elem) != 4:
            return Tile('rows', tuple(rows_fn(r) for r in range(rows)),
                        rows, cols, elem)
        acc = self.b.accumulator(name=f"acc{len(self.kernel.accumulators)}")
        self.b.comment(f"a computed {rows}x{cols} tile, into an accumulator "
                       f"(ISA sec 9B)")
        for r in range(rows):
            self.b.acc_write(acc, self.b.const(r), rows_fn(r))
        return Tile('accumulator', acc, rows, cols, elem)

    def _tile_binary(self, op, build):
        """An elementwise operation over a tile, a row at a time."""
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        rows, cols = shape
        getters = [self._row_getter(op.get_operand(i), rows, elem)
                   for i in range(op.get_num_operands())]
        self._tiles[op.get_result(0).id()] = self._materialise(
            lambda r: build([g(r) for g in getters], elem),
            rows, cols, elem)

    def _is_2d(self, op):
        return isinstance(
            _parse_type(str(op.get_result(0).get_type()))[0], tuple)

    def _op_tt_reduce_2d(self, op, kind, elem):
        """A reduction over a tile.

        Along the last axis it is ISA sec 8.7 applied to each row, which
        leaves one value per row.

        Along the first axis it is NOT a sec 8.7 reduction at all: it
        combines lane n of every row, and sixteen rows in sixteen
        registers combine with fifteen ordinary lane-local operations
        into one register. The answer is a vector, not a column, and it
        needs no cross-lane instruction.
        """
        tile = self._tile_of(op.get_operand(0))
        axis = op.get_int_attr('axis')
        get = self._tile_row_getter(tile, elem)
        rid = op.get_result(0).id()

        if axis == 0:
            acc = get(0)
            for r in range(1, tile.rows):
                acc = self.b.vbin(kind, acc, get(r), dtype=elem,
                                  comment=f"down the tile: row {r}"
                                  if r == 1 else None)
            self._v[rid] = acc
            return

        self._tiles[rid] = Tile(
            'col',
            tuple(self.b.reduce(kind, get(r), dtype=elem,
                                broadcast=elem.startswith('f'))
                  for r in range(tile.rows)),
            tile.rows, 1, elem)

    # ---- where a float lives ----------------------------------------------
    #
    # There is no scalar floating-point unit. ISA sec 7 has `s.add` and
    # `s.max.s` and no `s.fadd`, no scalar divide and no scalar SFU, so a
    # TTIR value of type `f32` -- what `tl.sum` returns, and what every
    # mean and every norm then divides by -- cannot be held in a scalar
    # register and operated on there.
    #
    # It is held in a vector register, broadcast to every lane. Sec 8.7's
    # `mod[0] = 1` reduction form produces exactly that, so the broadcast
    # costs nothing at the point the value is created, and `tt.splat` of
    # such a value is then a rename rather than an instruction.

    def _is_float_scalar(self, ttir_value):
        _, elem = _parse_type(str(ttir_value.get_type()))
        return elem.startswith('f') and elem != 'f'

    def _float_shape(self, op):
        """The shape a float operation should be lowered at.

        A tensor result keeps its own shape. A scalar float result is
        still lowered in the vector unit, because that is the only unit
        that has floating point -- so it reports VLEN, and the result is
        recorded as a broadcast value.
        """
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        if shape is not None:
            self._check_block(shape)
            return shape, elem, False
        if not elem.startswith('f'):
            return None, elem, False
        return self.target.vlen, elem, True

    def _vec(self, ttir_value, elem):
        """This value in a vector register, whatever it is held in now.

        A float scalar is already broadcast, so it comes back unchanged.
        An integer scalar -- a loop bound, an `n` about to be converted
        to a float -- is in a scalar register and gets a splat.
        """
        v = self._value(ttir_value)
        if v.kind == VECTOR:
            return v
        return self.b.splat(v, dtype=elem)

    # ---- transcendentals --------------------------------------------------
    # `math.*` in, a mathlib.py function out. Every one of these is built
    # on ISA sec 8.8, whose operations are approximate by contract;
    # mathlib.ACCURACY states what each lowering promises and why.
    _MATH = {
        'math.exp': 'exp', 'math.exp2': 'exp2',
        'math.log': 'log', 'math.log2': 'log2',
        'math.sqrt': 'sqrt', 'math.rsqrt': 'rsqrt',
        'math.absf': 'absf',
    }

    def _fconst(self, value, elem='f32'):
        """A whole-vector float constant, emitted once per value.

        Keyed on the bit pattern rather than the Python float so that
        +0.0 and -0.0 do not share a register: they are different
        numbers and `struct` keeps them apart even where `==` does not.
        """
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        key = (bits, elem)
        v = self._fconst_cache.get(key)
        if v is None:
            v = self.b.vmovi(bits, dtype=elem)
            self._fconst_cache[key] = v
        return v

    def _math(self, op, fname):
        self.target.require('sfu', f"emit {fname}")
        fn = getattr(mathlib, fname)
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: fn(
                self.b, lambda c: self._fconst(c, e), xs[0], dtype=e))
            return
        _, elem, _bcast = self._float_shape(op)
        self._v[op.get_result(0).id()] = fn(
            self.b, lambda c: self._fconst(c, elem),
            self._vec(op.get_operand(0), elem), dtype=elem)

    def _op_math_exp(self, op):
        self._math(op, 'exp')

    def _op_math_exp2(self, op):
        self._math(op, 'exp2')

    def _op_math_log(self, op):
        self._math(op, 'log')

    def _op_math_log2(self, op):
        self._math(op, 'log2')

    def _op_math_sqrt(self, op):
        self._math(op, 'sqrt')

    def _op_math_rsqrt(self, op):
        self._math(op, 'rsqrt')

    def _op_arith_divf(self, op):
        """IEEE division, which the ISA does not have (sec 8.2).

        A reciprocal and a multiply, with the reciprocal refined so the
        answer matches the division the IR asked for rather than being
        the four-ulp approximation the SFU returns.
        """
        self.target.require('sfu', 'emit a float division')
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: mathlib.div(
                self.b, lambda c: self._fconst(c, e), xs[0], xs[1],
                dtype=e))
            return
        _, elem, _bcast = self._float_shape(op)
        self._v[op.get_result(0).id()] = mathlib.div(
            self.b, lambda c: self._fconst(c, elem),
            self._vec(op.get_operand(0), elem),
            self._vec(op.get_operand(1), elem), dtype=elem)

    # ---- conversions ------------------------------------------------------
    def _convert(self, op, src, dst):
        _, elem, _bcast = self._float_shape(op)
        self._v[op.get_result(0).id()] = self.b.vcvt(
            src, dst, self._vec(op.get_operand(0), src))

    def _op_arith_truncf(self, op):
        self._float_convert(op)

    def _op_arith_extf(self, op):
        self._float_convert(op)

    def _float_convert(self, op):
        """A float width change.

        On a tile this records the destination type and emits nothing.
        The value is in an accumulator, which ISA sec 9C says is
        binary32 whatever the MMA's inputs were, and the conversion
        happens on the way out of it -- one `vf.cvt` per row in the
        epilogue, where there is no packed fragment to rebuild and so no
        need for the `v.pack.f16` that sec 8.4 reserves.
        """
        src, res = op.get_operand(0), op.get_result(0)
        tile = self._tiles.get(src.id())
        _, rtype = _parse_type(str(res.get_type()))
        if tile is not None:
            self._tiles[res.id()] = Tile(tile.kind, tile.handle, tile.rows,
                                         tile.cols, rtype, desc=tile.desc)
            return
        _, elem, _bcast = self._float_shape(op)
        _, stype = _parse_type(str(src.get_type()))
        self._v[res.id()] = self.b.vcvt(stype, rtype,
                                        self._vec(src, stype))

    def _op_arith_sitofp(self, op):
        self._convert(op, 's32', 'f32')

    def _op_arith_uitofp(self, op):
        self._convert(op, 'u32', 'f32')

    def _op_arith_fptosi(self, op):
        self._convert(op, 'f32', 's32')

    def _op_arith_fptoui(self, op):
        self._convert(op, 'f32', 'u32')

    def _op_arith_select(self, op):
        """`arith.select(cond, on_true, on_false)`.

        ISA 0x19 is `Vd[i] = mask[i] ? Va[i] : Vb[i]` and writes every
        lane, so the operand order carries across unchanged and no
        separate merge is needed.
        """
        cond, a, b = (op.get_operand(0), op.get_operand(1),
                      op.get_operand(2))
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: self.b.vselect(
                xs[0], xs[1], xs[2]))
            return
        shape, elem, _bcast = self._float_shape(op)
        if shape is None:
            raise Unsupported('a select on integer scalars')
        self._v[op.get_result(0).id()] = self.b.vselect(
            self._value(cond), self._vec(a, elem), self._vec(b, elem))

    def _binary(self, op, kind, fmm=None):
        a, b = op.get_operand(0), op.get_operand(1)
        rid = op.get_result(0).id()
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: self.b.vbin(
                kind, xs[0], xs[1], dtype=e, fmm=fmm))
            return
        _, relem = _parse_type(str(op.get_result(0).get_type()))
        if relem == 'i1':
            # A mask combined with a mask. `tl.where(m & (v == b), ...)`
            # is two comparisons and an AND, and a comparison leaves its
            # result in a predicate register, not a vector one (ISA sec
            # 0x18). Sending it through v.and would have splatted a
            # predicate register number as though it were a scalar's.
            if kind not in ('and', 'or', 'xor'):
                raise Unsupported(f"{kind!r} on two masks")
            self._v[rid] = self.b.pbin(kind, self._value(a), self._value(b))
            return
        shape, elem, _bcast = self._float_shape(op)
        if shape is None:
            self._v[rid] = self.b.sbin(kind, self._value(a), self._value(b))
        else:
            if elem.startswith('f'):
                self.target.require('vector_fp32', f"emit a {elem} {kind}")
            self._v[rid] = self.b.vbin(kind, self._vec(a, elem),
                                       self._vec(b, elem),
                                       dtype=elem, fmm=fmm)
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
        shape, elem = _parse_type(str(a.get_type()))
        if isinstance(shape, tuple):
            rows, cols = shape
            xs = self._row_getter(a, rows, elem)
            ys = self._row_getter(b, rows, elem)
            self._tiles[op.get_result(0).id()] = self._materialise(
                lambda r: self.b.vcmp(cc, xs(r), ys(r)), rows, cols, 'i1')
            return
        if elem == 'i1':
            # Comparing two masks. `(x == y)` on i1 is equality, which
            # is the negation of exclusive-or; the vector comparison
            # unit does not take predicate registers.
            if cc not in ('eq', 'ne'):
                raise Unsupported(f"a {cc!r} comparison of two masks")
            x = self.b.pbin('xor', self._value(a), self._value(b))
            self._v[op.get_result(0).id()] = (
                x if cc == 'ne' else self.b.pun('not', x))
            return
        self._v[op.get_result(0).id()] = self.b.vcmp(
            cc, self._vec(a, elem), self._vec(b, elem))

    # ---- memory -----------------------------------------------------------
    def _address(self, ptr_value):
        """A scalar byte address for a recognised uniform-base access.

        Returns (address, size suffix, element type, kind, index) where
        kind is 'contiguous' or 'indexed'.

        This walks the WHOLE chain of `tt.addptr`s back to the argument
        and adds up what each one contributed. It used to look at the
        last link only and take the argument as the base, which meant
        `p + row * N + offs` silently loaded from `p + offs`: row 0 gave
        the right answer and every other row gave row 0's data, with no
        error anywhere. Any kernel indexing a second dimension hits that,
        which is most of them.
        """
        pid = ptr_value.id()
        if pid not in self._ptr_base:
            raise Unsupported(
                'a pointer this importer did not track back to an '
                'argument',
                because='sec 12 recognises a uniform base plus a bounded '
                        'offset; a pointer built some other way would '
                        'need per-lane addressing',
                source='compiler_overview.md sec 12')
        name, elem = self._ptr_base[pid]
        esize = _DTYPE_BYTES.get(elem)
        if esize is None:
            raise Unsupported(f"element type {elem!r}")

        # Every offset added to this pointer, oldest first.
        chain = list(self._ptr_chain.get(pid, []))

        # Each link of the chain is sorted into one of three piles: a
        # compile-time element count, a scalar (lane-invariant) term, and
        # a lane-varying term. What the access turns out to be follows
        # from which piles are non-empty, rather than from the order the
        # links happened to arrive in.
        scalar_terms, lane_terms, const_elems = [], [], 0
        pure_arange = 0
        for value in chain:
            off = value.id()
            okind, ouni = self._offsets.get(off, (None, None))
            known = self._scalar_const(off)
            if known is not None and okind is None:
                # A constant number of elements. `p + k * N + offs` with a
                # constexpr k arrives here once Triton has folded `k * N`,
                # and folding it into the address beats materialising a
                # register to add zero.
                const_elems += known
            elif okind == 'arange':
                pure_arange += 1
                lane_terms.append(value)
            elif okind == 'contiguous':
                # `uniform + arange`, already summed into one vector. The
                # uniform part is lane-invariant, so hoist it into the
                # scalar address and let the arange be the lane term.
                pure_arange += 1
                lane_terms.append(value)
                scalar_terms.append(self._v[ouni])
            elif okind == 'uniform':
                scalar_terms.append(self._v[ouni])
            elif off in self._v and getattr(self._v[off], 'kind',
                                            None) == 'scalar':
                scalar_terms.append(self._v[off])
            else:
                # A general per-lane offset, in elements: ISA sec 9.1
                # scales a gather index by the access size, so it needs no
                # conversion to bytes the way a scalar term does.
                lane_terms.append(value)

        # A gather's address is `Sa + Vb[i]*sz`, so a scalar offset and a
        # per-lane offset on the same pointer are not in conflict: the
        # first belongs in `Sa` and the second in `Vb`. Refusing the
        # combination cost `x[row*N + col + dx]` -- which is every
        # stencil and every convolution -- for no reason.
        indexed = len(lane_terms) > pure_arange or len(lane_terms) > 1

        addr = self._arg_value(name)
        if const_elems:
            addr = self.b.sbini(
                'add', addr, const_elems * esize,
                comment=f"&{name}[{const_elems}] (a constant offset, "
                        f"x{esize})")
        if scalar_terms:
            total = scalar_terms[0]
            for extra in scalar_terms[1:]:
                total = self.b.sbin('add', total, extra,
                                    comment='the offsets add up')
            shift = {1: 0, 2: 1, 4: 2, 8: 3}[esize]
            # Every pointer in a kernel tends to share the same block
            # offset, so this is computed once rather than once per
            # access. Not a general CSE -- sec 11.1 says not to build one
            # here -- just not emitting the same shift three times.
            key = (id(total), shift)
            byte = self._byte_offsets.get(key)
            if byte is None:
                byte = self.b.sbini(
                    'shl', total, shift,
                    comment=f"element offset -> bytes (x{esize})")
                self._byte_offsets[key] = byte
            addr = self.b.sbin('add', addr, byte, comment=f"&{name}[...]")

        if indexed:
            # More than one lane-varying term, or one that is not the
            # plain lane index: sum them and gather. `contiguous` here
            # would give lane i the element at i, which is only the same
            # answer when the sum IS the lane index.
            index = self._value(lane_terms[0])
            for extra in lane_terms[1:]:
                index = self.b.vbin('add', index, self._value(extra),
                                    dtype=index.dtype,
                                    comment='per-lane offsets add up, in '
                                            'elements (ISA sec 9.1)')
            return addr, _SIZE_SUFFIX[esize], elem, 'indexed', index

        if not lane_terms:
            # Nothing varies per lane, so every lane wants the SAME
            # element. A contiguous vector load would give lane i the
            # element at i and be wrong by fifteen lanes out of sixteen.
            return addr, _SIZE_SUFFIX[esize], elem, 'broadcast', None

        return addr, _SIZE_SUFFIX[esize], elem, 'contiguous', None

    def _scalar_const(self, vid):
        """The value of `vid` if it is a scalar constant, else None.

        Constants are materialised lazily, so one that is still pending
        is not in `_v` yet and the chain walk would otherwise take it for
        a per-lane offset it cannot handle.
        """
        pending = self._pending_const.get(vid)
        if pending is not None and pending[0] == 'scalar':
            return int(pending[1])
        return None

    def _arg_value(self, name):
        for _, (n, v) in self._args.items():
            if n == name:
                return v
        raise Diagnostic(f"no kernel argument {name!r}")

    def _op_tt_load(self, op):
        ptr = op.get_operand(0)
        shape2d, elem2d = _parse_type(str(op.get_result(0).get_type()))
        if isinstance(shape2d, tuple):
            spec = self._ptr2d.get(ptr.id())
            if spec is None:
                raise Unsupported(
                    'a 2-D load through a pointer this does not recognise',
                    because='tiles.py: a 16x16 block cannot be assembled '
                            'from vector loads, so the access has to be '
                            'describable as a TensorMap',
                    source='doc/isa/TNCA-ISA-v0.1.md sec 9E.2')
            if op.get_num_operands() > 1:
                raise Unsupported(
                    'a masked 2-D load',
                    because='a transfer\'s lane mask is all or nothing '
                            '(ISA sec 9E); per-element bounds come from the '
                            "descriptor's extents, not from a mask",
                    remedy='let the descriptor pad, or load the full tile')
            name, elem, aff = spec
            desc = self._tile_descriptor(name, elem, aff, shape2d,
                                         'a 2-D tt.load')
            zero = self.b.const(0)
            self._tiles[op.get_result(0).id()] = self._dma_in(
                desc.value, [zero, zero], list(shape2d), elem,
                'a 2-D tt.load')
            return
        mask = self._value(op.get_operand(1)) if op.get_num_operands() > 1 \
            else None
        other = 0
        if op.get_num_operands() > 2:
            other = self._const.get(op.get_operand(2).id(), 0)
        addr, size, elem, kind, index = self._address(ptr)
        rid = op.get_result(0).id()
        shape, _rt = _parse_type(str(op.get_result(0).get_type()))
        if shape is None:
            # A load through a pointer that is not a tensor of pointers:
            # ONE element, into a scalar register. `A[m, k]` in a GEMM is
            # this, and it is usually on its way to a tt.splat. Emitting a
            # whole vector load here would read 16 elements and then
            # broadcast a vector register as though it were a scalar one.
            if mask is not None:
                raise Unsupported('a masked scalar load')
            self._v[rid] = self.b.sload(
                addr, size=size, dtype=elem,
                comment='one element, not a vector')
            return
        if kind == 'broadcast':
            # One element into every lane: a scalar load and a splat.
            if mask is not None:
                raise Unsupported(
                    'a masked load of a single broadcast element',
                    because='the mask would select lanes of a value that '
                            'is the same in every lane, so either all of '
                            'the access happens or none of it does')
            self._v[rid] = self.b.splat(
                self.b.sload(addr, size=size, dtype=elem,
                             comment='one element, broadcast to every lane'),
                dtype=elem)
        elif kind == 'contiguous':
            self._v[rid] = self.b.load(addr, mask=mask, other=other,
                                       dtype=elem, size=size)
        else:
            self.target.require('gather_scatter', 'emit v.ld.gather')
            self._v[rid] = self.b.gather(addr, index, mask=mask, other=other,
                                         dtype=elem, size=size)

    def _op_tt_store(self, op):
        ptr, value = op.get_operand(0), op.get_operand(1)
        shape2d, _ = _parse_type(str(value.get_type()))
        if isinstance(shape2d, tuple):
            spec = self._ptr2d.get(ptr.id())
            if spec is None:
                raise Unsupported(
                    'a 2-D store through a pointer this does not recognise',
                    source='doc/isa/TNCA-ISA-v0.1.md sec 9E.2')
            if op.get_num_operands() > 2:
                raise Unsupported('a masked 2-D store')
            name, elem, aff = spec
            tile = self._tile_of(value)
            if tile.kind != 'window' or (
                    self._ELEM_BYTES[tile.dtype] == 4
                    and tile.cols == self.target.vlen):
                if aff.col_stride != 1:
                    raise Unsupported(
                        f"a 2-D store whose columns are {aff.col_stride} "
                        f"elements apart: a row leaves the accumulator as "
                        f"one contiguous vector store")
                esize = self._ELEM_BYTES[elem]
                base = self._arg_value(name)
                if aff.const:
                    base = self.b.sbini('add', base, aff.const * esize,
                                        comment="the tile's origin")
                base = self._add_dynamic(base, aff, esize)
                self._drain_tile(tile, base, aff.row_stride * esize,
                                        elem)
                return
            desc = self._tile_descriptor(name, elem, aff, shape2d,
                                         'a 2-D tt.store')
            zero = self.b.const(0)
            self._dma_out(desc.value, [zero, zero], tile, elem,
                          'a 2-D tt.store')
            return
        mask = self._value(op.get_operand(2)) if op.get_num_operands() > 2 \
            else None
        addr, size, elem, kind, index = self._address(ptr)
        shape, _vt = _parse_type(str(value.get_type()))
        if shape is None:
            if mask is not None:
                raise Unsupported('a masked scalar store')
            self.b.sstore(addr, self._value(value), size=size)
            return
        if kind == 'broadcast':
            raise Unsupported(
                'a vector store through a pointer that does not vary by '
                'lane',
                because='every lane would write the same address, and '
                        'which lane won is not something the ISA decides',
                remedy='store one element, or give the pointer a per-lane '
                       'offset')
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
            if kind == 'tile':
                raise Unsupported(
                    f"a 2-D constant used as a value (type "
                    f"{ttir_value.get_type()})",
                    because='a 2-D block lives in a window or an '
                            'accumulator, and a constant is in neither '
                            'until something puts it there')
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
        tile = self._tiles.get(vid)
        if tile is not None and tile.kind == 'col':
            _, elem = _parse_type(str(ttir_value.get_type()))
            v = self._col_to_vector(tile, elem)
            self._v[vid] = v
            return v
        raise Unsupported(
            f"a value produced by an operation that was not lowered "
            f"(type {ttir_value.get_type()})")
