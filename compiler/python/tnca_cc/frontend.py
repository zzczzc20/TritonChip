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
from .ir import Builder, VECTOR, SCALAR, PRED, make_kernel
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
# 'uo', not 'u': the assembler's condition codes (ISA sec 8.3). The old
# spelling was never reached until `arith.cmpf uno` (an isnan) arrived from
# Inductor, and then failed at assembly rather than at import.
_CMP_F = {1: 'eq', 2: 'gt', 3: 'ge', 4: 'lt', 5: 'le', 6: 'ne',
          7: 'o', 8: 'uo', 13: 'ne'}

_TENSOR_RE = re.compile(r'tensor<((?:\d+x)+)(.+)>')
_PTR_RE = re.compile(r'!tt\.ptr<(.+?)>')

_FP8_FORMATS = {'f8E4M3FN': 'e4m3', 'f8E5M2': 'e5m2'}

_DTYPE_BYTES = {'f32': 4, 'i32': 4, 'f16': 2, 'bf16': 2, 'i16': 2,
                'i8': 1, 'i64': 8, 'f64': 8,
                'f8E4M3FN': 1, 'f8E5M2': 1}
_SIZE_SUFFIX = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}

# Integer element types narrower than a lane. ISA sec 9.1 mod[3] decides
# whether such a load sign-extends, and the assembler spells the
# extending form `.sb` / `.sh`.
_NARROW_SIGNED = {'i8', 'i16'}


def _load_suffix(size, elem):
    """The access-size suffix for a LOAD of `elem`.

    A store has nothing to extend, so it keeps the plain form; a load of
    a narrow signed integer does not. Loading INT8 as `.b` zero-extends,
    which turned every negative weight in a dequantisation into itself
    plus 256 -- a wrong answer with no symptom, since the arithmetic
    that followed was perfectly happy with it.
    """
    if elem in _NARROW_SIGNED and size in ('b', 'h'):
        return 's' + size
    return size


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
    text = re.sub(r'\bsi(8|16|32|64)\b', r'i\1', text)
    text = re.sub(r'\bui(8|16|32|64)\b', r'u\1', text)
    m = _TENSOR_RE.fullmatch(text.strip())
    if m:
        dims = tuple(int(d) for d in m.group(1).rstrip('x').split('x'))
        elem = m.group(2).strip()
        if re.fullmatch(r'si[0-9]+', elem):
            elem = elem[1:]
        return (dims[0] if len(dims) == 1 else dims), elem
    return None, text.strip()


def _pointee(text):
    text = re.sub(r'\bsi(8|16|32|64)\b', r'i\1', text)
    m = _PTR_RE.fullmatch(text.strip())
    return m.group(1).strip() if m else None


class _Descriptor:
    """A TensorMap this compiler built, and what it says.

    The slot alone is not enough: an accumulator drains to memory
    through the vector unit rather than through a transfer, and doing
    that needs the base and the row stride that went INTO the map, not
    a handle to the snapshot of it.
    """

    __slots__ = ('value', 'base', 'byte_strides', 'block', 'elem', 'address', 'extents')

    def __init__(self, value, *, base, byte_strides, block, elem, address=None, extents=None):
        self.value = value
        self.base = base
        self.byte_strides = list(byte_strides)
        self.block = list(block)
        self.elem = elem
        self.address = address
        self.extents = extents


class Unsupported(Diagnostic):
    """A TTIR construct with no TNCA lowering yet."""


class TTIRImporter:
    """Walk a TTIR module and build the TNCA IR for it."""

    def __init__(self, target, name='kernel', scratch_addr=None):
        self.target = target
        # Where a kernel compiled for LITERAL arguments puts its
        # TensorMaps and its repacks. In dynamic mode the address
        # arrives in an argument slot; in static mode there are no
        # argument slots, so the caller that placed the buffers places
        # this too.
        self.scratch_addr = scratch_addr
        self.kernel = make_kernel(name)
        self.b = Builder(self.kernel, self.kernel.role(0).region)
        self.name = name
        self._packed = set()
        self._wide = {}
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
        # Descriptor loads whose only use is tl.trans, lowered at the trans
        # as ONE transposed transfer (ISA sec 9E mod[5]); see _plan_transposes.
        self._xpose_loads = set()
        self._deferred = {}
        self._descs = {}    # TTIR value id -> our descriptor Value
        self._scratch = None      # the descriptor scratch base, read once
        self.scratch_maps = 0     # how many TensorMaps this kernel builds
        self.scratch_used = 0     # bytes of scratch handed out, ever
        self._force_vector = 0    # depth inside a region body
        self._control_seq = 0
        self._atomic_seq = 0      # unique labels for CAS retry loops
        self._lane0_mask = None     # the mask that selects only lane 0
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
        self._plan_transposes(ops)

        ops, self._bodies = self._split_regions(ops)
        # Triton's public Python binding exposes block identity but not
        # successors. Read only successor labels from the printed CFG;
        # values and instructions still come from the MLIR objects.
        self._cfg_edges = []
        cfg_blocks = {}
        if any(op.get_name().startswith('cf.') for op in ops):
            printed = str(module)
            match = re.search(r'(?m)^  tt.func[^\n]*@' + re.escape(func_name) + r'\(', printed)
            if match is None:
                raise Unsupported('cannot locate the function CFG')
            body_text = printed[match.start():]
            body_text = re.split(r'(?m)^  }', body_text, maxsplit=1)[0]
            labels = ['0'] + re.findall(r'(?m)^  \^bb(\d+)', body_text)
            blocks = []
            for op in ops:
                if op.get_name() in ('tt.func', 'builtin.module'):
                    continue
                block = op.get_block()
                if block.id() not in blocks:
                    if block.get_num_arguments() and blocks:
                        raise Unsupported('CFG successor block arguments')
                    blocks.append(block.id())
            if len(blocks) != len(labels):
                raise Unsupported('CFG block labels do not match MLIR traversal')
            cfg_blocks = {bid: 'bb' + label for bid, label in zip(blocks, labels)}
            for line in body_text.splitlines():
                if re.search(r'\bcf\.(?:cond_br|br)\b', line):
                    if re.search(r'\^bb\d+\(', line):
                        raise Unsupported('CFG edge arguments')
                    self._cfg_edges.append(re.findall(r'\^bb(\d+)', line))
        current_block = None
        entry_caches = {n: dict(getattr(self, n)) for n in
                        ('_scalar_cache', '_fconst_cache', '_byte_offsets')}

        for op in ops:
            name = op.get_name()
            if name in ('tt.func', 'builtin.module'):
                continue
            if cfg_blocks and op.get_block().id() != current_block:
                current_block = op.get_block().id()
                self.b.label(cfg_blocks[current_block])
                for key, cache in entry_caches.items():
                    setattr(self, key, dict(cache))
            if name == 'tt.return':
                self.b.exit()
                continue
            from .packed import lower
            from .wide import lower as lower_wide
            if lower_wide(self, op) or lower(self, op):
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

    # ---- structured control flow -----------------------------------------
    def _control_label(self, prefix):
        self._control_seq += 1
        return f"{prefix}_{self._control_seq}"

    def _copy_value(self, value):
        if value.kind == SCALAR:
            return self.b.sbini('add', value, 0)
        if value.kind == VECTOR:
            out = self.b.vmovi(0, dtype=value.dtype)
            self.b._emit('assign', [], [out, value])
            return out
        raise Unsupported('a loop-carried predicate or pointer tensor')

    def _condition_scalar(self, value):
        if value.kind == PRED:
            value = self.b.vselect(value, self.b.vmovi(1), self.b.vmovi(0))
        return self.b.vextract(value, 0) if value.kind == VECTOR else value

    def _op_cf_cond_br(self, op):
        yes, no = self._cfg_edges.pop(0)
        self.b.branch_nz(self._condition_scalar(self._value(op.get_operand(0))), 'bb'+yes)
        self.b.branch_nz(self.b.const(1), 'bb'+no)

    def _op_cf_br(self, op):
        dest, = self._cfg_edges.pop(0)
        self.b.branch_nz(self.b.const(1), 'bb'+dest)

    def _op_scf_while(self, op):
        before = self._bodies[op.get_region(0).id()]
        after = self._bodies[op.get_region(1).id()]
        before_block, after_block = before[0].get_block(), after[0].get_block()
        carried = [self._copy_value(self._value(op.get_operand(i)))
                   for i in range(op.get_num_operands())]
        for i, val in enumerate(carried):
            self._v[before_block.get_argument(i).id()] = val
        start, end = self._control_label('while'), self._control_label('endwhile')
        saved = {n: dict(getattr(self, n)) for n in
                 ('_scalar_cache', '_fconst_cache', '_byte_offsets')}
        saved_single = (self._scratch, self._lane0_mask)
        pending_values = {k for k in self._pending_const if k in self._v}
        self.b.label(start)
        for inner in before:
            name = inner.get_name()
            if name == 'scf.condition':
                for i in range(op.get_num_results()):
                    val = self._copy_value(self._value(inner.get_operand(i+1)))
                    self._v[op.get_result(i).id()] = val
                    self._v[after_block.get_argument(i).id()] = val
                cond = self._condition_scalar(self._value(inner.get_operand(0)))
                self.b.branch_nz(self.b.scmp('eq', cond, self.b.const(0)), end)
            else:
                self._dispatch(inner)
        for inner in after:
            if inner.get_name() == 'scf.yield':
                values = [self._copy_value(self._value(inner.get_operand(i)))
                          for i in range(len(carried))]
                for dst, val in zip(carried, values):
                    self.b._emit('assign', [], [dst, val])
            else:
                self._dispatch(inner)
        self.b.branch_nz(self.b.const(1), start)
        self.b.label(end)
        for key, cache in saved.items():
            setattr(self, key, cache)
        self._scratch, self._lane0_mask = saved_single
        for key in self._pending_const.keys() - pending_values:
            self._v.pop(key, None)

    def _dispatch(self, op):
        from .packed import lower
        from .wide import lower as lower_wide
        if lower_wide(self, op) or lower(self, op):
            return
        handler = getattr(self, '_op_' + op.get_name().replace('.', '_'), None)
        if handler is None:
            raise Unsupported(f'no TNCA lowering for {op.get_name()}')
        handler(op)

    def _op_scf_for(self, op):
        block, body = self._region_block(op)
        lo, hi, step = [self._value(op.get_operand(i)) for i in range(3)]
        iv = self._copy_value(lo)
        self._v[block.get_argument(0).id()] = iv
        carried = []
        pointer_tiles = set()
        mask_tiles = set()
        for i in range(op.get_num_results()):
            src = op.get_operand(3 + i)
            shape, elem = _parse_type(str(src.get_type()))
            if isinstance(shape, tuple) and _pointee(elem):
                name, pelem, offset = self._ptr2d[src.id()]
                val = self._materialise(self._offset_row(offset), shape[0], shape[1], 'i32')
                for dest in (block.get_argument(i + 1), op.get_result(i)):
                    self._ptr2d[dest.id()] = (name, pelem, val)
                pointer_tiles.add(i)
            elif isinstance(shape, tuple) and elem == 'i1':
                # A mask carried round the loop. An i1 tile is a rule
                # recomputed on every read (_materialise), so it has no
                # storage a yield could write: carry it as 0/1 words and
                # read it back as a comparison.
                get = self._row_getter(src, shape[0], 'i1')
                store = self._materialise(
                    lambda r, get=get: self.b.vselect(get(r), self.b.vmovi(1), self.b.vmovi(0)),
                    shape[0], shape[1], 'i32')
                sget = self._tile_row_getter(store)
                val = Tile('lazy', lambda r, sget=sget: self.b.vcmp('ne', sget(r), self.b.vmovi(0)),
                           shape[0], shape[1], 'i1')
                self._tiles[block.get_argument(i + 1).id()] = val
                self._tiles[op.get_result(i).id()] = val
                mask_tiles.add(i)
                val = store
            elif isinstance(shape, tuple):
                get = self._row_getter(src, shape[0], elem)
                val = self._materialise(get, shape[0], shape[1], elem)
                self._tiles[block.get_argument(i + 1).id()] = val
                self._tiles[op.get_result(i).id()] = val
            else:
                pointer = _pointee(elem)
                val = self._copy_value(self._address(src)[0] if pointer else self._value(src))
                self._v[block.get_argument(i + 1).id()] = val
                self._v[op.get_result(i).id()] = val
                if pointer:
                    name = self._control_label('ptr_carry')
                    self._args[len(self._args)] = (name, val)
                    for dest in (block.get_argument(i + 1), op.get_result(i)):
                        self._ptr_base[dest.id()] = (name, pointer)
            carried.append(val)
        start, end = self._control_label('for'), self._control_label('endfor')
        # A loop may execute zero times. Constants cached inside its body
        # must not be reused afterwards as if they had run unconditionally.
        caches = ('_scalar_cache', '_fconst_cache', '_byte_offsets')
        saved = {name: dict(getattr(self, name)) for name in caches}
        saved_single = (self._scratch, self._lane0_mask)
        pending_values = {k for k in self._pending_const if k in self._v}
        self.b.label(start)
        self.b.branch_nz(self.b.scmp('ge', iv, hi), end)
        for inner in body:
            name = inner.get_name()
            if name == 'scf.yield':
                # Snapshot every yielded value before assigning any carry;
                # parallel SSA updates include swaps and cross-dependencies.
                values = []
                for i, dst in enumerate(carried):
                    src = inner.get_operand(i)
                    if isinstance(dst, Tile):
                        if i in pointer_tiles:
                            get = self._offset_row(self._ptr2d[src.id()][2])
                        elif i in mask_tiles:
                            mget = self._row_getter(src, dst.rows, 'i1')
                            get = (lambda r, mget=mget: self.b.vselect(
                                mget(r), self.b.vmovi(1), self.b.vmovi(0)))
                        else:
                            get = self._tile_row_getter(self._tile_of(src))
                        values.append(self._materialise(get, dst.rows, dst.cols, dst.dtype))
                    else:
                        _, elem = _parse_type(str(src.get_type()))
                        values.append(self._copy_value(self._address(src)[0]
                                      if _pointee(elem) else self._value(src)))
                for dst, src in zip(carried, values):
                    if isinstance(dst, Tile):
                        get = self._tile_row_getter(src)
                        for r in range(dst.rows):
                            if dst.kind == 'spilled':
                                at = self.b.sbini('add', self._scratch_base(), dst.handle + r*self.target.vlen*4)
                                self.b.store(at, get(r), size='w')
                            else:
                                self.b.acc_write(dst.handle, self.b.const(r), get(r))
                        if dst.kind == 'spilled':
                            self.b.fence()
                    else:
                        self.b._emit('assign', [], [dst, src])
            else:
                from .packed import lower
                from .wide import lower as lower_wide
                if lower_wide(self, inner) or lower(self, inner):
                    continue
                handler = getattr(self, '_op_' + name.replace('.', '_'), None)
                if handler is None:
                    raise Unsupported(f"no TNCA lowering for TTIR operation {name!r}")
                handler(inner)
        nxt = self.b.sbin('add', iv, step)
        self.b._emit('assign', [], [iv, nxt])
        self.b.branch_nz(self.b.const(1), start)
        self.b.label(end)
        for name, cache in saved.items():
            setattr(self, name, cache)
        self._scratch, self._lane0_mask = saved_single
        for key in self._pending_const.keys() - pending_values:
            self._v.pop(key, None)

    # ---- reductions -------------------------------------------------------
    # The combinator inside a `tt.reduce` region, as the reduction it is.
    # `maximumf`/`minimumf` propagate NaN, and ISA sec 8.7 does not say what
    # `vf.reduce.max` does with one, so they are NOT mapped onto it: they
    # become 'max_prop'/'min_prop', which _reduce_vec lowers from what the
    # ISA does specify (see there).
    _COMBINATOR = {
        'arith.addf': 'add', 'arith.addi': 'add',
        'arith.maxnumf': 'max', 'arith.minnumf': 'min',
        'arith.maximumf': 'max_prop', 'arith.minimumf': 'min_prop',
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
            # An argmax or an argmin: a value and an index folded
            # together. ISA sec 8.7 reduces one vector to one scalar, so
            # there is no instruction for this -- but the body says what
            # the step is, and fifteen steps of it is the answer.
            shape, _ = _parse_type(str(op.get_operand(0).get_type()))
            if isinstance(shape, tuple) and op.get_int_attr('axis') != 1:
                raise Unsupported('a multi-operand reduction along a row block\'s rows')
            return self._fold(op, keep_partials=False)

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
        self._v[op.get_result(0).id()] = self._reduce_vec(
            kind, self._value(src), dtype=rtype, mask=self._shape_mask(shape),
            broadcast=rtype.startswith('f'))

    def _reduce_vec(self, kind, v, *, dtype, mask, broadcast):
        """One ISA sec 8.7 reduction, or a NaN-propagating max/min built from
        specified behaviour.

        `vf.reduce.max` over a vector with no NaN is fully defined whatever the
        hardware does with a NaN. So 'max_prop' is that reduction, plus an OR
        reduction of `x unordered x` over the SAME active lanes; if any lane
        was NaN the result is the canonical qNaN. That is `arith.maximumf`
        folded over the lanes, except for the order of +0 and -0, which the
        plain 'max' path leaves to the hardware as well.
        """
        if kind not in ('max_prop', 'min_prop'):
            return self.b.reduce(kind, v, dtype=dtype, mask=mask, broadcast=broadcast)
        base = kind[:3]
        r = self.b.reduce(base, v, dtype=dtype, mask=mask, broadcast=True)
        nan = self.b.vcmp('uo', v, v)
        if mask is not None:
            nan = self.b.pbin('and', nan, mask)
        flags = self.b.vselect(nan, self.b.vmovi(1), self.b.vmovi(0))
        any_nan = self.b.reduce('or', flags, dtype='i32', broadcast=True)
        is_nan = self.b.vcmp('ne', any_nan, self.b.vmovi(0))
        out = self.b.vselect(is_nan, self._fconst(float('nan'), dtype), r)
        if not broadcast:
            raise Unsupported(f'a NaN-propagating {base} reduction of {dtype} to a scalar register')
        return out

    # ---- atomics ----------------------------------------------------------
    #
    # ISA sec 9.2's atomics are scalar and 4 or 8 bytes wide: there is
    # no vector atomic and no sub-word one, because the memory's word is
    # eight bytes and a narrower read-modify-write would need a path
    # nothing else in the design has. So an atomic over a tensor is one
    # atomic per element, and there is nothing cheaper to lower it to.

    # Triton's ATOMIC_OP, to the ISA's names. FADD is deliberately
    # absent: sec 9.2 has no floating-point atomic and says so -- "FP32
    # `atom.add` is expanded to a `atom.cas` loop by the compiler in
    # v0.1" -- so it is expanded here rather than mapped to an
    # instruction that does something else.
    _ATOMIC_OP = {1: 'and', 2: 'or', 3: 'xor', 4: 'add', 6: 'max',
                  7: 'min', 8: 'maxu', 9: 'minu', 10: 'xchg'}
    _ATOMIC_SEM = {1: 'relaxed', 2: 'acquire', 3: 'release',
                   4: 'acq_rel'}
    _ATOMIC_SCOPE = {1: 'device', 2: 'core', 3: 'system'}

    def _op_llvm_intr_assume(self, op):
        """`tl.assume(cond)`: a fact the program promises, for the optimiser.

        It changes no value -- a program whose assumption is false is already
        undefined -- so lowering it to nothing is exact. It is listed here
        rather than skipped so that "every operation lowers explicitly" stays
        true: an unknown operation is still refused.
        """
        return

    # TRAP_USER code a failed `tl.device_assert` raises (ISA sec 10:
    # `trap #imm24`, code in TRAP_CODE[23:0]). The runtime reports it as a
    # FAULT with this code, so an assertion is told apart from a hardware
    # fault by its number.
    ASSERT_TRAP_CODE = 0xA55E7

    def _op_tt_assert(self, op):
        """`tl.device_assert(cond)`: trap if any active element is false.

        Inductor emits one for every indirect index (`0 <= idx < size` on
        an embedding lookup), so it is on the path of every language
        model. It is a check, not a hint: lowering it to nothing would let
        an out-of-range index read whatever memory it lands on. Lanes past
        the block's length are not elements and are not checked.
        """
        cond = op.get_operand(0)
        shape, _elem = _parse_type(str(cond.get_type()))

        def failing(v, width):
            """A scalar, non-zero if any of the first `width` lanes is false."""
            if v.kind == PRED:
                bad = self.b.vselect(v, self.b.vmovi(0), self.b.vmovi(1))
            elif v.kind == VECTOR:
                bad = self.b.vselect(self.b.vcmp('eq', v, self.b.vmovi(0)),
                                     self.b.vmovi(1), self.b.vmovi(0))
            else:
                return self.b.scmp('eq', v, self.b.const(0))
            return self.b.reduce('or', bad, dtype='i32',
                                 mask=self._shape_mask(width, None))

        if isinstance(shape, tuple):
            # A 2-D condition: each row of its tile is one register.
            if len(shape) != 2 or shape[1] > self.target.vlen:
                raise Unsupported(
                    f"a tl.device_assert over a {'x'.join(map(str, shape))} block",
                    source='compiler_overview.md sec 12')
            tile = self._tiles.get(cond.id())
            if tile is not None:
                get = self._tile_row_getter(tile)
                any_bad = None
                for r in range(shape[0]):
                    f = failing(get(r), shape[1])
                    any_bad = f if any_bad is None else self.b.sbin('or', any_bad, f)
            elif shape[0] * shape[1] <= self.target.vlen:
                any_bad = failing(self._value(cond), shape[0] * shape[1])
            else:
                raise Unsupported(
                    f"a tl.device_assert over a {'x'.join(map(str, shape))} block "
                    f"held neither as a tile nor in one register")
        else:
            any_bad = failing(self._value(cond), shape if shape is not None else 1)
        ok = self._control_label('assert_ok')
        self.b.branch_nz(self.b.scmp('eq', any_bad, self.b.const(0)), ok)
        self.b._emit('trap', [], [], code=self.ASSERT_TRAP_CODE)
        self.b.label(ok)

    def _op_ttg_barrier(self, op):
        """`tl.debug_barrier`, as ISA sec 9.2's fence.

        A kernel that exchanges values between lanes through memory --
        which is what a `tl.load` from a lane-varying index is, whatever
        the SHUF family could have done instead -- depends on the store
        before it being visible to that load. The ISA does not state
        that for the same address from the same context, so the barrier
        is not decoration: it is the ordering the kernel depends on,
        made explicit.
        """
        self.b.fence(order='acq_rel', scope='device')

    def _op_tt_atomic_rmw(self, op):
        kind = op.get_int_attr('atomic_rmw_op')
        sem = self._ATOMIC_SEM.get(op.get_int_attr('sem'), 'relaxed')
        scope = self._ATOMIC_SCOPE.get(op.get_int_attr('scope'), 'device')
        ptr, value = op.get_operand(0), op.get_operand(1)
        shape, elem = _parse_type(str(value.get_type()))
        esize = self._ELEM_BYTES.get(elem)
        if esize not in (4, 8):
            raise Unsupported(f'an atomic on {elem}: ISA supports only 4/8 bytes')
        if elem.startswith('f') and (kind != 5 or elem != 'f32'):
            raise Unsupported(f'floating atomic operation {kind} on {elem}')
        size = _SIZE_SUFFIX[esize]
        mask = op.get_operand(2) if op.get_num_operands() > 2 else None
        if mask is not None and self._is_all_true(mask):
            mask = None
        if isinstance(shape, tuple):
            rows, cols = shape
            name, _, offset = self._ptr2d[ptr.id()]
            base = self._arg_value(name)
            get_index = self._offset_row(offset)
            get_value = self._row_getter(value, rows, elem)
            get_mask = self._row_getter(mask, rows, 'i1') if mask is not None else lambda r: None
        else:
            rows, cols = 1, shape or 1
            base, _, _, address_kind, index = self._address(ptr)
            if address_kind == 'contiguous':
                index = self.b.arange()
            elif address_kind == 'broadcast':
                index = self.b.vmovi(0)
            get_index = lambda r: index
            v = self._vec(value, elem)
            get_value = lambda r: v
            pred = self._value(mask) if mask is not None else None
            get_mask = lambda r: pred

        def atomic_row(r):
            values, offsets, pred = get_value(r), get_index(r), get_mask(r)
            active = (self.b.vselect(pred, self.b.vmovi(1), self.b.vmovi(0))
                      if pred is not None else None)
            result = self.b.vmovi(0, dtype=elem)
            for c in range(cols):
                old = self.b.const(0)
                end = self._control_label('atomic_skip')
                if active is not None:
                    enabled = self.b.vextract(active, c)
                    self.b.branch_nz(self.b.scmp('eq', enabled, self.b.const(0)), end)
                off = self.b.vextract(offsets, c)
                off = self.b.sbini('shl', off, 32)
                off = self.b.sbini('ashr', off, 32)
                addr = self.b.sbin('add', base, self.b.sbini('mul', off, esize))
                v = self.b.vextract(values, c, dtype=elem)
                if elem.startswith('f'):
                    got = self._atomic_fadd(addr, v, elem, sem, scope, size, f'{r},{c}')
                else:
                    got = self.b.atomic(self._ATOMIC_OP[kind], addr, v, size=size,
                                        order=sem, scope=scope, dtype=elem)
                self.b._emit('assign', [], [old, got])
                if active is not None:
                    self.b.label(end)
                self.b.vinsert(result, c, old)
            return result

        if isinstance(shape, tuple):
            self._tiles[op.get_result(0).id()] = self._materialise(atomic_row, rows, cols, elem)
        else:
            result = atomic_row(0)
            self._v[op.get_result(0).id()] = (self.b.vextract(result, 0, dtype=elem)
                                            if shape is None and not elem.startswith('f') else result)

    def _atomic_fadd(self, addr, value, elem, sem, scope, size, label):
        """A floating-point add, as ISA sec 9.2 says to do it.

        There is no FP atomic. The compare-and-swap loop is the
        expansion the ISA names, and the add inside it runs in the
        vector unit because there is no scalar floating point either --
        so each attempt broadcasts the old value, adds, and reads lane
        zero back out with `v.extract`.
        """
        tag = f"amo{self._atomic_seq}"
        self._atomic_seq += 1
        self.b.comment(f"atomic fadd [{label}]: a CAS loop (ISA sec 9.2 "
                       f"has no FP atomic)")
        old = self.b.sload(addr, size=size, dtype=elem,
                           comment='the value to add to')
        addend = self.b.splat(value, dtype=elem)
        self.b.label(tag)
        vsum = self.b.vbin('add', self.b.splat(old, dtype=elem), addend,
                           dtype=elem,
                           comment='the add, in the only unit that has one')
        new = self.b.vextract(vsum, 0, dtype=elem)
        got = self.b.atomic_cas(addr, old, new, size=size, order=sem,
                                scope=scope, dtype=elem)
        again = self.b.scmp('ne', got, old)
        self.b._emit('assign', [], [old, got])
        self.b.branch_nz(again, tag)
        return old

    def _is_all_true(self, value):
        pending = self._pending_const.get(value.id())
        return pending is not None and int(pending[1]) != 0

    # ---- folds with a body -----------------------------------------------
    #
    # `tt.reduce` with one input and a single-instruction combinator is
    # an ISA sec 8.7 reduction. Everything else that carries a region --
    # an argmax, which folds a value and an index together, and
    # `tt.scan`, which keeps every partial result -- is a sequential
    # fold, and the body is the step.
    #
    # The step is evaluated by BINDING the region's block arguments and
    # dispatching the body through the ordinary lowerings. That is why
    # an argmax needs no pattern-matching against the shape Triton's
    # `tl.argmax` happens to expand to: whatever the body computes, the
    # same handlers that lower it anywhere else lower it here.
    #
    # Everything in the fold is a broadcast vector, including the
    # integers. The body's operands have scalar types and there is no
    # scalar floating point, no scalar predicate and no scalar select
    # of a float -- so running it in the vector unit is not a choice
    # between two working forms, it is the one that exists.

    def _region_block(self, op):
        body = self._bodies.get(op.get_region(0).id(), [])
        if not body:
            raise Unsupported(f"{op.get_name()} with an empty body")
        return body[0].get_block(), body

    def _eval_region(self, op, args):
        """Run the region body once, on `args`. Returns its results."""
        block, body = self._region_block(op)
        n = block.get_num_arguments()
        if n != len(args):
            raise Unsupported(
                f"{op.get_name()} whose body takes {n} arguments and was "
                f"given {len(args)}")
        for i, value in enumerate(args):
            self._v[block.get_argument(i).id()] = value

        self._force_vector += 1
        try:
            out = None
            for inner in body:
                name = inner.get_name()
                if name in ('tt.reduce.return', 'tt.scan.return'):
                    out = [self._value(inner.get_operand(i))
                           for i in range(inner.get_num_operands())]
                    continue
                handler = getattr(self, '_op_' + name.replace('.', '_'),
                                  None)
                if handler is None:
                    raise Unsupported(
                        f"no TNCA lowering for {name!r} inside a "
                        f"{op.get_name()} body")
                handler(inner)
        finally:
            self._force_vector -= 1
        if out is None:
            raise Unsupported(f"a {op.get_name()} body that does not return")
        return out

    def _one_row(self, value):
        """(vector, length) for a 1-D block or a one-row 2-D block (1, n).

        Inductor's reductions run one row per program (tnca.inductor), so
        what they fold arrives as `(1, n)`, reduced along axis 1 -- the
        lanes of one register, the same as a 1-D block along axis 0.
        """
        shape, elem = _parse_type(str(value.get_type()))
        if isinstance(shape, tuple):
            if len(shape) != 2 or shape[0] != 1 or shape[1] > self.target.vlen:
                raise Unsupported(f"a fold over a {'x'.join(map(str, shape))} block")
            return self._row_getter(value, 1, elem)(0), shape[1]
        return self._value(value), shape

    def _lanes_of(self, value):
        """Every lane of a 1-D (or one-row) value, each broadcast to a whole vector."""
        _, elem = _parse_type(str(value.get_type()))
        v, n = self._one_row(value)
        col = self._vector_to_col(v, elem, n)
        get = self._tile_row_getter(col, elem)
        return [get(r) for r in range(n)]

    def _fold(self, op, *, keep_partials):
        """A sequential fold over the lanes, body-defined.

        Sixteen lanes and fifteen steps, unrolled. A loop would need the
        running value to live across a back edge in a register the body
        does not name, and the body is inlined rather than called.
        """
        nin = op.get_num_operands()
        lanes = [self._lanes_of(op.get_operand(i)) for i in range(nin)]
        vlen = len(lanes[0])

        running = [lanes[i][0] for i in range(nin)]
        partials = [list(running)]
        for r in range(1, vlen):
            running = self._eval_region(
                op, running + [lanes[i][r] for i in range(nin)])
            partials.append(list(running))

        if not keep_partials:
            for i in range(op.get_num_results()):
                self._v[op.get_result(i).id()] = running[i]
            return
        for i in range(op.get_num_results()):
            rshape, elem = _parse_type(str(op.get_result(i).get_type()))
            col = Tile('col', tuple(p[i] for p in partials), vlen, 1, elem)
            v = self._col_to_vector(col, elem)
            if isinstance(rshape, tuple):
                # A one-row scan: the result is a one-row tile.
                self._tiles[op.get_result(i).id()] = Tile('rows', [v], 1, rshape[1], elem)
            else:
                self._v[op.get_result(i).id()] = v

    def _op_tt_scan(self, op):
        """An inclusive scan.

        ISA sec 8.7 has a reduction and no scan. This is the fold
        above with every partial result kept, taken apart and put back
        together with `v.extract` and `v.insert`.
        """
        shape, _ = _parse_type(str(op.get_operand(0).get_type()))
        lanes_axis = 1 if (isinstance(shape, tuple) and len(shape) == 2 and shape[0] == 1) else 0
        if op.get_int_attr('axis') != lanes_axis:
            raise Unsupported('a scan along an axis other than the lanes')
        if op.get_bool_attr('reverse'):
            raise Unsupported('a reversed scan')
        self._fold(op, keep_partials=True)

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
        # Triton flattens a descriptor argument into its handle, shape and
        # strides. Keep the host-facing descriptor record together until this
        # ABI boundary, then bind every actual MLIR argument explicitly.
        self.argument_slots = len(param_names)
        descriptor_fields = {}
        argument_indices = {}
        expanded_names, expanded_values = [], {}
        cursor = 0
        for slot, name in enumerate(param_names):
            if cursor >= n:
                break
            ty = str(fn.args(cursor).get_type())
            if ty.startswith('!tt.tensordesc<'):
                record = arg_values.get(name)
                if self.dynamic_args:
                    shape, elem = _parse_type(ty.replace('!tt.tensordesc<', 'tensor<'))
                    rank = len(shape)
                    record = dict(address=record, shape=[0]*rank, strides=[0]*rank)
                    size = 1 if elem.startswith('f8') or elem == 'i8' else 2 if elem in ('f16', 'bf16') else 4
                    for j in range(rank):
                        descriptor_fields[f'{name}.shape{j}'] = (slot, 8+8*j, 1)
                        descriptor_fields[f'{name}.stride{j}'] = (slot, 48+8*j, size)
                if not isinstance(record, dict):
                    raise Unsupported(f"descriptor argument {name!r} needs "
                                      "address, shape and strides")
                rank = len(record['shape'])
                names = [name] + [f'{name}.shape{i}' for i in range(rank)]
                names += [f'{name}.stride{i}' for i in range(rank)]
                values = [record['address']] + list(record['shape']) + list(record['strides'])
                argument_indices[name] = slot
                expanded_names.extend(names)
                expanded_values.update(zip(names, values))
                cursor += len(names)
            else:
                argument_indices[name] = slot
                expanded_names.append(name)
                if name in arg_values:
                    expanded_values[name] = arg_values[name]
                cursor += 1
        if len(expanded_names) != len(param_names):
            param_names, arg_values = expanded_names, expanded_values
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
                if name in descriptor_fields:
                    slot, offset, size = descriptor_fields[name]
                    address = self.b.sload(arg_base, offset=8*slot, size='d')
                    v = self.b.sload(address, offset=offset, size='d')
                    if size > 1:
                        v = self.b.sbini('ashr', v, size.bit_length()-1)
                else:
                    v = self._load_argument(arg_base, argument_indices[name], name, ty)
            else:
                literal = self._literal_argument(value, ty, sig, i, name)
                v = self._scalar_cache.get(literal)
                if v is None:
                    v = self.b.const(literal)
                    self._scalar_cache[literal] = v
            self._v[arg.id()] = v
            if not self.dynamic_args and ty.startswith('i'):
                self._pending_const[arg.id()] = ('scalar', literal, ty)
            self._args[i] = (name, v)
            if _pointee(ty):
                self._ptr_base[arg.id()] = (name, _pointee(ty))
            elif ty.startswith('!tt.tensordesc<'):
                shape, elem = _parse_type(ty.replace('!tt.tensordesc<', 'tensor<'))
                if not isinstance(shape, tuple) or any(d != 1 for d in shape[:-2]):
                    raise Unsupported(f'external descriptor type {ty}')
                self._descs[arg.id()] = _Descriptor(
                    self.b.descriptor(v), base=self.b.sload(v, size='d'),
                    byte_strides=[self.b.sload(v, offset=48 + 8*j, size='d')
                                  for j in range(len(shape))],
                    block=list(shape), elem=elem, address=v)

    @staticmethod
    def _literal_argument(value, ty, sig, index, name):
        """One argument as the bits a scalar register would hold.

        For everything but a float that is the value itself. For a
        float it is the binary32 pattern, which is what the dynamic
        path's `s.ld.w` puts in the register -- the two modes must
        differ in where the number comes from and not in what it is.
        """
        declared = str(sig[index] if index < len(sig) and sig[index]
                       else ty).strip()
        if declared in ('f32', 'fp32', 'float32'):
            return struct.unpack('<I', struct.pack('<f', float(value)))[0]
        return value

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
        kind = 'ptr' if _pointee(ty) or ty.startswith('!tt.tensordesc<') else str(ty).strip()
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

        if is_float:
            # A float parameter takes a number in BOTH modes. Refusing
            # one in static mode was wrong rather than strict: the
            # dynamic path loads the slot's four bytes into a scalar
            # register and the register then holds the f32 bit pattern,
            # so the static path materialising that same pattern is the
            # identical kernel with the load folded away.
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
        # `\S+` up to the first blank, not up to ` :` -- a boolean prints as
        # `arith.constant true` with no type after it, and requiring the
        # colon dropped it from the list, which then no longer lined up with
        # the walk and left every float constant in the kernel unreadable.
        printed = re.findall(r'=\s*arith\.constant\s+(\S+)', text)
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
        if isinstance(shape, int) and 0 < shape <= vlen:
            return
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
        start = op.get_int_attr('start') or 0
        if start:
            v = self.b.vbin('add', v, self.b.vmovi(start))
        self._v[op.get_result(0).id()] = v
        self._offsets[op.get_result(0).id()] = ('arange' if not start else 'indexed', None)

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
                if self._ptr_chain.get(src.id()):
                    address, _, _, kind, _ = self._address(src)
                    if kind != 'broadcast':
                        raise Unsupported('2-D splat of a lane-varying pointer')
                    name = self._control_label('ptr_origin')
                    self._args[len(self._args)] = (name, address)
                self._ptr2d[rid] = (name, pelem, Affine2D())
                return
            if elem.startswith('f'):
                # `sm_scale` over the whole tile. Broadcast once; every
                # row names the same register.
                v = self._broadcast_scalar(self._value(src), elem)
                self._tiles[rid] = Tile('lazy', lambda r: v, shape[0],
                                        shape[1], elem)
                return
            if elem == 'i1':
                # A condition over the whole tile -- `idx_q < BATCH` in
                # Inductor's batched-GEMM store mask. It is a predicate,
                # not a term of an index: taken as one (below) its rows
                # came back as 0/1 integers, and ANDed into the mask they
                # turned every lane off, so the tile was never stored.
                v = self._value(src)
                if v.kind == PRED:
                    p = v
                else:
                    if v.kind != VECTOR:
                        v = self.b.splat(v, dtype='i32')
                    p = self.b.vcmp('ne', v, self.b.vmovi(0))
                self._tiles[rid] = Tile('lazy', lambda r, p=p: p, shape[0],
                                        shape[1], 'i1')
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
        if v.kind in (VECTOR, PRED):
            # Already broadcast to every lane -- a reduced float, or a
            # float constant. Splatting it again would emit a `v.splat`
            # that reads a vector register, which is not what that
            # instruction does.
            self._v[rid] = v
        else:
            self._v[rid] = self.b.splat(v, dtype=elem)
        k = self._scalar_const(src.id())
        if k is not None and elem.startswith('i'):
            self._pending_const[rid] = ('vector', k, elem)
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

    def _integer_division(self, op, signed, remainder):
        """Exact restoring division; never round indices through floats."""
        divisor_const = self._scalar_const(op.get_operand(1).id())
        pending = self._pending_const.get(op.get_operand(1).id())
        if pending is not None:
            divisor_const = int(pending[1])
        def divide(xs, elem):
            if elem not in ('i32', 'i8', 'i16'):
                raise Unsupported(f'integer vector division on {elem}')
            a, divisor = xs
            zero, one = self.b.vmovi(0), self.b.vmovi(1)
            if signed:
                aneg = self.b.vcmp('lt', a, zero)
                bneg = self.b.vcmp('lt', divisor, zero)
                a = self.b.vselect(aneg, self.b.vbin('sub', zero, a), a)
                divisor = self.b.vselect(bneg, self.b.vbin('sub', zero, divisor), divisor)
            if (divisor_const is not None and divisor_const > 0
                    and divisor_const & (divisor_const - 1) == 0):
                result = (self.b.vbin('and', a, self.b.vmovi(divisor_const-1))
                          if remainder else self.b.vbin('lshr', a,
                              self.b.vmovi(divisor_const.bit_length()-1)))
                return (self.b.vselect(aneg, self.b.vbin('sub', zero, result), result)
                        if signed else result)
            q, rem = self.b.vmovi(0), self.b.vmovi(0)
            bit = self.b.const(31)
            label = self._control_label('idiv')
            self.b.label(label)
            shift = self.b.splat(bit)
            digit = self.b.vbin('and', self.b.vbin('lshr', a, shift), one)
            r = self.b.vbin('or', self.b.vbin('shl', rem, one), digit)
            overflow = self.b.vcmp('lt', rem, zero)
            take = self.b.pbin('or', overflow, self.b.vcmp('geu', r, divisor))
            rnext = self.b.vselect(take, self.b.vbin('sub', r, divisor), r)
            qnext = self.b.vbin('or', q, self.b.vselect(
                take, self.b.vbin('shl', one, shift), zero))
            self.b._emit('assign', [], [rem, rnext])
            self.b._emit('assign', [], [q, qnext])
            self.b._emit('assign', [], [bit, self.b.sbini('sub', bit, 1)])
            self.b.branch_nz(self.b.scmp('ge', bit, self.b.const(0)), label)
            result = rem if remainder else q
            if signed:
                neg = aneg if remainder else self.b.pbin('xor', aneg, bneg)
                result = self.b.vselect(neg, self.b.vbin('sub', zero, result), result)
            return result
        if self._is_2d(op):
            self._tile_binary(op, divide)
            return
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        result = divide([self._vec(op.get_operand(i), elem) for i in range(2)], elem)
        if shape is None:
            result = self.b.vextract(result, 0)
            if signed:
                result = self.b.sbini('ashr', self.b.sbini('shl', result, 32), 32)
        self._v[op.get_result(0).id()] = result

    def _op_arith_divsi(self, op):
        self._integer_division(op, True, False)

    def _op_arith_divui(self, op):
        self._integer_division(op, False, False)

    def _op_arith_remsi(self, op):
        self._integer_division(op, True, True)

    def _op_arith_remui(self, op):
        self._integer_division(op, False, True)

    def _op_arith_extsi(self, op):
        shape, src = _parse_type(str(op.get_operand(0).get_type()))
        _, dst = _parse_type(str(op.get_result(0).get_type()))
        if src == 'i1':
            self._extend_bool(op, signed=True)
            return
        if isinstance(shape, tuple):
            bits = int(src[1:])
            self._tile_binary(op, lambda xs, elem: self.b.vbin('ashr',
                self.b.vbin('shl', xs[0], self.b.vmovi(32-bits), dtype='i32'),
                self.b.vmovi(32-bits), dtype='i32'))
            return
        value = self._value(op.get_operand(0))
        if shape is None and dst == 'i64':
            if value.kind == VECTOR:
                value = self.b.vextract(value, 0)
            bits = int(src[1:])
            value = self.b.sbini('ashr', self.b.sbini('shl', value, 64-bits), 64-bits)
        elif dst == 'i64':
            raise Unsupported('a vector of 64-bit integers')
        self._v[op.get_result(0).id()] = value

    def _op_arith_extui(self, op):
        if _parse_type(str(op.get_operand(0).get_type()))[1] == 'i1':
            self._extend_bool(op, signed=False)
            return
        self._op_arith_trunci(op)

    def _extend_bool(self, op, signed):
        shape, _ = _parse_type(str(op.get_result(0).get_type()))
        true = -1 if signed else 1
        def extend(pred):
            return self.b.vselect(pred, self.b.vmovi(true), self.b.vmovi(0))
        if isinstance(shape, tuple):
            self._tile_binary(op, lambda xs, elem: extend(xs[0]))
        else:
            value = extend(self._value(op.get_operand(0)))
            if shape is None:
                value = self.b.vextract(value, 0)
                if signed:
                    value = self.b.sbini('ashr', self.b.sbini('shl', value, 32), 32)
            self._v[op.get_result(0).id()] = value

    def _op_arith_trunci(self, op):
        src = op.get_operand(0)
        shape, stype = _parse_type(str(src.get_type()))
        _, dtype = _parse_type(str(op.get_result(0).get_type()))
        if self._is_2d(op):
            bits = min(int(stype[1:]), int(dtype[1:]))
            self._tile_binary(op, lambda xs, elem: self.b.vbin(
                'and', xs[0], self.b.vmovi((1 << bits) - 1), dtype='i32'))
            return
        value = self._value(src)
        bits = min(int(stype[1:]), int(dtype[1:]))
        mask = (1 << bits) - 1
        if value.kind == SCALAR:
            value = self.b.sbin('and', value, self.b.const(mask))
        else:
            value = self.b.vbin('and', value, self.b.vmovi(mask))
        self._v[op.get_result(0).id()] = value

    def _op_tt_reshape(self, op):
        from .packed import shape, value
        import math
        src, dst = op.get_operand(0), op.get_result(0)
        dims, elem = shape(dst)
        tile = self._tiles.get(src.id())
        if tile is not None and tuple(d for d in dims if d != 1) == tuple(
                d for d in (tile.rows, tile.cols) if d != 1):
            self._tiles[dst.id()] = tile
            return
        if math.prod(dims) > self.target.vlen:
            raise Unsupported(f'a reshape to {dims} wider than one vector')
        self._v[dst.id()] = value(self, src)
        self._packed.add(dst.id())

    def _op_tt_join(self, op):
        shape, elem = _parse_type(str(op.get_operand(0).get_type()))
        if not isinstance(shape, int) or shape * 2 > self.target.vlen:
            raise Unsupported('a join wider than one vector')
        a, b = [self._value(op.get_operand(i)) for i in range(2)]
        out = self.b.vmovi(0, dtype=elem)
        for i in range(shape):
            self.b.vinsert(out, 2*i, self.b.vextract(a, i))
            self.b.vinsert(out, 2*i+1, self.b.vextract(b, i))
        self._v[op.get_result(0).id()] = out
        self._packed.add(op.get_result(0).id())

    def _op_tt_bitcast(self, op):
        src, dst = op.get_operand(0), op.get_result(0)
        _, elem = _parse_type(str(dst.get_type()))
        if _pointee(elem):
            if src.id() in self._ptr_base:
                name, _ = self._ptr_base[src.id()]
                self._ptr_base[dst.id()] = (name, _pointee(elem))
                self._ptr_chain[dst.id()] = list(self._ptr_chain.get(src.id(), []))
            elif src.id() in self._ptr2d:
                name, _, off = self._ptr2d[src.id()]
                self._ptr2d[dst.id()] = (name, _pointee(elem), off)
            return
        shape, selem = _parse_type(str(src.get_type()))
        if isinstance(shape, tuple):
            # A tile reinterpreted (Inductor's select_one sums a float row
            # as integer bits). Same width, same registers: only the meaning
            # changes, so the rows are read as they are.
            if self._ELEM_BYTES.get(selem) != self._ELEM_BYTES.get(elem):
                raise Unsupported(f'a {selem} -> {elem} tile bitcast changes the element width')
            get = self._row_getter(src, shape[0], selem)
            self._tiles[dst.id()] = Tile('lazy', get, shape[0], shape[1], elem)
            return
        self._v[dst.id()] = self._value(src)

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
            # repeated; the first takes the value's lanes apart with
            # `v.extract`, and the second does not have to.
            if axis == 0:
                shape, elem = _parse_type(
                    str(op.get_result(0).get_type()))
                v = self._value(src)
                self._tiles[op.get_result(0).id()] = Tile(
                    'lazy', lambda r: v, shape[0], shape[1], elem)
                return
            # `x[:, None]`: lane r of one register becomes the whole
            # of row r, which is `v.extract` (ISA 0x08) sixteen times
            # and then a broadcast of each.
            _, selem = _parse_type(str(src.get_type()))
            shape, _ = _parse_type(str(op.get_result(0).get_type()))
            self._tiles[op.get_result(0).id()] = self._vector_to_col(
                self._value(src), selem, shape[0])
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
            return False
        rid = op.get_result(0).id()
        if kind == 'add':
            self._affine[rid] = a.plus(b)
            return True
        if kind == 'sub':
            self._affine[rid] = a.plus(b.scaled(-1))
            return True
        if kind == 'mul':
            # One side must be a constant: an index that multiplies two
            # varying terms is not affine, and no descriptor describes it.
            for x, y in ((a, b), (b, a)):
                if (y.row_stride, y.col_stride, y.dynamic) == (0, 0, []):
                    self._affine[rid] = x.scaled(y.const)
                    return True
            return False
        raise Unsupported(f"a 2-D index built with {op.get_name()!r}")

    def _try_addptr_2d(self, rid, base, offs):
        """Accumulate a 2-D pointer's index. True if this was one.

        A 2-D pointer arrives as `splat(p)` broadcast to the tile shape,
        with each `tt.addptr` contributing one affine term. Nothing is
        emitted: what the chain adds up to becomes a descriptor when the
        load at the end of it asks.
        """
        aff = self._affine_of(offs)
        offset_tile = self._tiles.get(offs.id())
        if aff is None and offset_tile is None:
            return False
        prev = self._ptr2d.get(base.id())
        if prev is None:
            spec = self._ptr_base.get(base.id())
            if spec is None:
                return False
            shape, _ = _parse_type(str(base.get_type()))
            result_shape, _ = _parse_type(str(offs.get_type()))
            if not isinstance(result_shape, tuple) and not isinstance(shape, tuple):
                return False
            name, elem = spec
            prev = (name, elem, Affine2D())
        name, elem, acc = prev
        if aff is not None and isinstance(acc, Affine2D):
            combined = acc.plus(aff)
        else:
            shape, _ = _parse_type(str(offs.get_type()))
            if not isinstance(shape, tuple):
                shape = (offset_tile.rows, offset_tile.cols)
            left, right = self._offset_row(acc), self._offset_row(aff or offset_tile)
            combined = self._materialise(
                lambda r: self.b.vbin('add', left(r), right(r), dtype='i32'),
                shape[0], shape[1], 'i32')
        self._ptr2d[rid] = (name, elem, combined)
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

    def _load_tile_rows(self, name, elem, aff, shape):
        """A tile whose element type has no usable fragment, row by row.

        ISA sec 9.1 gives lane i of a vector load the address
        `base + i*sz` and sign-extends a narrow integer per `mod[3]`, so
        one `v.ld.b` puts a whole row of sixteen INT8s in sixteen lanes
        already widened. That is the form arithmetic wants, and it never
        passes through the packed layout that has no unpack.
        """
        rows, cols = shape
        if aff.col_stride != 1:
            raise Unsupported(
                f"a tile whose columns are {aff.col_stride} elements apart",
                because='a row is one vector load, which is contiguous')
        if cols != self.target.vlen:
            raise Unsupported(
                f"a {elem} tile {cols} elements wide on a VLEN="
                f"{self.target.vlen} machine")
        esize = self._ELEM_BYTES[elem]
        base = self._arg_value(name)
        if aff.const:
            base = self.b.sbini('add', base, aff.const * esize,
                                comment="the tile's origin")
        base = self._add_dynamic(base, aff, esize)
        self.b.comment(f"a {rows}x{cols} {elem} tile, one row per vector "
                       f"load: {elem} has no fragment this can unpack")
        size = _load_suffix(_SIZE_SUFFIX[esize], elem)
        stride = aff.row_stride * esize

        def row(r):
            at = base if r == 0 else self.b.sbini('add', base, r * stride,
                                                  comment=f"row {r}")
            return self.b.load(at, dtype=elem, size=size)

        # Through the same placement rule as a computed tile, so a
        # 32-bit one lands in an accumulator rather than holding sixteen
        # vector registers from the load to the last use of it. A mask
        # tile loaded eagerly ran the allocator out before the attention
        # it was masking got a register.
        return self._materialise(row, rows, cols, elem)

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

    _ELEM_BYTES = {'f16': 2, 'bf16': 2, 'f32': 4, 'i8': 1, 'i32': 4,
                   'f8E4M3FN': 1, 'f8E5M2': 1}

    # Which element types a tile can be TRANSFERRED as. The list is ISA
    # sec 9D.3's dtype codes narrowed to the ones this backend can then
    # read back: `i8` has a code but a packed i8 fragment cannot be
    # taken apart (sec 8.4 reserves v.unpack), and a 32-bit integer has
    # no code at all. Both load a row at a time instead, which needs no
    # descriptor and no unpacking -- one `v.ld` puts sixteen elements in
    # sixteen lanes whatever their width, sign-extended per sec 9.1.
    _TRANSFERABLE = ('f16', 'bf16', 'f32')

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
                if self.scratch_addr is None:
                    raise Unsupported(
                        'a descriptor in a kernel compiled for literal '
                        'arguments and given no scratch address',
                        because='the TensorMap has to be written to memory, '
                                'and in this mode no argument slot says '
                                'where',
                        remedy='pass scratch_addr, or compile with dynamic '
                               'arguments')
                self._scratch = self.b.const(self.scratch_addr)
                return self._scratch
            self._scratch = self.b.sload(
                self.b.special('arg_base'), offset=8 * self.argument_slots,
                size='d',
                comment='descriptor scratch: one slot past the declared '
                        'arguments (ABI v1 offsets are unchanged)')
        return self._scratch

    def _spill_region(self, nbytes):
        """Reserve scratch for a repack, and return its address.

        Repacking through memory is what the architecture leaves for
        this one case: there is no scratchpad (sec 5.2) and the ISA
        reserves `v.pack.f16` and `v.unpack.f16` (sec 8.4), so a
        sixteen-bit fragment cannot be assembled in registers. Taking a
        vector apart by LANE needs no memory -- `v.extract` and
        `v.insert` do that -- and the lowerings that need it use them.

        Rounded to a whole TensorMap so the descriptors that follow
        stay 8-byte aligned, which ISA sec 9E.2 requires of a base.
        """
        return self.b.sbini('add', self._scratch_base(),
                            self._take_scratch(nbytes),
                            comment=f"{nbytes} bytes of repack scratch")

    def _take_scratch(self, nbytes):
        """Hand out `nbytes` of scratch, once. Returns the offset.

        One cursor for descriptors and repacks alike. They used to have
        a counter each, both measured from the same base, so a
        TensorMap built after a repack landed on top of it -- and the
        repack is where a tile is while it changes shape.

        Rounded to eight so the next thing out stays aligned; ISA sec
        9E.2 wants a descriptor's base on an 8-byte boundary, and a
        repacked tile IS a descriptor's base.
        """
        at = self.scratch_used
        self.scratch_used += (nbytes + 7) & ~7
        return at

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
        if out_dtype in _FP8_FORMATS:
            # TensorMap has no FP8 dtype code. Read the packed bytes as
            # words into a register window; the MMA supplies their format.
            # This avoids inventing a descriptor dtype or changing values.
            window = self._window_for(nrows, cols, out_dtype)
            self.b.fence()
            for i in range(window.length):
                at = addr if i == 0 else self.b.sbini(
                    'add', addr, i * self.target.vlen * 4)
                self.b.window_put(window, i, self.b.load(at, dtype='i32', size='w'))
            return Tile('window', window, nrows, cols, out_dtype)
        desc = self._build_descriptor(
            base=addr, elem=out_dtype, extents=[nrows, cols],
            byte_strides=[row_bytes, esize], block_shape=[nrows, cols],
            what='a repacked tile')
        zero = self.b.const(0)
        return self._dma_in(desc, [zero, zero], [nrows, cols],
                            out_dtype, 'a repacked tile')

    def _vector_to_col(self, v, elem, rows=None):
        """One vector register into one value per row.

        `v.extract Sd, Va, #lane` (ISA 0x08) reads a single lane into a
        scalar register, zero-extended -- which for a float is its bit
        pattern, and a scalar register holds a float as its bit pattern
        anyway. Sixteen of those.
        """
        if elem == 'i1':
            v = self.b.vselect(v, self.b.vmovi(1), self.b.vmovi(0))
        def row(r):
            result = self.b.splat(self.b.vextract(v, r), dtype=elem)
            return (self.b.vcmp('ne', result, self.b.vmovi(0))
                    if elem == 'i1' else result)
        return Tile('lazy', row, rows or self.target.vlen, 1, elem)

    def _col_to_vector(self, tile, elem):
        """One value per row, gathered into one vector register.

        `v.insert Vd, Sa, #lane` (ISA 0x09) writes one lane and leaves
        the others, so the register is built up in place from a zeroed
        start -- which also means the sixteen inserts are one value's
        lifetime and not sixteen, and the allocator has to be told so.
        """
        out = self.b.vmovi(0, dtype=elem)
        self.b.comment('one value per row into one vector (ISA 0x09)')
        for r, x in enumerate(tile.handle):
            if x.kind == VECTOR:
                # A broadcast reduction result: every lane is the value,
                # so lane 0 is it.
                x = self.b.vextract(x, 0, dtype=elem)
            self.b.vinsert(out, r, x)
        return out

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
                            self._take_scratch(TENSORMAP_BYTES),
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
                           block=list(block_shape), elem=elem, address=addr, extents=extents)

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

    # Operations that may write memory a deferred load would read.
    _WRITES = ('tt.store', 'tt.descriptor_store', 'tt.atomic', 'tt.descriptor_scatter',
               'tt.descriptor_reduce', 'tt.call', 'scf.', 'cf.')

    def _plan_transposes(self, ops):
        """Which descriptor loads become one transposed transfer.

        ISA sec 9E mod[5] writes the tile transposed while it is in flight.
        On a target with `tensor_transpose`, a 16-bit descriptor load whose
        only user is `tt.trans` is not emitted where it stands; the trans
        emits the transposed transfer instead of loading the tile and then
        gathering its columns (what _op_tt_trans does otherwise).

        Moving a load later is only a no-op if nothing in between can write
        what it reads, so the trans must be in the SAME block with no store,
        atomic, call or nested control flow between the two. Anything else
        keeps the ordinary lowering.
        """
        if not self.target.has('tensor_transpose'):
            return
        users, where = {}, {}
        for i, op in enumerate(ops):
            where[id(op)] = i
            for k in range(op.get_num_operands()):
                users.setdefault(op.get_operand(k).id(), []).append(op)
        for i, op in enumerate(ops):
            if op.get_name() != 'tt.descriptor_load':
                continue
            us = users.get(op.get_result(0).id(), [])
            if not us or any(u.get_name() != 'tt.trans' for u in us):
                continue
            block = op.get_block().id()
            ok = True
            for u in us:
                j = where.get(id(u))
                if j is None or j < i or u.get_block().id() != block:
                    ok = False
                    break
                if any(ops[k].get_name().startswith(self._WRITES) for k in range(i + 1, j)):
                    ok = False
                    break
            if ok:
                self._xpose_loads.add(op.get_result(0).id())

    def _xpose_geometry_ok(self, d):
        """The engine's transposed-write rule (rtl/core/tnca_tensor.v S_TSETUP):
        a 2-D tile of 16-bit elements with 4, 8 or 16 source rows."""
        return (len(d.block) == 2 and d.elem in ('f16', 'bf16')
                and d.block[0] in (4, 8, 16))

    def _op_tt_descriptor_load(self, op):
        d = self._descs[op.get_operand(0).id()]
        coords = [self._value(op.get_operand(1 + i))
                  for i in range(op.get_num_operands() - 1)]
        # A host descriptor is a TensorMap in the engine's own layout
        # (tnca.tensor.TensorDescriptor.pack), so the transfer can use it
        # directly; its run-time geometry is checked by the runtime before
        # launch against the artifact's `dma_descriptors` (see aot.py).
        if (op.get_result(0).id() in self._xpose_loads and d.value is not None
                and self._xpose_geometry_ok(d)):
            self._deferred[op.get_result(0).id()] = (d, coords)
            return
        if d.address is not None or len(d.block) > 2 or d.elem not in ('f16', 'bf16', 'f32'):
            base = self._descriptor_origin(d, coords)
            rows, cols = d.block[-2:]
            extents = d.extents or [self.b.sload(d.address, offset=8+8*i, size='d')
                                     for i in range(len(d.block))]
            get_mask = self._descriptor_mask(d, coords, extents)
            esize = self._ELEM_BYTES[d.elem]
            def row(r):
                at = self._scaled_address(base, r, d.byte_strides[-2])
                return self.b.load(at, dtype=d.elem,
                    size=_load_suffix(_SIZE_SUFFIX[esize], d.elem),
                    mask=get_mask(r), other=0)
            tile = self._materialise(row, rows, cols, d.elem)
            self._tiles[op.get_result(0).id()] = tile
        else:
            self._tiles[op.get_result(0).id()] = self._dma_in(
                d, coords, d.block, d.elem, 'tt.descriptor_load')

    def _descriptor_mask(self, d, coords, extents):
        def row(r):
            mask = None
            for axis, (coord, extent) in enumerate(zip(coords, extents)):
                if isinstance(extent, int):
                    extent = self.b.const(extent)
                pos = self.b.splat(coord)
                if axis == len(coords)-1:
                    pos = self.b.vbin('add', pos, self.b.arange())
                elif axis == len(coords)-2:
                    pos = self.b.vbin('add', pos, self.b.splat(self.b.const(r)))
                valid = self.b.pbin('and', self.b.vcmp('ge', pos, self.b.vmovi(0)),
                                   self.b.vcmp('lt', pos, self.b.splat(extent)))
                mask = valid if mask is None else self.b.pbin('and', mask, valid)
            return self._shape_mask(d.block[-1], mask)
        return row

    def _scaled_address(self, base, index, stride):
        index = self.b.const(index) if isinstance(index, int) else index
        offset = (self.b.sbini('mul', index, stride) if isinstance(stride, int)
                  else self.b.sbin('mul', index, stride))
        return self.b.sbin('add', base, offset)

    def _descriptor_origin(self, d, coords):
        base = d.base
        for coord, stride in zip(coords, d.byte_strides):
            base = self._scaled_address(base, coord, stride)
        return base

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
        coords = [self._value(op.get_operand(i))
                  for i in range(1, op.get_num_operands())
                  if op.get_operand(i).id() != value.id()]
        if tile.kind != 'window':
            # The result is in an accumulator and the destination is
            # described by a TensorMap whose base and stride this
            # compiler wrote. Draining through the vector unit reaches
            # it directly (ISA sec 9B); routing it back through a window
            # would be a transfer to reach an address already in hand.
            extents = d.extents or [self.b.sload(d.address, offset=8+8*i, size='d')
                                    for i in range(len(d.block))]
            self._drain_tile(tile, self._descriptor_origin(d, coords),
                             d.byte_strides[-2], d.elem,
                             self._descriptor_mask(d, coords, extents))
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

    def _dma_in(self, desc, coords, block, elem, what):
        """One asynchronous tile load, and the wait for it.

        Armed for exactly one operation. ISA sec 9E.1 step 8 signals a
        completion once per command however many transactions it expands
        into, which is what makes a static count checkable -- and the
        count being checkable is the only reason to arm it rather than
        poll.
        """
        window = self._window_for(block[-2], block[-1], elem)
        bar = self.b.barrier(name=f"fill{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.comment(f"{what}: one tile into a window (ISA sec 9E)")
        self.b.dma_load(window, desc.value if hasattr(desc, 'value')
                        else desc, coords, tok, dtype=elem)
        self.b.barrier_wait(tok)
        return Tile('window', window, block[-2], block[-1], elem,
                    desc=(desc, coords))

    def _dma_out(self, desc, coords, tile, elem, what):
        if tile.kind != 'window':
            raise Unsupported(
                f"{what} from a tile that is not in a transfer window")
        bar = self.b.barrier(name=f"drain{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.comment(f"{what}: one tile out of a window (ISA sec 9E)")
        self.b.dma_store(tile.handle, desc, coords, tok, dtype=elem)
        self.b.barrier_wait(tok)

    def _drain_tile(self, tile, base, row_stride_bytes, dst_elem, get_mask=None):
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
            if dst_elem != tile.store:
                v = self.b.vcvt(tile.store, dst_elem, v,
                                comment=f"row {r}: {tile.store} -> "
                                        f"{dst_elem} (ISA sec 8.4)")
            addr = base if r == 0 else self._scaled_address(base, r, row_stride_bytes)
            self.b.store(addr, v, size=size, mask=get_mask(r) if get_mask else None)

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
        """`tl.trans`, by reading the source down its columns.

        ISA sec 9E's `mod[5]` asks the transfer engine to write a tile
        transposed, which would make this free. The reference
        implementation does not decode it: `tnca_core.v` passes
        `x_mod[2:0]` to the engine and nothing else, so an emitted
        `t.ld.async.w.t` transfers the tile UNTRANSPOSED and the kernel
        gets a wrong answer rather than a fault. A compiler must not
        emit an instruction whose behaviour the machine does not
        implement, so this does not -- see doc/r37_isa_gaps.md.

        What it does instead is read the columns. Row r of the
        transpose is column r of the source, which is sixteen elements
        `row_stride` apart -- and ISA sec 9.1's `v.ld.gather` addresses
        lane i at `Sa + Vb[i]*sz`, one instruction per output row.

        For a 32-bit tile the gathered rows are values and arithmetic
        can use them. For a narrow one they are raw bit patterns sitting
        in wide lanes -- `v.ld.gather.h` puts two bytes in a 32-bit lane
        -- so those are packed straight back into a fragment, which
        moves the bits without ever interpreting them, and the result is
        usable as an MMA operand and as nothing else.
        """
        src = op.get_operand(0)
        if src.id() in self._deferred:
            d, coords = self._deferred.pop(src.id())
            rows, cols = d.block
            window = self._window_for(cols, rows, d.elem)
            bar = self.b.barrier(name=f"fill{len(self.kernel.barriers)}")
            tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
            self.b.comment('tl.trans of a descriptor tile: one transposed '
                           'transfer (ISA sec 9E mod[5])')
            self.b.dma_load(window, d.value if hasattr(d, 'value') else d, coords,
                            tok, dtype=d.elem, transpose=True)
            self.b.barrier_wait(tok)
            # Window element (c, r) = source (r, c), flat row-major: the
            # layout the gathered path below builds with _rows_to_window.
            self._tiles[op.get_result(0).id()] = Tile('window', window, cols, rows, d.elem)
            return
        tile = self._tile_of(src)
        if tile.desc is None:
            elem = tile.store
            esize = self._ELEM_BYTES[elem]
            addr = self._spill_region(tile.rows * tile.cols * esize)
            self._drain_tile(tile, addr, tile.cols * esize, elem)
            self.b.fence()
            desc = _Descriptor(None, base=addr, byte_strides=[tile.cols*esize, esize],
                               block=[tile.rows, tile.cols], elem=elem)
            tile.desc = (desc, [self.b.const(0), self.b.const(0)])
        desc, coords = tile.desc
        if not hasattr(desc, 'base'):
            raise Unsupported('tl.trans of a tile with no known base')

        elem = tile.dtype
        esize = self._ELEM_BYTES[elem]
        stride = desc.byte_strides[-2]
        row_elems = (self.b.const(stride // esize) if isinstance(stride, int)
                     else self.b.sbini('lshr', stride, esize.bit_length() - 1))
        base = self._descriptor_origin(desc, coords)
        self.b.comment(f"tl.trans: {tile.cols} columns, gathered "
                       f"{row_elems} elements apart (ISA sec 9.1)")
        self.target.require('gather_scatter', 'emit v.ld.gather for tl.trans')

        index = self.b.vbin('mul', self.b.arange(),
                            self.b.splat(row_elems),
                            dtype='i32',
                            comment='one element per source row')
        size = _SIZE_SUFFIX[esize]
        rows = []
        for r in range(tile.cols):
            at = base if r == 0 else self.b.sbini(
                'add', base, r * esize, comment=f"column {r}")
            rows.append(self.b.gather(at, index, dtype=elem,
                                      size=_load_suffix(size, elem)))

        rid = op.get_result(0).id()
        if esize == 4 or elem.startswith('i'):
            self._tiles[rid] = Tile('rows', tuple(rows), tile.cols,
                                    tile.rows, elem)
            return
        # Narrow: the lanes hold bits, not numbers. Straight into a
        # fragment, where they are bits again.
        self._tiles[rid] = self._rows_to_window(
            lambda r: rows[r], elem, elem, tile.rows, nrows=tile.cols)

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
        if a.dtype == b.dtype == 'f32':
            self._dot_f32(op, a, b)
            return
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
        if a.dtype not in ('f16', 'bf16') and a.dtype not in _FP8_FORMATS:
            raise Unsupported(
                f"a tl.dot with {a.dtype} inputs",
                because="ISA sec 9C's mod[3:2] names FP16, BF16 and the two "
                        "FP8 encodings, and reserves the rest -- there is "
                        "no binary32 matrix unit to lower this onto",
                remedy='cast the operands to tl.float16, which is exact for '
                       'an integer input and a choice for a float one',
                source='doc/isa/TNCA-ISA-v0.1.md sec 9C')

        acc = self.b.accumulator(name=f"acc{len(self.kernel.accumulators)}")
        # `tt.dot`'s third operand is the value accumulated into. Triton
        # passes an explicit zero tensor for a plain product, and an
        # accumulator carried across a loop otherwise; only the first is
        # expressible while a tile has no way to enter an accumulator.
        cin = op.get_operand(2)
        if self._is_zero_tensor(cin):
            self.b.acc_zero(acc)
        else:
            _, ctype = _parse_type(str(cin.get_type()))
            get_c = self._row_getter(cin, a.rows, ctype)
            for r in range(a.rows):
                row = get_c(r)
                if ctype.startswith('i'):
                    row = self.b.vcvt('s32', 'f32', row)
                self.b.acc_write(acc, self.b.const(r), row)

        bar = self.b.barrier(name=f"mma{len(self.kernel.barriers)}")
        tok = self.b.barrier_arm(bar, arrivals=0, ops=1)
        self.b.mma_async(acc, a.handle, b.handle, tok,
                         dtype=_FP8_FORMATS.get(a.dtype, a.dtype))
        # ISA sec 9C.3: the accumulator may not be read by the vector
        # unit before the command completes, and its sources may not be
        # overwritten. The wait is what makes both true.
        self.b.barrier_wait(tok)

        _, rtype = _parse_type(str(op.get_result(0).get_type()))
        tile = Tile('accumulator', acc, a.rows, b.cols, 'f32')
        if rtype.startswith('i'):
            # The accumulator is binary32 (sec 9C) and the products were
            # integers, so every value in it IS an integer -- but the IR
            # calls the result i32, and an arith.sitofp on it would
            # otherwise reinterpret a float as one.
            get = self._tile_row_getter(tile)
            tile = self._materialise(
                lambda r: self.b.vcvt('f32', 's32', get(r)),
                a.rows, b.cols, rtype)
        self._tiles[op.get_result(0).id()] = tile

    def _dot_f32(self, op, a, b):
        """Full binary32 dot through vector FMAs, with no FP16 narrowing.

        There is no FP32 MMA encoding. Broadcast each element of an A
        row and FMA the corresponding B row into a vector, then keep that
        result in an accumulator row. This also accepts a nonzero C tile.
        """
        if a.cols != b.rows or a.cols > self.target.vlen:
            raise Unsupported(
                f"an f32 dot of {a.rows}x{a.cols} and {b.rows}x{b.cols}",
                because='the vector dot path needs matching K dimensions '
                        'and one vector per input row')
        get_a = self._tile_row_getter(a)
        get_b = self._tile_row_getter(b)
        cin = op.get_operand(2)
        if self._is_zero_tensor(cin):
            zero = self._fconst(0.0)
            get_c = lambda r: zero
        else:
            get_c = self._row_getter(cin, a.rows, 'f32')

        def row(r):
            av = get_a(r)
            result = get_c(r)
            for k in range(a.cols):
                lane = self.b.splat(self.b.vextract(av, k), dtype='f32')
                result = self.b.vfma(lane, get_b(k), result, dtype='f32')
            return result

        self._tiles[op.get_result(0).id()] = self._materialise(
            row, a.rows, b.cols, 'f32')

    def _as_fragment(self, tile, what):
        """This tile in the layout `m.mma.async` reads."""
        if tile.dtype in ('i8', 'i32'):
            return self._integer_fragment(tile)
        if tile.kind == 'window':
            return tile
        if tile.dtype in _FP8_FORMATS:
            return self._rows_to_window(self._tile_row_getter(tile),
                                        tile.dtype, tile.dtype, tile.cols,
                                        nrows=tile.rows)
        if tile.dtype in ('i8', 'i32'):
            return self._integer_fragment(tile)
        if tile.dtype not in ('f16', 'bf16'):
            raise Unsupported(
                f"{what}: a computed {tile.dtype} tile",
                because="ISA sec 9C's mod[3:2] names FP16, BF16 and the two "
                        "FP8 encodings as input formats, and nothing else",
                remedy='convert it, which tl.dot on a narrower type does '
                       'anyway')
        return self._rows_to_window(self._tile_row_getter(tile), tile.store,
                                    tile.dtype, tile.cols, nrows=tile.rows)

    # The largest K for which an INT8 product accumulated in binary32 is
    # exact. |a*b| <= 128*128 = 2**14, and every partial sum has to stay
    # at or below 2**24, where binary32 still represents every integer.
    INT8_EXACT_K = (1 << 24) // (1 << 14)

    def _integer_fragment(self, tile):
        """An integer tile as an FP16 MMA operand, exactly.

        ISA sec 9C's `mod[3:2]` has FP16, BF16 and the two FP8
        encodings; there is no integer mode. Converting is not an
        approximation of one:

          * every INT8 is an integer of magnitude at most 128, and FP16
            represents every integer up to 2048 exactly, so the operands
            convert without loss;
          * sec 9C.1 forms the product at full width and rounds once per
            accumulation step, so each product -- at most 2**14 -- is
            exact in binary32;
          * every partial sum is an integer of magnitude at most
            K * 2**14, and binary32 represents every integer up to
            2**24, so no step rounds either.

        Which makes the result the exact integer dot product for K up to
        INT8_EXACT_K. Above that a partial sum could round, so it is
        refused rather than quietly approximated.
        """
        if tile.cols > self.INT8_EXACT_K:
            raise Unsupported(
                f"an integer tl.dot with K={tile.cols}",
                because=f"ISA sec 9C has no integer MMA, and the exact FP "
                        f"substitute holds only while every partial sum "
                        f"stays below 2**24 -- K up to "
                        f"{self.INT8_EXACT_K}",
                source='doc/isa/TNCA-ISA-v0.1.md sec 9C.1')
        get = self._tile_row_getter(tile)
        self.b.comment(f"an {tile.dtype} MMA operand as f16: exact for "
                       f"K <= {self.INT8_EXACT_K} (ISA sec 9C.1)")
        rows = [self.b.vcvt('s32', 'f32', get(r),
                            comment=f"row {r}: integer -> f32, exact")
                for r in range(tile.rows)]
        return self._rows_to_window(lambda r: rows[r], 'f32', 'f16',
                                    tile.cols, nrows=tile.rows)

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
        aff = self._affine_of(value)
        if aff is not None:
            return lambda r: self._affine_row(aff, r)
        shape, selem = _parse_type(str(value.get_type()))
        if not isinstance(shape, tuple):
            v = self._vec(value, elem)
            return lambda r: v
        pending = self._pending_const.get(value.id())
        if pending is not None and pending[0] == 'tile':
            # A dense constant over the tile: the same value in every
            # lane of every row, so one register serves all sixteen.
            v = self.b.vmovi(int(pending[1]), dtype=pending[2])
            if elem == 'i1':
                v = self.b.vcmp('ne', v, self.b.vmovi(0))
            return lambda r: v
        raise Unsupported(
            f"a 2-D operand of type {value.get_type()} that is not a tile")

    def _shape_mask(self, width, mask=None):
        if width >= self.target.vlen:
            return mask
        active = self.b.vcmp('lt', self.b.arange(), self.b.splat(self.b.const(width)))
        return active if mask is None else self.b.pbin('and', active, mask)

    def _affine_row(self, aff, r):
        base = self.b.const(aff.const + aff.row_stride * r)
        base = self._add_dynamic(base, aff, 1)
        lanes = self.b.vbin('mul', self.b.arange(),
                           self.b.splat(self.b.const(aff.col_stride)), dtype='i32')
        return self.b.vbin('add', lanes, self.b.splat(base), dtype='i32')

    def _offset_row(self, offset):
        if isinstance(offset, Affine2D):
            return lambda r: self._affine_row(offset, r)
        get = self._tile_row_getter(offset)
        # A pointer broadcast keeps its accumulated offset tile. A 1xN
        # offset therefore repeats row zero when a later addptr makes MxN.
        return (lambda r: get(0)) if offset.rows == 1 else get

    def _lane0(self):
        """A mask with lane 0 active and nothing else."""
        if self._lane0_mask is None:
            self._lane0_mask = self.b.vcmp(
                'lt', self.b.arange(), self.b.splat(self.b.const(1)))
        return self._lane0_mask

    def _broadcast_scalar(self, v, dtype):
        return v if v.kind == VECTOR else self.b.splat(v, dtype=dtype)

    def _tile_row_getter(self, tile, elem=None):
        if tile.kind == 'spilled':
            def row(r):
                at = self.b.sbini('add', self._scratch_base(),
                                  tile.handle + r * self.target.vlen * 4)
                return self.b.load(at, dtype=tile.store, size='w')
            return row
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
        if tile.kind == 'window' and tile.desc is not None and self._ELEM_BYTES[tile.store] < 4:
            desc, coords = tile.desc
            base = self._descriptor_origin(desc, coords)
            def row(r):
                at = self._scaled_address(base, r, desc.byte_strides[-2])
                return self.b.load(at, dtype=tile.store,
                                  size=_load_suffix(_SIZE_SUFFIX[self._ELEM_BYTES[tile.store]], tile.store))
            return row
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
        if self._ELEM_BYTES.get(elem) != 4 or elem == 'i32' or len(self.kernel.accumulators) >= 8:
            offset = self._take_scratch(rows * self.target.vlen * 4)
            addr = self.b.sbini('add', self._scratch_base(), offset)
            for r in range(rows):
                at = addr if r == 0 else self.b.sbini('add', addr, r * self.target.vlen * 4)
                self.b.store(at, rows_fn(r), size='w')
            self.b.fence()
            return Tile('spilled', offset, rows, cols, elem)
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
            if kind in ('max_prop', 'min_prop'):
                # lane-local down the tile: sec 8.3's `.prop` form is exactly
                # the NaN-propagating combinator, one element pair at a time
                acc = get(0)
                for r in range(1, tile.rows):
                    acc = self.b.vbin(kind[:3], acc, get(r), dtype=elem, fmm='prop')
                self._v[rid] = acc
                return
            acc = get(0)
            for r in range(1, tile.rows):
                acc = self.b.vbin(kind, acc, get(r), dtype=elem,
                                  comment=f"down the tile: row {r}"
                                  if r == 1 else None)
            self._v[rid] = acc
            return

        self._tiles[rid] = Tile(
            'col',
            tuple(self._reduce_vec(kind, get(r), dtype=elem, mask=self._shape_mask(tile.cols),
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
        if not elem.startswith('f') and not self._force_vector:
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
        # A mask converted to a number is 0 or 1 -- -1 for a signed
        # conversion of i1 -- in each lane first: a predicate register has
        # no lanes for v.cvt to read (`(x != y).to(tl.float32)`).
        _, operand = _parse_type(str(op.get_operand(0).get_type()))
        true = -1 if src.startswith('s') else 1

        def lanes(v):
            if operand != 'i1':
                return v
            if v.kind == SCALAR:
                v = self.b.splat(v)
            if v.kind == VECTOR:
                v = self.b.vcmp('ne', v, self.b.vmovi(0))
            return self.b.vselect(v, self.b.vmovi(true), self.b.vmovi(0))
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: self.b.vcvt(src, dst, lanes(xs[0])))
            return
        _, elem, _bcast = self._float_shape(op)
        v = self._value(op.get_operand(0)) if operand == 'i1' else self._vec(op.get_operand(0), src)
        self._v[op.get_result(0).id()] = self.b.vcvt(src, dst, lanes(v))

    def _op_arith_truncf(self, op):
        self._float_convert(op)

    def _op_arith_extf(self, op):
        self._float_convert(op)

    def _op_tt_fp_to_fp(self, op):
        rounding = op.get_str_attr('rounding')
        if rounding not in (None, '', 'rtne'):
            raise Unsupported(f'float conversion rounding {rounding}')
        self._float_convert(op)

    def _float_convert(self, op):
        """Convert each row now, preserving every requested rounding step.

        Narrow values occupy the low bits of wide lanes in spill storage;
        widening must convert those bits before floating-point arithmetic.
        """
        src, res = op.get_operand(0), op.get_result(0)
        tile = self._tiles.get(src.id())
        _, rtype = _parse_type(str(res.get_type()))
        if tile is not None:
            if rtype == tile.store:
                self._tiles[res.id()] = tile
            else:
                get = self._tile_row_getter(tile)
                self._tiles[res.id()] = self._materialise(
                    lambda r: self.b.vcvt(tile.store, rtype, get(r)),
                    tile.rows, tile.cols, rtype)
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
        shape, elem = _parse_type(str(op.get_result(0).get_type()))
        if _pointee(elem):
            left, right = self._address(a), self._address(b)
            if shape is None:
                av, bv = left[0], right[0]
                out = self._copy_value(av)
                pred = self.b.vextract(self.b.vselect(self._value(cond),
                    self.b.vmovi(1), self.b.vmovi(0)), 0)
                end = self._control_label('ptrselect')
                self.b.branch_nz(pred, end)
                self.b._emit('assign', [], [out, bv])
                self.b.label(end)
                name = f'pointer{self._control_seq}'
                self._args[-len(self._args)-1] = (name, out)
                self._ptr_base[op.get_result(0).id()] = (name, _pointee(elem))
                self._v[op.get_result(0).id()] = out
                return
            raise Unsupported('a per-lane select between pointers')
        if self._is_2d(op):
            # A select of masks combines predicates. v.select moves vector
            # registers, and handed two predicate registers as its data it
            # read the vector registers that share their numbers --
            # Inductor's `acc = tl.where(m, acc | x, acc)` in every `any`.
            self._tile_binary(op, lambda xs, e: (
                self.b.pbin('or', self.b.pbin('and', xs[0], xs[1]),
                            self.b.pbin('and', self.b.pun('not', xs[0]), xs[2]))
                if e == 'i1' else self.b.vselect(xs[0], xs[1], xs[2])))
            return
        shape, elem, _bcast = self._float_shape(op)
        if elem == 'i1':
            mask, av, bv = self._value(cond), self._value(a), self._value(b)
            self._v[op.get_result(0).id()] = self.b.pbin('or',
                self.b.pbin('and', mask, av),
                self.b.pbin('and', self.b.pun('not', mask), bv))
            return
        if shape is None:
            av, bv = self._value(a), self._value(b)
            if av.kind == VECTOR:
                av = self.b.vextract(av, 0)
            if bv.kind == VECTOR:
                bv = self.b.vextract(bv, 0)
            value = self._copy_value(av)
            mask = self._value(cond)
            choose_a = self.b.vextract(self.b.vselect(
                mask, self.b.vmovi(1), self.b.vmovi(0)), 0)
            end = self._control_label('select')
            self.b.branch_nz(choose_a, end)
            self.b._emit('assign', [], [value, bv])
            self.b.label(end)
            self._v[op.get_result(0).id()] = value
            return
        self._v[op.get_result(0).id()] = self.b.vselect(
            self._value(cond), self._vec(a, elem), self._vec(b, elem))

    def _binary(self, op, kind, fmm=None):
        a, b = op.get_operand(0), op.get_operand(1)
        rid = op.get_result(0).id()
        if self._is_2d(op):
            self._tile_binary(op, lambda xs, e: (self.b.pbin(kind, xs[0], xs[1])
                if e == 'i1' else self.b.vbin(
                    kind, xs[0], xs[1], dtype=e, fmm=fmm)))
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
        if elem == 'i1' and not self._force_vector:
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
            name, elem, aff = spec
            if (op.get_num_operands() > 1 or not isinstance(aff, Affine2D)
                    or aff.col_stride != 1 or shape2d[1] != self.target.vlen):
                base = self._arg_value(name)
                get_index = self._offset_row(aff)
                get_mask = (self._row_getter(op.get_operand(1), shape2d[0], 'i1')
                            if op.get_num_operands() > 1 else lambda r: None)
                other = (self._const.get(op.get_operand(2).id(), 0)
                         if op.get_num_operands() > 2 else 0)
                size = _load_suffix(_SIZE_SUFFIX[self._ELEM_BYTES[elem]], elem)

                def row(r):
                    v = self.b.gather(base, get_index(r),
                                      mask=self._shape_mask(shape2d[1], get_mask(r)),
                                      other=other, dtype=elem, size=size)
                    if shape2d == (1, 1):
                        # One element: it lands in lane 0 only, but a
                        # `[1, 1]` that tt.broadcast stretches along the row
                        # is used by naming its register, so every lane must
                        # hold it. Inductor's one-row reductions load a flag
                        # or an id this way once per program; without the
                        # splat `tl.where(flag, x, 0)` kept one element of
                        # sixteen. (`[R, 1]` loads with R > 1 are left as
                        # they were: R3.7 consumes them correctly as they
                        # are, and changing them changed its artifacts.)
                        v = self.b.splat(self.b.vextract(v, 0, dtype=elem), dtype=elem)
                    return v
                self._tiles[op.get_result(0).id()] = self._materialise(
                    row, shape2d[0], shape2d[1], elem)
                return
            if elem not in self._TRANSFERABLE:
                self._tiles[op.get_result(0).id()] = self._load_tile_rows(
                    name, elem, aff, shape2d)
                return
            if aff.row_stride == 0:
                # Every row is the same row: Inductor's GEMM template with
                # M = 1 (a decode step's projections) drops the row index,
                # so `A` is one row of K broadcast down the tile. A
                # TensorMap cannot step rows by zero; one vector load can
                # be every row. A tl.dot reaches it through the same
                # repack as any computed tile (_as_fragment).
                esize = self._ELEM_BYTES[elem]
                at = self._arg_value(name)
                if aff.const:
                    at = self.b.sbini('add', at, aff.const * esize,
                                      comment="the row's origin")
                at = self._add_dynamic(at, aff, esize)
                row = self.b.load(at, dtype=elem,
                                  size=_load_suffix(_SIZE_SUFFIX[esize], elem))
                self._tiles[op.get_result(0).id()] = Tile(
                    'lazy', lambda r, row=row: row, shape2d[0], shape2d[1], elem)
                return
            desc = self._tile_descriptor(name, elem, aff, shape2d,
                                         'a 2-D tt.load')
            zero = self.b.const(0)
            self._tiles[op.get_result(0).id()] = self._dma_in(
                desc, [zero, zero], list(shape2d), elem, 'a 2-D tt.load')
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
                # Gather the same address in active lanes only. This also
                # suppresses an invalid target pointer when target_ok is false.
                value = self.b.gather(addr, self.b.vmovi(0), mask=mask,
                                      other=other, dtype=elem,
                                      size=_load_suffix(size, elem))
                self._v[rid] = self.b.vextract(value, 0, dtype=elem)
                return
            self._v[rid] = self.b.sload(
                addr, size=_load_suffix(size, elem), dtype=elem,
                comment='one element, not a vector')
            return
        mask = self._shape_mask(shape, mask)
        if kind == 'broadcast':
            # One element into every lane: a scalar load and a splat.
            if mask is not None:
                self._v[rid] = self.b.gather(addr, self.b.vmovi(0), mask=mask,
                    other=other, dtype=elem, size=_load_suffix(size, elem))
                return
            self._v[rid] = self.b.splat(
                self.b.sload(addr, size=_load_suffix(size, elem), dtype=elem,
                             comment='one element, broadcast to every lane'),
                dtype=elem)
        elif kind == 'contiguous':
            self._v[rid] = self.b.load(addr, mask=mask, other=other,
                                       dtype=elem,
                                       size=_load_suffix(size, elem))
        else:
            self.target.require('gather_scatter', 'emit v.ld.gather')
            self._v[rid] = self.b.gather(addr, index, mask=mask, other=other,
                                         dtype=elem,
                                         size=_load_suffix(size, elem))

    def _op_tt_store(self, op):
        ptr, value = op.get_operand(0), op.get_operand(1)
        shape2d, _ = _parse_type(str(value.get_type()))
        if isinstance(shape2d, tuple):
            spec = self._ptr2d.get(ptr.id())
            if spec is None:
                raise Unsupported(
                    'a 2-D store through a pointer this does not recognise',
                    source='doc/isa/TNCA-ISA-v0.1.md sec 9E.2')
            name, elem, aff = spec
            tile = self._tile_of(value)
            if (op.get_num_operands() > 2 or not isinstance(aff, Affine2D)
                    or aff.col_stride != 1 or shape2d[1] != self.target.vlen):
                base = self._arg_value(name)
                get_index, get = self._offset_row(aff), self._tile_row_getter(tile)
                get_mask = (self._row_getter(op.get_operand(2), shape2d[0], 'i1')
                            if op.get_num_operands() > 2 else lambda r: None)
                for r in range(shape2d[0]):
                    v = get(r)
                    if elem != tile.store:
                        v = self.b.vcvt(tile.store, elem, v)
                    self.b.scatter(base, get_index(r), v,
                        mask=self._shape_mask(shape2d[1], get_mask(r)),
                        size=_SIZE_SUFFIX[self._ELEM_BYTES[elem]])
                return
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
            v = self._value(value)
            if v.kind == VECTOR:
                # A float scalar lives broadcast in a vector register,
                # because there is no scalar floating point -- so the
                # ONE element it stands for is written by a vector store
                # masked to lane 0. `narrow_lane0_store` turns that back
                # into an `s.st` once it has proved the mask selects
                # exactly lane zero.
                self.b.store(addr, v, size=size, mask=self._lane0(),
                             comment='one element: lane 0 of a broadcast')
                return
            self.b.sstore(addr, v, size=size)
            return
        mask = self._shape_mask(shape, mask)
        if kind == 'broadcast' and shape == 1:
            self.b.store(addr, self._value(value), mask=mask, size=size)
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
