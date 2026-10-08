"""Two-dimensional blocks: how a 16x16 tile is described and where it lives.

A vector register is one dimension. A `tensor<16x16xf16>` is not, and the
way this machine holds one is not "sixteen vector registers" -- it is an
**RF window** in the transfer engine's fragment layout (ISA sec 9C.2),
which is exactly what `t.ld.async` writes and exactly what
`m.mma.async` reads. Nothing repacks in between, which is the whole
argument for the design (sec 7.3).

That has a consequence worth stating plainly, because it decides the
shape of the lowering rather than being an optimisation: **a 2-D f16
tile cannot be loaded with ordinary vector loads.** A 16x16 FP16
fragment is 512 bytes in eight registers, two rows per register, and
`v.ld.h` puts one 2-byte element in each of sixteen 32-bit lanes. The
packed form needs `v.pack.f16`, which ISA sec 8.4 marks *Reserved*. So
the transfer engine is not the fast path for a tile -- it is the only
path, and a 2-D `tt.load` has to become a descriptor and a DMA.

Which is also what makes `tl.make_tensor_descriptor` cheap to support:
it says directly what the pointer arithmetic below has to be
reverse-engineered into.

## Recognising a tile from pointer arithmetic

Triton builds `a_ptr + om[:, None]*K + ok[None, :]` out of
`tt.expand_dims`, `arith.muli`, `tt.broadcast` and `tt.addptr`. Rather
than lower each of those -- none of which has a meaning on its own here
-- the importer tracks what the index tensor *is*:

    index(row, col) = row_stride * row + col_stride * col + constant

`Affine2D` below is that triple. A tile access is recognised when
`col_stride` is 1, which is not an arbitrary restriction: ISA sec 9E.2
refuses a descriptor whose `byte_stride[1]` is not `sizeof(dtype)`,
because the engine copies runs rather than gathering per element.
"""
import struct

# ISA sec 9D.3. The word indices of the 256-byte TensorMap, as 32
# little-endian 64-bit words. Named here once; the test programs each
# carried their own copy of this table.
TENSORMAP_WORDS = 32
TENSORMAP_BYTES = TENSORMAP_WORDS * 8

# sec 9D.3 `dtype`. Anything else is reserved, so an unlisted element
# type is refused rather than encoded as the nearest thing. The engine moves
# bits; the code decides only what NaN padding means.
DTYPE_CODE = {'f16': 0, 'bf16': 1, 'f32': 2, 'i8': 3, 'i16': 4, 'i32': 5, 'i64': 6,
              'f8E4M3FN': 7, 'f8E5M2': 8, 'f64': 9}

# sec 9D.3 `padding mode`: 0 zero-fill, 1 reserved (was clamp), 2 fault on
# any out-of-bounds element, 3 quiet-NaN fill (floating dtypes).
PAD_ZERO, PAD_CLAMP, PAD_FAULT, PAD_NAN = 0, 1, 2, 3


class TileError(Exception):
    """A 2-D access this machine cannot describe."""


def tensormap_words(*, base, extents, byte_strides, block_shape, dtype,
                    pad=PAD_ZERO):
    """The 256-byte TensorMap of ISA sec 9D.3, as 32 64-bit words.

    Only the first 128 bytes are produced. The second 128 are hints, and
    sec 9D.3 is explicit that nothing there may change an address, a
    value, an access or an order -- so a compiler that has nothing to
    hint leaves them zero rather than inventing something.
    """
    rank = len(extents)
    if rank != len(byte_strides) or rank != len(block_shape):
        raise TileError('a descriptor whose rank does not agree across its '
                        'extents, strides and block shape')
    if dtype not in DTYPE_CODE:
        raise TileError(
            f"a descriptor of {dtype}: ISA sec 9D.3 allocates dtype codes "
            f"for {', '.join(sorted(DTYPE_CODE))} and reserves the rest")

    w = [0] * TENSORMAP_WORDS
    w[0] = base                                  # bytes 0-7
    for i, e in enumerate(extents):              # bytes 8-47
        w[1 + i] = e
    for i, sbytes in enumerate(byte_strides):    # bytes 48-87
        w[6 + i] = sbytes & ((1 << 64) - 1)      # signed, stored raw
    # bytes 88-107: block_shape[5], 32-bit each. Two per word.
    for i, bs in enumerate(block_shape):
        word, half = divmod(i, 2)
        w[11 + word] |= (bs & 0xFFFFFFFF) << (32 * half)
    # bytes 108-111: rank, dtype, padding mode, semantic flags -- one
    # byte each, in word 13's low half.
    w[13] |= (rank & 0xFF) << 32
    w[13] |= (DTYPE_CODE[dtype] & 0xFF) << 40
    w[13] |= (pad & 0xFF) << 48
    return w


def check_geometry(*, byte_strides, block_shape, elem_bytes, base=None):
    """Refuse the geometries ISA sec 9E.2 refuses, and say which.

    The engine places whole 8-byte units into window registers through
    the existing lane write enables. Every condition here is that one
    fact stated at a different end of the transfer, and an engine that
    proceeded anyway would return data shifted by a couple of bytes per
    row -- a wrong answer wearing the shape of a right one.
    """
    if not 1 <= len(block_shape) <= 5:
        raise TileError(f"a rank-{len(block_shape)} tile: ISA sec 9E executes rank 1-5")
    if any(not 1 <= b <= 256 for b in block_shape):
        raise TileError(f"a block {list(block_shape)}: ISA sec 9E.2 takes each block "
                        f"dimension in 1..256")
    if byte_strides[-1] != elem_bytes:
        raise TileError(
            f"a tile whose innermost stride is {byte_strides[-1]} bytes and "
            f"not one element ({elem_bytes}): ISA sec 9E.2 -- the transfer "
            f"engine copies runs, it does not gather per element")
    if (block_shape[-1] * elem_bytes) % 8:
        raise TileError(
            f"a tile {block_shape[-1]} elements wide: ISA sec 9E.2 -- "
            f"{block_shape[-1] * elem_bytes} bytes a row is not a whole "
            f"number of 8-byte units, so one would straddle two window "
            f"registers")
    for s in byte_strides[:-1]:
        if isinstance(s, int) and s % 8:
            raise TileError(
                f"a stride of {s} bytes: ISA sec 9E.2 -- rows after the first "
                f"would start off an 8-byte boundary")
    if base is not None and base % 8:
        raise TileError(
            f"a tile based at {base:#x}: ISA sec 9E.2 wants 8-byte "
            f"alignment, or the first unit of every row lands shifted")


class Affine2D:
    """`row_stride*row + col_stride*col + constant`, in ELEMENTS.

    `constant` is either an int or a scalar Value; the two are kept
    apart because a compile-time offset folds into the descriptor's
    base and a runtime one has to be added with an instruction.
    """

    __slots__ = ('row_stride', 'col_stride', 'const', 'dynamic')

    def __init__(self, row_stride=0, col_stride=0, const=0, dynamic=None):
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.const = const
        # Scalar Values added to the offset whose size is not known here.
        # (scalar Value, coefficient in elements). A coefficient is
        # carried rather than applied so that scaling the whole index
        # -- which a split-k matmul does -- stays exact and stays one
        # multiply at the end.
        self.dynamic = [(v, 1) if not isinstance(v, tuple) else v
                        for v in (dynamic or ())]

    def scaled(self, k):
        """Multiply the whole index by a constant.

        A runtime term keeps its own coefficient rather than being
        refused: `s * K` with `s` a program id is how every split-k
        matmul finds its slice, and the scale is one `s.muli` at the
        point the base address is computed.
        """
        return Affine2D(self.row_stride * k, self.col_stride * k,
                        self.const * k,
                        [(v, c * k) for v, c in self.dynamic])

    def plus(self, other):
        return Affine2D(self.row_stride + other.row_stride,
                        self.col_stride + other.col_stride,
                        self.const + other.const,
                        self.dynamic + other.dynamic)

    def __repr__(self):
        return (f"Affine2D(row*{self.row_stride} + col*{self.col_stride} + "
                f"{self.const}{'+dyn' if self.dynamic else ''})")


class Tile:
    """A 2-D value, and where on the machine it is.

    `kind` is 'window' for a transfer-engine fragment -- the form
    `m.mma.async` reads -- or 'accumulator' for a 16x16 binary32 result
    that the vector unit reaches a row at a time (ISA sec 9B).
    """

    __slots__ = ('kind', 'handle', 'rows', 'cols', 'dtype', 'desc',
                 'store')

    def __init__(self, kind, handle, rows, cols, dtype, desc=None,
                 store=None):
        self.kind = kind
        self.handle = handle
        self.rows = rows
        self.cols = cols
        # What the tile MEANS, which is the IR's type for it.
        self.dtype = dtype
        self.desc = desc
        # What the registers actually hold, which is not always the
        # same. An accumulator is binary32 whatever the MMA's inputs
        # were (ISA sec 9C), so `acc.to(tl.float16)` records f16 as the
        # meaning and leaves f32 in the registers -- the conversion
        # happens once, on the way out, where sec 8.4's reserved pack is
        # not needed. Reading `dtype` there instead emitted a
        # convert-f16-to-f16 and stored the f32 bits.
        self.store = store or dtype

    def __repr__(self):
        held = '' if self.store == self.dtype else f" held as {self.store}"
        return (f"Tile({self.kind}, {self.rows}x{self.cols}x{self.dtype}"
                f"{held})")


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]
