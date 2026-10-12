"""Host tensor descriptors for AOT kernels; no compiler or Triton dependency."""
from dataclasses import dataclass
import struct
from .errors import ArgumentError


@dataclass(frozen=True)
class TensorDescriptor:
    """A buffer view passed to a ``tensordesc<dtype[block,...]>`` parameter.

    Shape and strides are in elements, dimension 0 outermost. ``buffer`` is a
    Session buffer or ``(buffer, byte_offset)``. The runtime snapshots the
    TensorMap per submit and leases the underlying buffer until the launch
    completes.

    Every descriptor that is legal for a TMA on NVIDIA GPUs (Triton's
    ``triton.tools.tensor_descriptor.TensorDescriptor``) is legal here and
    means the same thing: rank 1-5, ``padding`` of ``"zero"`` or ``"nan"``
    (floating-point only) for out-of-bounds elements of a load. NVIDIA's
    alignment rules (16-byte base and strides, 16-byte block rows) are not
    imposed: they are TMA hardware limits, and the compiler reads a host
    descriptor through the LSU, which has none.
    """
    buffer: object
    shape: tuple
    strides: tuple
    block_shape: tuple
    dtype: str
    padding: str = 'zero'

    # ISA sec 9D.3 dtype codes. Unsigned integers share their signed width's
    # code: the transfer engine moves bits, and the code only decides what
    # NaN padding means.
    FORMATS = {'fp16': (2, 0), 'bf16': (2, 1), 'fp32': (4, 2), 'i8': (1, 3), 'u8': (1, 3),
               'i16': (2, 4), 'u16': (2, 4), 'i32': (4, 5), 'u32': (4, 5),
               'i64': (8, 6), 'u64': (8, 6), 'e4m3': (1, 7), 'e5m2': (1, 8), 'fp64': (8, 9)}
    FLOATS = ('fp16', 'bf16', 'fp32', 'fp64', 'e4m3', 'e5m2')
    # ISA sec 9D.3 padding modes: 0 zero-fill, 3 quiet-NaN fill.
    PADDING = {'zero': 0, 'nan': 3}

    def __post_init__(self):
        for name in ('shape', 'strides', 'block_shape'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        rank = len(self.shape)
        if not 1 <= rank <= 5 or len(self.strides) != rank or len(self.block_shape) != rank:
            raise ArgumentError('tensor descriptor requires matching ranks in 1..5')
        if any(type(x) is not int or x <= 0 for x in self.shape + self.strides + self.block_shape):
            raise ArgumentError('tensor descriptor dimensions and strides must be positive integers')
        if any(x > 0x7fffffff for x in self.shape + self.block_shape):
            raise ArgumentError('tensor descriptor dimensions exceed signed i32')
        if self.dtype not in self.FORMATS:
            raise ArgumentError(f'unsupported tensor descriptor dtype {self.dtype!r}')
        if self.padding not in self.PADDING:
            raise ArgumentError(f'tensor descriptor padding must be "zero" or "nan", '
                                f'not {self.padding!r}')
        if self.padding == 'nan' and self.dtype not in self.FLOATS:
            raise ArgumentError('padding "nan" is only defined for floating-point tensors')

    def pack(self, address, available):
        size, code = self.FORMATS[self.dtype]
        needed = (1 + sum((d-1)*s for d, s in zip(self.shape, self.strides))) * size
        if needed > available:
            raise ArgumentError(f'tensor descriptor spans {needed} bytes, buffer has {available}')
        if any(s*size >= 1 << 63 for s in self.strides):
            raise ArgumentError('tensor descriptor byte stride exceeds signed i64')
        data = bytearray(256)
        struct.pack_into('<Q', data, 0, address)
        for i, (extent, stride, block) in enumerate(zip(self.shape, self.strides, self.block_shape)):
            struct.pack_into('<Q', data, 8+8*i, extent)
            struct.pack_into('<q', data, 48+8*i, stride*size)
            struct.pack_into('<I', data, 88+4*i, block)
        struct.pack_into('4B', data, 108, len(self.shape), code, self.PADDING[self.padding], 0)
        return bytes(data)
