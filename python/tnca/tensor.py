"""Host tensor descriptors for AOT kernels; no compiler or Triton dependency."""
from dataclasses import dataclass
import struct
from .errors import ArgumentError


@dataclass(frozen=True)
class TensorDescriptor:
    """A buffer view passed to a ``tensordesc<dtype[block,...]>`` parameter.

    Shape and strides are in elements. ``buffer`` is a Session buffer or
    ``(buffer, byte_offset)``. The runtime snapshots the TensorMap per submit
    and leases the underlying buffer until the launch completes.
    """
    buffer: object
    shape: tuple
    strides: tuple
    block_shape: tuple
    dtype: str

    def __post_init__(self):
        for name in ('shape', 'strides', 'block_shape'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        rank = len(self.shape)
        if not 2 <= rank <= 5 or len(self.strides) != rank or len(self.block_shape) != rank:
            raise ArgumentError('tensor descriptor requires matching ranks in 2..5')
        if any(type(x) is not int or x <= 0 for x in self.shape + self.strides + self.block_shape):
            raise ArgumentError('tensor descriptor dimensions and strides must be positive integers')
        if any(x > 0x7fffffff for x in self.shape + self.block_shape):
            raise ArgumentError('tensor descriptor dimensions exceed signed i32')
        if self.dtype not in self.FORMATS:
            raise ArgumentError(f'unsupported tensor descriptor dtype {self.dtype!r}')

    # Software descriptor accesses retain the Triton arithmetic type. Formats
    # without a hardware transfer code use the same-width storage code.
    FORMATS = {'fp16': (2, 0), 'bf16': (2, 1), 'fp32': (4, 2),
               'i32': (4, 2), 'i8': (1, 3), 'e4m3': (1, 3), 'e5m2': (1, 3)}

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
        struct.pack_into('4B', data, 108, len(self.shape), code, 0, 0)
        return bytes(data)
