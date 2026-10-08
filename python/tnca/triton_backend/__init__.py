"""TNCA as a Triton backend (entry point `triton.backends`: cpu -- the backend
Inductor's `cpu_backend = "triton"` uses).

compiler.py compiles TTIR with the lowering `tnca.aot.compile_aot` uses;
driver.py launches through `tnca.runtime.Session`. See doc/r38_pytorch_p1plus.md.
"""


def _descriptor_class():
    from dataclasses import dataclass
    from triton.tools.tensor_descriptor import TensorDescriptor as _Triton

    @dataclass
    class TensorDescriptor(_Triton):
        """triton.tools.tensor_descriptor.TensorDescriptor without the CUDA TMA
        alignment asserts (16-byte base and strides).

        Those are an NVIDIA hardware requirement written into Triton's host
        class. TNCA's transfer engine states its own geometry (ISA sec 9E.2)
        and raises DESCRIPTOR_FAULT on a violation, and the compiler reads a
        descriptor it cannot hand the engine through the LSU. Rank, contiguity
        of the last dimension and the padding mode are still checked.
        """

        def __post_init__(self):
            rank = len(self.shape)
            assert len(self.strides) == rank and len(self.block_shape) == rank, f'rank mismatch: {self}'
            assert 0 < rank <= 5, 'rank must be 1..5'
            assert all(d > 0 for d in self.shape), 'shape must be positive'
            assert self.strides[-1] == 1, 'last dimension must be contiguous'
            assert self.padding in ('zero', 'nan'), 'illegal padding'

    return TensorDescriptor


def __getattr__(name):
    if name == 'TensorDescriptor':
        cls = _descriptor_class()
        globals()['TensorDescriptor'] = cls
        return cls
    raise AttributeError(name)
