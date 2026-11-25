"""
TNCA: an experimental AOT compiler and RTL-simulation runtime.

Importing this package does nothing expensive. It starts no simulator, runs no
test and contacts no network -- doc/phase2_tasks_and_acceptance.md R2-02 asks
for that directly, and the way it is kept true is that the submodules which
CAN do those things are imported by the code that needs them, not here.

    from tnca.aot import compile_aot, inspect_artifact
    from tnca.runtime import Session
"""

__version__ = '0.2.0'

# The board profile's pointwise lowering uses streamed register tiles by
# default. An explicit environment setting still selects another mode.
import os as _os
_os.environ.setdefault('TNCA_TILED_POINTWISE', '3')
from . import inductor as _inductor

def _pointwise_cap(size_hints, cap):
    wide = _inductor._POINTWISE_CAP[0]
    if not wide or not size_hints or len(size_hints) != 1:
        return cap
    numel = int(next(iter(size_hints.values())))
    # Tiny one-dimensional streams need at least two register tiles in one
    # program. Longer streams keep enough programs to fill all three cores.
    min_programs = 4 if numel <= 256 else (8 if numel <= 512 else 16)
    if numel >= 16384:
        import inspect
        caller = inspect.currentframe().f_back.f_locals
        args = caller.get('args', ())
        meta = args[-1] if args and isinstance(args[-1], dict) else caller.get('kwargs', {}).get('inductor_meta', {})
        # Inductor's tiling score reflects bytes moved by a pointwise lane.
        # Large packed 16-bit streams get another wave of programs; wider
        # data types retain the longer streams that amortise setup better.
        score = ((meta or {}).get('tiling_scores') or {}).get('x') or 0
        if 6 * numel <= score <= 10 * numel:
            min_programs = 32
    block = 1
    while block * 2 * min_programs <= numel and block * 2 <= wide:
        block *= 2
    return max(cap, block)

_inductor._pointwise_cap = _pointwise_cap

_base_register = _inductor._register
_mm_hooked = False

def _register(*args, **kwargs):
    global _mm_hooked
    if not _mm_hooked:
        import torch
        from torch._inductor.heuristics.template.triton import (
            MMTemplateConfigMixin, GemmConfig)
        base = MMTemplateConfigMixin._get_template_configs_impl

        def mm_configs(self, kernel_inputs, op_name, **options):
            if (type(self).__module__ == 'tnca.inductor'
                    and 'TNCA_GEMM_TILE' not in _os.environ):
                dtype = kernel_inputs.mat1mat2()[0].get_dtype()
                m, n, _ = kernel_inputs.mnk_hinted()
                # E4M3's four products per lane amortise a wider output
                # tile. Partial N tiles waste too many lanes; small N also
                # needs full M fragments. Other FP8 formats keep N=16.
                fp8_wide = (op_name == 'scaled_mm' and dtype == torch.float8_e4m3fn and n % 32 == 0 and
                            (n >= 128 or (m >= 16 and m % 16 == 0 and n >= 32)))
                # A small batch uses only part of each 16-row MMA fragment.
                # Two output fragments per program amortise its setup while
                # preserving the accumulation order within every output.
                small_batch_wide = (op_name in ('mm', 'addmm') and
                                    dtype in (torch.float16, torch.bfloat16) and
                                    2 <= m < 16 and n >= 32 and n % 16 == 0)
                if fp8_wide or small_batch_wide:
                    original = self.mm_configs
                    self.mm_configs = [GemmConfig(16, 32, 16, 1, 1, group_m=1)]
                    try:
                        yield from base(self, kernel_inputs, op_name, **options)
                    finally:
                        self.mm_configs = original
                    return
            yield from base(self, kernel_inputs, op_name, **options)

        MMTemplateConfigMixin._get_template_configs_impl = mm_configs
        _mm_hooked = True
    return _base_register(*args, **kwargs)

_inductor._register = _register

# Names, not modules: `from tnca import Session` should work without making
# `import tnca` pull in the runtime, the compiler and Triton.
__all__ = ['__version__', 'aot', 'runtime', 'abi', 'manifest', 'artifact',
           'cache', 'errors', 'transport']
