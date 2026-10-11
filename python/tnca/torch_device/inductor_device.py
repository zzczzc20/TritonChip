"""Inductor and Dynamo for the `tnca` device (R40-01).

Each piece through the upstream extension point MTIA uses for its own
Triton-programmed accelerator (torch/_dynamo/device_interface.py
MtiaInterface, torch/_inductor/codegen/python_wrapper_mtia.py,
codegen/mtia/device_op_overrides.py):

  DeviceInterface    `register_interface_for_device("tnca", ...)`: what
                     Dynamo and Inductor ask a device -- count, current,
                     streams, properties. The properties are the PROFILE's
                     (its target file: cores, Role Contexts, vector length,
                     dtype capabilities), not a placeholder of unlimited
                     resources (doc R40-01).
  Scheduling and     `register_backend_for_device("tnca", TritonScheduling,
  wrapper            PythonWrapperTnca)`: Inductor's Triton code generation
                     and its Python wrapper, as for CUDA and MTIA.
  op overrides       what the wrapper writes to get a stream, set the device,
                     synchronise.
  policy             `tnca` is NOT one of Inductor's GPU_TYPES: where
                     lowering asks `is_gpu()` it takes the CPU path's (B39's)
                     answer, and tnca.torch_device.policy gives it the CPU
                     path's answer where lowering asks `device.type == "cpu"`
                     -- measured, doc/r40_tnca_device.md sec 3.

Kernels compile for Triton target GPUTarget("tnca", 0, 1), which TNCA's Triton
backend serves (tnca.triton_backend); the launcher passes a tnca tensor's own
device buffer (tnca.triton_backend.driver, R40-01 resident path).
"""
import contextlib
import os
from types import SimpleNamespace

NAME = 'tnca'
_registered = False


def properties(index=0):
    """The device's properties, from the profile's target file."""
    from tnca import device as _device
    from tnca.adapters import r1
    if index not in (None, 0):
        raise RuntimeError(f'tnca:{index} does not exist')
    profile = os.environ.get('TNCA_PROFILE', 'core-sim')
    target, info = r1.target_for(_device.base_profile(profile))
    caps = target.capabilities if isinstance(target.capabilities, dict) else dict(target.capabilities)
    num_cores = int(getattr(target, 'num_cores', 1) or 1)
    num_rctx = int(getattr(target, 'num_rctx', 1) or 1)
    vlen = int(getattr(target, 'vlen', 16) or 16)
    from . import MEMORY
    heap = MEMORY.session(0)._heap
    return SimpleNamespace(
        name=f'TNCA {profile}', profile=profile, type=NAME, index=index,
        # Inductor reads these: the cores a grid spreads over, a "warp" is
        # one program (TNCA has no SIMT threads; num_warps is ignored), and
        # a program's lanes are one vector register.
        multi_processor_count=num_cores, warp_size=1, major=0, minor=0,
        max_threads_per_block=vlen, max_threads_per_multi_processor=vlen * num_rctx,
        regs_per_multiprocessor=int(target.phys_vregs),
        num_rctx=num_rctx, vlen=vlen, capabilities=caps,
        total_memory=heap.limit - heap.base,
    )


class _device:
    """`torch.tnca.device(i)`: a context manager; one device in R40-01."""

    def __init__(self, device):
        import torch
        self.idx = device if isinstance(device, int) else (torch.device(device).index or 0)

    def __enter__(self):
        if self.idx not in (0, -1, None):
            raise RuntimeError(f'tnca:{self.idx} does not exist')
        return self

    def __exit__(self, *exc):
        return False


def get_raw_stream(device_idx=0):
    """Return the current logical stream identity for the native wrapper."""
    import torch
    return torch.tnca.current_stream().stream_id


def _interface():
    import torch
    from torch._dynamo.device_interface import DeviceInterface
    mod = torch.tnca

    class TncaInterface(DeviceInterface):
        device = _device
        Event = mod.Event
        Stream = mod.Stream

        class Worker:
            @staticmethod
            def set_device(device):
                pass

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_properties(device=None):
                return properties(0)

        current_device = staticmethod(mod.current_device)
        set_device = staticmethod(mod.set_device)
        device_count = staticmethod(mod.device_count)
        synchronize = staticmethod(mod.synchronize)
        current_stream = staticmethod(mod.current_stream)
        get_raw_stream = staticmethod(get_raw_stream)
        memory_allocated = staticmethod(mod.memory_allocated)

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def maybe_exchange_device(device):
            return 0

        @staticmethod
        def exchange_device(device):
            return 0

        @staticmethod
        def stream(stream):
            return mod.stream(stream)

        @staticmethod
        def set_stream(stream):
            mod.set_stream(stream)

        @staticmethod
        def _set_stream_by_id(stream_id, device_index, device_type):
            mod._set_stream_by_id(stream_id, device_index, device_type)

        @staticmethod
        def get_device_properties(device=None):
            return properties(0)

        @staticmethod
        def get_compute_capability(device=None):
            return 0

        @staticmethod
        def is_bf16_supported(including_emulation=False):
            return bool(properties(0).capabilities.get('matrix_bf16'))

        @staticmethod
        def is_triton_capable(device=None):
            return True

        @staticmethod
        def raise_if_triton_unavailable(device=None):
            import triton.backends
            from torch._dynamo.exc import TritonUnavailableError
            from tnca.triton_backend.compiler import TNCABackend
            if not any(b.compiler is TNCABackend for b in triton.backends.backends.values()):
                raise TritonUnavailableError("Triton has no TNCA backend (the tnca package's entry point)")

    return TncaInterface


def register_inductor():
    """Register the tnca device with Dynamo and Inductor (idempotent)."""
    global _registered
    if _registered:
        return
    import torch
    from torch._dynamo.device_interface import register_interface_for_device
    from torch._inductor.codegen.common import (DeviceOpOverrides, register_backend_for_device,
                                                register_device_op_overrides)
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    torch.tnca.device = _device
    register_interface_for_device(NAME, _interface())

    class TncaDeviceOpOverrides(DeviceOpOverrides):
        def import_get_raw_stream_as(self, name):
            return f'from tnca.torch_device.inductor_device import get_raw_stream as {name}'

        def set_device(self, device_idx):
            return f'torch.tnca.set_device({device_idx})'

        def synchronize(self):
            return 'torch.tnca.synchronize()'

        def device_guard(self, device_idx):
            return f'torch.tnca.device({device_idx})'

        def current_stream(self):
            return 'torch.tnca.current_stream()'

    register_device_op_overrides(NAME, TncaDeviceOpOverrides())

    class PythonWrapperTnca(PythonWrapperCodegen):
        """The upstream Python wrapper; its header makes sure the device is
        registered in whatever process runs the compiled module."""

        def write_header(self):
            super().write_header()
            self.imports.splice('import tnca.torch_device\ntnca.torch_device.register()')

        @staticmethod
        def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
            if is_subgraph:
                return PythonWrapperCodegen.create(is_subgraph, subgraph_name, parent_wrapper,
                                                   partition_signatures)
            return PythonWrapperTnca()

    register_backend_for_device(NAME, TritonScheduling, PythonWrapperTnca)
    # Triton's registry gets an entry under the device's name: Inductor sets
    # up a Triton device's driver by looking for an active backend not named
    # `cpu` (triton_helpers.set_driver_to_gpu, at kernel load and at
    # precompile). The entry carries TNCA's driver -- the one already active
    # -- and a compiler that claims no target, because Triton refuses a
    # target two compilers support (compiler.make_backend) and the `cpu`
    # entry already compiles for `tnca`. It is added after Triton has created
    # its one active driver (triton.runtime.driver), which it does by
    # counting active drivers once.
    import triton.backends
    import triton.runtime
    from tnca.triton_backend.compiler import TNCABackend
    from tnca.triton_backend.driver import TNCADriver
    triton.runtime.driver.active.get_current_target()
    if NAME not in triton.backends.backends:
        class _NamesTheDevice(TNCABackend):
            """Registry entry `tnca`: the driver is TNCA's; compiling is the
            `cpu` entry's (TNCABackend supports target `tnca`)."""

            @staticmethod
            def supports_target(target):
                return False

        triton.backends.backends[NAME] = type(next(iter(triton.backends.backends.values())))(
            compiler=_NamesTheDevice, driver=TNCADriver)
    # A tnca kernel is compiled and gets its launcher with TNCA's driver
    # active, also in a process where CUDA is (R40-01 acceptance 2). Inductor
    # picks a driver per kernel by device type: `cpu` -> the `cpu` registry
    # entry (TNCA's backend), anything else -> the first active GPU backend,
    # which with CUDA present is NVIDIA's. For tnca the precompile step
    # selects the `cpu` entry's driver and holds it there with upstream's own
    # skip_gpu_driver_setup(), so the launcher it binds is TNCA's. CUDA and
    # CPU kernels are untouched.
    from torch._inductor.runtime import triton_heuristics as th
    from torch._inductor.runtime import triton_helpers
    up_pre = th.CachingAutotuner._precompile_config
    if not getattr(up_pre, '_tnca', False):
        def _precompile_config(self, cfg, *args, **kwargs):
            if getattr(self.device_props, 'type', None) != NAME:
                return up_pre(self, cfg, *args, **kwargs)
            triton_helpers.set_driver_to_cpu()
            with triton_helpers.skip_gpu_driver_setup():
                return up_pre(self, cfg, *args, **kwargs)
        _precompile_config._tnca = True
        th.CachingAutotuner._precompile_config = _precompile_config

    # tnca is NOT added to Inductor's GPU_TYPES: where lowering asks
    # `is_gpu()` it gets the non-GPU answer, which is the CPU path's (B39's);
    # the few decisions keyed on `device.type == "cpu"` are given the CPU
    # path's answer by tnca.torch_device.policy, each behind its own switch.
    from . import policy
    policy.install()
    _registered = True
