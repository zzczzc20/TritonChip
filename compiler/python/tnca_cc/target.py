"""
The target contract.

compiler_overview.md sec 15 C0 fixes what a TNCA target must describe before
any lowering is written: RCTX count, VLEN, physical and logical V register
limits, accumulator count, DMA rank/dtype/layout capabilities, MMA throughput
and command constraints, barrier count and count width, source-release
capability, and SFU/atomics/fence capabilities.

This module reads that description; it does not contain it. The numbers live in
compiler/targets/*.json, generated from the RTL parameters by
tools/isa/gen_target.py, and tb/core/progs/t15_target.s asserts the hardware
agrees with them. Hard-coding a second copy here is the same mistake sec 11.3
forbids for opcodes.
"""
import json
import os

from .diagnostics import CapabilityError, ResourceError

from ._datapath import ROOT, find

TARGET_DIR = find('compiler', 'targets')

# Why the compiler asks, so a refusal can say what it was about to do.
_CAPABILITY_USE = {
    'sfu': 'emit v.*.approx for a transcendental',
    'tensor_transpose': "ask the transfer engine for a transposed write",
    'tensor_transpose_8bit': "ask the transfer engine for a transposed write of 1-byte elements",
    'atomics_device': 'emit a device-scope atomic',
    'atomics_system': 'emit a system-scope atomic',
    'fence': 'emit a memory fence',
    'packed_f16': 'emit packed FP16 arithmetic',
    'gather_scatter': 'emit v.ld.gather / v.st.scatter',
    'matrix_f16': 'emit m.mma.async with FP16 inputs',
    'matrix_bf16': 'emit m.mma.async with BF16 inputs',
    'matrix_fp8': 'emit m.mma.async with OCP FP8 (E5M2 or E4M3) inputs',
    'matrix_dp': 'emit m.mma.dp.async (grouped dot-product accumulation, ISA sec 9C.4)',
    'tensor_dma': 'emit t.ld.async / t.st.async',
    'tensor_store': 'emit t.st.async (window -> global)',
    'mbarrier': 'emit mbarrier operations',
    'vector_fp32': 'emit FP32 vector arithmetic',
    'cache_hints': 'attach a cache hint to a memory access',
}

# The fallback each missing capability has, if any. sec 12 requires the choice
# to be explicit: software math for the SFU, and nothing at all for atomics.
_CAPABILITY_FALLBACK = {
    'sfu': ('lower to software math, and re-verify the error bound against '
            'the ISA sec 8.6 contract rather than assuming an upstream '
            'polynomial meets it'),
    'cache_hints': 'drop the hint; it changes no address, value or order',
    'packed_f16': 'widen to FP32 and narrow on store',
    'tensor_transpose': ('read the source down its columns with '
                         'v.ld.gather, which is what the compiler does '
                         'for tl.trans'),
    'tensor_transpose_8bit': ('gather the 1-byte tile down its columns, as '
                              'every image before R41 X8 does'),
    'atomics_device': None,
    'atomics_system': None,
    'fence': None,
}


class TargetInfo:
    """One TNCA configuration, as the backend sees it."""

    def __init__(self, data):
        self._d = data
        missing = [k for k in ('num_rctx', 'vlen', 'registers', 'accumulators',
                               'barriers', 'dma', 'mma', 'capabilities')
                   if k not in data]
        if missing:
            raise ValueError(f"target description is missing {missing}")

    # ---- loading ----------------------------------------------------------
    @classmethod
    def load(cls, name_or_path):
        path = name_or_path
        if not os.path.exists(path):
            path = os.path.join(TARGET_DIR, f"{name_or_path}.json")
        if not os.path.exists(path):
            known = sorted(f[:-5] for f in os.listdir(TARGET_DIR)
                           if f.endswith('.json'))
            raise FileNotFoundError(
                f"no target {name_or_path!r}; known targets: {known}")
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def default(cls):
        return cls.load('tnca-sim-v0.1')

    # ---- identity ---------------------------------------------------------
    @property
    def name(self):
        return self._d['name']

    @property
    def isa_version(self):
        return self._d['isa_version']

    # ---- execution shape --------------------------------------------------
    @property
    def num_rctx(self):
        """Role Contexts per core: the ceiling on a Context Specialization plan."""
        return self._d['num_rctx']

    @property
    def vlen(self):
        return self._d['vlen']

    @property
    def num_cores(self):
        return self._d['num_cores']

    # ---- registers --------------------------------------------------------
    @property
    def vregs_per_rctx(self):
        """Logical V registers one RCTX may name."""
        return self._d['registers']['vreg_logical_per_rctx']

    @property
    def phys_vregs(self):
        """Physical V registers in the core.

        Deliberately separate from vregs_per_rctx * num_rctx. The slack is what
        allows one window to be visible to two Role Contexts, which sec 13.1
        raised and tb/core/progs/t13_share.s now demonstrates.
        """
        return self._d['registers']['vreg_physical_total']

    @property
    def sregs_per_rctx(self):
        return self._d['registers']['sreg_per_rctx']

    @property
    def pregs_per_rctx(self):
        return self._d['registers']['preg_per_rctx']

    # ---- whole-program resources (sec 10.1) -------------------------------
    @property
    def num_accumulators(self):
        return self._d['accumulators']['count']

    @property
    def num_barriers(self):
        return self._d['barriers']['count']

    @property
    def barrier_index_width(self):
        return self._d['barriers']['index_width']

    @property
    def num_windows(self):
        return self._d['windows']['count']

    @property
    def num_descriptors(self):
        return self._d['descriptors']['count']

    # ---- asynchronous engines ---------------------------------------------
    @property
    def dma(self):
        return dict(self._d['dma'])

    @property
    def dma_max_rank(self):
        return self._d['dma']['max_rank']

    @property
    def mma(self):
        return dict(self._d['mma'])

    @property
    def mma_shape(self):
        s = self._d['mma']['shape']
        return (s['m'], s['n'], s['k'])

    @property
    def mma_source_release(self):
        """When an m.mma.async stops reading its source registers.

        v0.1 has a single completion covering both source and destination, so
        a source register cannot be rewritten before the whole command retires.
        sec 10.2 wants a separate, earlier source-reusable event; until the
        hardware provides one, the scheduler must take the conservative bound
        rather than assume the loose one.
        """
        return self._d['mma']['source_release']

    @property
    def has_separate_source_completion(self):
        return bool(self._d['mma'].get('separate_source_completion'))

    # Triton descriptor padding -> TNCA padding mode. sec 12: "descriptor
    # `nan` padding -- 当前 TNCA zero/clamp/fault 不等价，必须显式补齐或
    # fallback." Zero is not NaN: a masked softmax that pads with zero scores
    # a real -0 where it meant "absent", so substituting is a wrong answer.
    _PADDING_MAP = {'zero': 'zero', 'clamp': 'clamp', 'fault': 'fault'}

    def dma_padding_mode(self, requested):
        """Map a requested padding behaviour, or refuse it."""
        available = self._d['dma']['padding_modes']
        mapped = self._PADDING_MAP.get(requested)
        if mapped is not None and mapped in available:
            return mapped
        raise CapabilityError(
            f"target {self.name!r} cannot pad a transfer with {requested!r}",
            because=(f"the transfer engine implements {available}; ISA "
                     "sec 9E.1 fills padded elements rather than performing a "
                     "masked access, and no fill value is equivalent to "
                     f"{requested!r}"),
            remedy=('materialise the padding explicitly before the transfer, '
                    'or fall back to a masked LSU loop'),
            source='compiler_overview.md sec 12')

    @property
    def gather_index_scaling(self):
        """'element' -- gather indices are scaled by the access size.

        ISA sec 9.1: lane i reads Sa + sext(Vb[i]) * sz. A lowering that
        computes byte offsets is wrong for every lane but lane 0, which is
        exactly the shape of bug that survives a careless test.
        """
        return self._d['memory']['gather_index_scaling']

    @property
    def has_scalar_fp(self):
        """False: SALU is integer only.

        sec 12 warns against assuming otherwise. A uniform FP32 value has to
        travel through the vector domain, or as a bit pattern a scalar
        register merely carries.
        """
        return bool(self._d['scalar']['has_floating_point'])

    @property
    def gather_index_bits(self):
        return self._d['memory']['gather_index_bits']

    def gather_reach_elements(self):
        """How far a gather can reach from its scalar base, in elements.

        sec 12: offsets beyond this need rebasing, grouping or a per-lane
        scalar fallback. The index is scaled by the access size, so the reach
        is in elements and the byte reach depends on the dtype.
        """
        bits = self.gather_index_bits
        return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)

    @property
    def has_addressable_scratchpad(self):
        return bool(self._d['memory']['addressable_scratchpad'])

    # ---- capability gate (sec 12) -----------------------------------------
    # Capabilities a target file names only where they are true (R41 X8,
    # tools/isa/gen_target.py): absent is False, so the files of every image
    # built before them stay byte-identical and keep the old lowering.
    _OPTIONAL_CAPABILITIES = frozenset({'tensor_transpose_8bit', 'matrix_dp'})

    def has(self, capability):
        if (capability in self._OPTIONAL_CAPABILITIES
                and capability not in self._d['capabilities']):
            return False
        if capability not in self._d['capabilities']:
            raise KeyError(
                f"unknown capability {capability!r}; target describes "
                f"{sorted(self._d['capabilities'])}")
        return bool(self._d['capabilities'][capability])

    def require(self, capability, what=None):
        """Assert a capability, or refuse with something actionable.

        Never returns False: sec 12 wants a diagnostic, and the caller that
        ignores a returned flag is exactly how a trapping opcode gets emitted.
        """
        if self.has(capability):
            return True
        what = what or _CAPABILITY_USE.get(capability, f"use {capability}")
        fallback = _CAPABILITY_FALLBACK.get(capability, 'unknown')
        if fallback is None:
            remedy = ('none: there is no correct substitute, so the kernel '
                      'cannot be compiled for this target')
        elif fallback == 'unknown':
            remedy = 'no fallback is defined for this capability'
        else:
            remedy = fallback
        raise CapabilityError(
            f"target {self.name!r} cannot {what}",
            because=(f"capability {capability!r} is not present; ISA sec 11 "
                     f"makes the encoding raise ILLEGAL_INSTRUCTION"),
            remedy=remedy,
            source='compiler_overview.md sec 12')

    @property
    def capabilities(self):
        return dict(self._d['capabilities'])

    # ---- resource budgeting (sec 10.1, 10.3) ------------------------------
    def check_budget(self, *, barriers=0, accumulators=0, windows=0,
                     descriptors=0, rctx=0):
        """Refuse a plan that does not fit, naming what to shrink.

        sec 10.3 orders the fallbacks; this reports the overflow so a caller
        can apply them, rather than silently reusing a slot that an async
        command has not finished with.
        """
        limits = [
            ('barriers', barriers, self.num_barriers, 'fewer pipeline stages'),
            ('accumulators', accumulators, self.num_accumulators,
             'a smaller output tile'),
            ('RF windows', windows, self.num_windows, 'fewer in-flight stages'),
            ('descriptors', descriptors, self.num_descriptors,
             'reuse a descriptor slot across tensors'),
            ('Role Contexts', rctx, self.num_rctx, 'a coarser partition'),
        ]
        over = [(n, w, h, fix) for n, w, h, fix in limits if w > h]
        if not over:
            return True
        name, want, have, fix = over[0]
        raise ResourceError(
            f"plan needs {want} {name}, target {self.name!r} has {have}",
            because='these are allocated for the whole program, not per role',
            remedy=f"try {fix}",
            source='compiler_overview.md sec 10.1, 10.3')

    def __repr__(self):
        return (f"<TargetInfo {self.name!r} rctx={self.num_rctx} "
                f"vlen={self.vlen} vregs={self.vregs_per_rctx}/"
                f"{self.phys_vregs} acc={self.num_accumulators} "
                f"bar={self.num_barriers}>")
