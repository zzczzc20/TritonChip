"""
The Triton out-of-tree backend plugin for TNCA.

Triton discovers third-party backends as directories under third_party/ that
expose a BaseBackend subclass; building TNCA that way keeps the frontend, the
Triton dialects and the pass infrastructure upstream, which is what
compiler_overview.md sec 3.1 asks for ("直接复用，尽量不修改").

This file needs the pinned Triton (compiler/pin/toolchain.json) importable. It
deliberately does no work at import time beyond reading the target description,
so the stage list and the target contract can be inspected -- and tested --
on a host with no Triton installed. See compiler/README.md.
"""
import functools
import hashlib
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'compiler', 'python'))

from tnca_cc import TargetInfo  # noqa: E402

# sec 14's pipeline, as the stages this backend dumps between. Every stage
# writes its IR when TNCA_DUMP_IR is set, which C0 requires ("每阶段 IR dump").
STAGES = [
    ('ttir', 'Triton IR, upstream frontend output, unmodified'),
    ('ttgir', 'Triton GPU IR: the single-vector-domain subset of sec 4.1'),
    ('tnca_resource', 'resources and async semantics attached (sec 5.2)'),
    ('tnca_channel', 'channels, leases and typed completions (sec 7.4)'),
    ('tnca_protocol', 'protocol lowered to mbarrier operations (sec 8)'),
    ('tnca_alloc', 'whole-program then role-local allocation (sec 10)'),
    ('tnca_asm', 'TNCA assembly, for tools/asm/tnca_asm.py (sec 11.1)'),
    ('bin', 'the assembled image'),
]


@dataclass(frozen=True)
class TNCAOptions:
    """Backend options.

    num_rctx is not a warp count. sec 4.2 and the frozen decision 2 are
    explicit that the role count must not enter the layout as a warp
    dimension; it is an execution-resource budget, kept separate on purpose.
    """
    target: str = 'tnca-sim-v0.1'
    num_rctx: int = 2
    num_stages: int = 2
    debug: bool = False
    dump_ir: bool = False

    def hash(self):
        key = '-'.join(f"{k}={v}" for k, v in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=None)
def target_info(name='tnca-sim-v0.1'):
    return TargetInfo.load(name)


def validate_options(opts, target):
    """Reject an option set the target cannot honour, before any lowering.

    sec 11.1: running out of a resource must be an explicit error. Finding out
    during register allocation, after the protocol has been lowered, produces
    a diagnostic nobody can act on.
    """
    target.check_budget(rctx=opts.num_rctx)
    if opts.num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    # One window and one barrier pair per in-flight stage, per role.
    target.check_budget(windows=opts.num_stages * opts.num_rctx,
                        barriers=2 * opts.num_stages)
    return True


def stage_names():
    return [name for name, _ in STAGES]


class TNCABackend:
    """The BaseBackend Triton loads.

    Subclassing triton.backends.compiler.BaseBackend is deferred to import
    time inside make_backend() so this module stays importable without Triton,
    which is what lets the option and target logic above be unit tested on a
    host that has no LLVM.
    """

    def __init__(self, target):
        self.target = target
        self.info = target_info()
        self.binary_ext = 'tnca'

    def parse_options(self, opts):
        known = {k: v for k, v in opts.items()
                 if k in TNCAOptions.__dataclass_fields__}
        options = TNCAOptions(**known)
        validate_options(options, self.info)
        return options

    def get_codegen_implementation(self, options):
        return {}

    def load_dialects(self, ctx):
        raise NotImplementedError(
            "the tnca dialect is not registered yet; see compiler/README.md "
            "-- milestone C1 is the first Triton->RTL closure")

    def add_stages(self, stages, options, language):
        raise NotImplementedError(
            "lowering lands in C1; C0 delivers the target contract, the ISA "
            "schema and the pinned toolchain")


def make_backend():
    """Build the real backend class, importing Triton only when asked."""
    from triton.backends.compiler import BaseBackend

    class _TNCABackend(BaseBackend, TNCABackend):
        @staticmethod
        def supports_target(target):
            return target.backend == 'tnca'

    return _TNCABackend
