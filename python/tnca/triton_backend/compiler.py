"""TNCA as a Triton backend: the compile half.

Triton finds this through the `triton.backends` entry point `tnca`
(pyproject.toml). With it, `kernel[grid](...)` and anything built on Triton's
JIT -- PyTorch Inductor included -- compile for TNCA without a TNCA-specific
frontend: Triton parses, specialises and caches; this backend only adds the
stage Triton cannot supply, TTIR to TNCA machine code.

That stage is `tnca.aot._lower_ttir`, the same function `compile_aot` uses,
and the container is written by the same `tnca.aot._seal`. There is no
second lowering here to drift away from the first.

Profile and optimisation level come from the environment (TNCA_PROFILE,
default core-sim; TNCA_OPT_LEVEL, default 2), because Triton's options are
per-kernel keyword arguments that Inductor and user code do not know about.
Both enter the options hash, so a different profile is a different cache
entry rather than a stale one.
"""
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import tempfile

from triton.backends.compiler import BaseBackend, GPUTarget

from tnca_cc.triton_options import (CODEGEN_FNS, DOT_INPUT_PRECISIONS)


def _profile():
    return os.environ.get('TNCA_PROFILE', 'core-sim')


def _opt_level():
    return int(os.environ.get('TNCA_OPT_LEVEL', '2'))


def _index_opt():
    # R39-02a's importer switch, on for the standard path since its ablation
    # passed. TNCA_INDEX_OPT=0 turns it off; compile_aot defaults it off.
    return int(os.environ.get('TNCA_INDEX_OPT', '1'))


def _schedule():
    # R39-01's switch, on for the standard path since its ablation passed
    # (doc/r39_tasks_and_acceptance.md R39-01). TNCA_SCHEDULE=0 turns it off.
    return int(os.environ.get('TNCA_SCHEDULE', '1'))


def _sparse_tensormaps():
    # Store only a TensorMap's non-zero words; the runtime zero-fills
    # scratch. On for the standard path since its ablation passed
    # (doc/r39_tasks_and_acceptance.md R39-08); TNCA_SPARSE_TENSORMAPS=0 off.
    return int(os.environ.get('TNCA_SPARSE_TENSORMAPS', '1'))


def _physical_chunks():
    from tnca.aot import _physical_chunks_default
    return _physical_chunks_default()


def _rctx_spmd():
    # R39-05's switch: every Role Context runs programs of one grid row, for
    # a kernel that allows it. On for the standard path since its ablation
    # passed (doc/r39_tasks_and_acceptance.md R39-05); TNCA_RCTX_SPMD=0 off.
    return int(os.environ.get('TNCA_RCTX_SPMD', '1'))


def _rctx_roles():
    # R39-12's switch: a GEMM loop as a producer and a consumer on a Role
    # Context pair (Mode A), on a target with rctx_pair. Off until its
    # ablation is in; TNCA_RCTX_ROLES=1 on.
    return int(os.environ.get('TNCA_RCTX_ROLES', '0'))


def _rctx_role_pairs():
    return int(os.environ.get('TNCA_RCTX_ROLE_PAIRS', '0'))


def _single_role_pipeline():
    return int(os.environ.get('TNCA_SINGLE_ROLE_PIPELINE', '0'))


def _rctx_stages():
    from tnca.aot import _pipeline_stages_default
    return _pipeline_stages_default()


def _bank_alloc():
    value = int(os.environ.get('TNCA_BANK_ALLOC', '0'))
    if value not in (0, 1):
        raise ValueError('TNCA_BANK_ALLOC must be 0 or 1')
    return value


def _window_reuse():
    value = int(os.environ.get('TNCA_WINDOW_REUSE', '0'))
    if value not in (0, 1):
        raise ValueError('TNCA_WINDOW_REUSE must be 0 or 1')
    return value


def _ieee_div():
    value = int(os.environ.get('TNCA_IEEE_DIV', '0'))
    if value not in (0, 1):
        raise ValueError('TNCA_IEEE_DIV must be 0 or 1')
    return value


def _packed_window_unpack():
    value = int(os.environ.get('TNCA_PACKED_WINDOW_UNPACK', '0'))
    if value not in (0, 1):
        raise ValueError('TNCA_PACKED_WINDOW_UNPACK must be 0 or 1')
    return value


def _tiled_pointwise():
    value = int(os.environ.get('TNCA_TILED_POINTWISE', '0'))
    if value not in (0, 1, 2):
        raise ValueError('TNCA_TILED_POINTWISE must be 0, 1 or 2')
    return value


@dataclass(frozen=True)
class TNCAOptions:
    # What ASTSource.make_ir reads; tnca_cc.triton_options.frontend_options
    # is the same list for the AOT entry point.
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    debug: bool = False
    sanitize_overflow: bool = False
    arch: str = None
    allowed_dot_input_precisions: tuple = DOT_INPUT_PRECISIONS
    supported_fp8_dtypes: tuple = ('fp8e4nv', 'fp8e5')
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    default_dot_input_precision: str = 'ieee'
    max_num_imprecise_acc_default: int = 0
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    extern_libs: tuple = ()
    backend_name: str = 'tnca'
    # Options Triton's JIT passes to every backend; accepted and inert.
    instrumentation_mode: str = ''
    ir_override: str = None
    enable_fp_fusion: bool = True
    # TNCA's own
    profile: str = field(default_factory=_profile)
    opt_level: int = field(default_factory=_opt_level)
    schedule: int = field(default_factory=_schedule)
    index_opt: int = field(default_factory=_index_opt)
    physical_chunks: int = field(default_factory=_physical_chunks)
    rctx_spmd: int = field(default_factory=_rctx_spmd)
    sparse_tensormaps: int = field(default_factory=_sparse_tensormaps)
    rctx_roles: int = field(default_factory=_rctx_roles)
    rctx_role_pairs: int = field(default_factory=_rctx_role_pairs)
    single_role_pipeline: int = field(default_factory=_single_role_pipeline)
    rctx_stages: int = field(default_factory=_rctx_stages)
    bank_alloc: int = field(default_factory=_bank_alloc)
    window_reuse: int = field(default_factory=_window_reuse)
    packed_window_unpack: int = field(default_factory=_packed_window_unpack)
    tiled_pointwise: int = field(default_factory=_tiled_pointwise)
    ieee_div: int = field(default_factory=_ieee_div)

    def __post_init__(self):
        if type(self.rctx_role_pairs) is not int or self.rctx_role_pairs not in (0, 1, 2):
            raise ValueError('rctx_role_pairs must be 0, 1 or 2')
        if self.rctx_role_pairs and not self.rctx_roles:
            raise ValueError('rctx_role_pairs requires rctx_roles')
        if type(self.single_role_pipeline) is not int or self.single_role_pipeline not in (0, 1):
            raise ValueError('single_role_pipeline must be 0 or 1')
        if self.single_role_pipeline and self.rctx_roles:
            raise ValueError('single_role_pipeline and rctx_roles are mutually exclusive')
        if type(self.rctx_stages) is not int or self.rctx_stages not in (1, 2, 3):
            raise ValueError('rctx_stages must be 1, 2 or 3')

    def hash(self):
        key = '_'.join(f'{k}-{v}' for k, v in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode()).hexdigest()


# MLIR scalar type -> the argument ABI's type (tnca.abi)
_SCALAR = {'i32': 'i32', 'i64': 'i64', 'f32': 'f32'}
_FUNC_RE = re.compile(r'tt\.func\s+(?:public\s+|private\s+)?@(\w+)\(')


def _split_top(text):
    """Split at commas that are not inside <>, (), [] or {}."""
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '<([{':
            depth += 1
        elif ch in '>)]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def ttir_params(text):
    """(kernel name, [(name, abi type)]) from a TTIR module's entry function.

    The printed module keeps the Python parameter names (`%x_ptr`). Triton's
    JIT expands a tensor-descriptor parameter into the descriptor plus
    `name.shape.k` / `name.stride.k` scalars; the TNCA importer reads those
    from the TensorMap, exactly as on the compile_aot entry point, so they are
    not arguments of the artifact. A parameter the JIT specialised into a
    constant is not in the function at all.
    """
    m = _FUNC_RE.search(text)
    if not m:
        raise ValueError('no tt.func in the module')
    name = m.group(1)
    depth, i = 1, m.end()
    while depth:
        depth += {'(': 1, ')': -1}.get(text[i], 0)
        i += 1
    params = []
    for k, arg in enumerate(_split_top(text[m.end():i - 1])):
        pname, ty = arg.split(':', 1)
        pname = pname.strip().lstrip('%')
        if '.' in pname:
            continue
        if pname.isdigit() or pname.startswith('arg') and pname[3:].isdigit():
            pname = f'arg{k}'
        ty = re.sub(r'\s+loc\(.*\)\s*$', '', ty.strip(), flags=re.S)
        ty = re.sub(r'\s*\{.*\}\s*$', '', ty, flags=re.S)       # argument attributes
        if ty.startswith(('!tt.ptr', '!tt.tensordesc')):
            kind = 'device_ptr64'
        elif ty in _SCALAR:
            kind = _SCALAR[ty]
        else:
            raise ValueError(f'argument {pname} has type {ty!r}, which the TNCA argument ABI does not carry')
        params.append((pname, kind))
    return name, params


_FINGERPRINT = None


def _compiler_fingerprint():
    global _FINGERPRINT
    if _FINGERPRINT is None:
        import tnca_cc
        from tnca import aot
        h = hashlib.sha256(aot._lock_hash().encode())
        files = sorted(Path(tnca_cc.__file__).parent.rglob('*.py'))
        files += [Path(aot.__file__), Path(__file__), Path(__file__).with_name('libdevice.py')]
        from tnca_cc._datapath import find
        files.append(Path(find('spec', 'r39_sched_model.json')))
        for f in files:
            h.update(str(f.name).encode())
            h.update(f.read_bytes())
        _FINGERPRINT = 'tnca-' + h.hexdigest()[:32]
    return _FINGERPRINT


class TNCABackend(BaseBackend):
    binary_ext = 'tnca'

    @staticmethod
    def supports_target(target: GPUTarget):
        # 'cpu' is what PyTorch Inductor asks for when it generates Triton for
        # CPU tensors (cpu_backend="triton"). With this driver active there is
        # no CPU Triton backend to serve it, and those host tensors are exactly
        # what the TNCA launcher offloads -- so Inductor's CPU-Triton target is
        # routed here. With TNCA_TRITON=0 the driver is inactive and nothing
        # is routed.
        return target.backend in ('tnca', 'cpu')

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def hash(self):
        """Part of Triton's cache key for every kernel this backend compiles.

        It must change whenever the code that produces the artifact does, or
        Triton's disk cache serves an artifact from an older compiler. So it
        covers the toolchain lock and the content of the compile path: the
        tnca_cc package, tnca.aot, and this backend's compile half.
        """
        return _compiler_fingerprint()

    def parse_options(self, opts):
        known = TNCAOptions.__dataclass_fields__
        args = {k: v for k, v in opts.items() if k in known and v is not None}
        # Triton passes list-valued options as lists; the options object
        # must be hashable.
        for k, v in list(args.items()):
            if isinstance(v, list):
                args[k] = tuple(v)
        # Warps and pipeline stages are GPU launch shapes; TNCA's lowering
        # reads neither, so configs that differ only there are one compile.
        args['num_warps'] = 1
        args['num_stages'] = 1
        resolved = TNCAOptions(**args)
        if not isinstance(resolved.physical_chunks, int) or resolved.physical_chunks not in (0, 1):
            raise ValueError('physical_chunks must be 0 or 1')
        return resolved

    def pack_metadata(self, metadata):
        return (metadata.num_warps, metadata.num_ctas, metadata.shared)

    def get_codegen_implementation(self, options):
        return dict(CODEGEN_FNS)

    def get_module_map(self):
        # Imported here, not at module import: libdevice.py defines @triton.jit
        # functions, and this module is loaded while Triton is still
        # discovering its backends.
        from . import libdevice
        return {'triton.language.extra.libdevice': libdevice}

    def load_dialects(self, ctx):
        pass

    def add_stages(self, stages, options, language=None):
        # TTIR arrives from the frontend as it is: the TNCA lowering does its
        # own legalisation (inlining) and optimisation, exactly as on the AOT
        # entry point, so running Triton's GPU passes here would make the two
        # entry points lower different modules.
        stages['ttir'] = lambda src, metadata: src
        stages['tnca'] = lambda src, metadata: self.make_tnca(src, metadata, options)

    def make_tnca(self, module, metadata, options):
        from tnca import aot, device as _device, dump as _dump
        from tnca.adapters import r1 as adapter
        text = str(module)
        kernel_name, params = ttir_params(text)
        prov = adapter.Provenance()
        # The artifact is for the architecture; a memory mode ('.../hbm',
        # '.../slow') only says where the session runs it, which
        # tnca_profile below keeps for the launcher.
        target, pinfo = adapter.target_for(_device.base_profile(options.profile), prov)
        prov.record('source_sha256', hashlib.sha256(text.encode()).hexdigest())
        prov.record('source_kind', 'triton')
        prov.record('entry_point', 'triton-backend')
        prov.record('triton_version', __import__('triton').__version__)
        stages = _dump.Dump(kernel_name)
        stages.record('ttir', text)
        ctx = module.context
        compile_options = {'opt_level': options.opt_level}
        stage_default = os.environ.get('TNCA_PIPELINE', os.environ.get('TNCA_RCTX_STAGES', '2'))
        if options.rctx_stages != 2 or int(stage_default) != options.rctx_stages:
            compile_options['rctx_stages'] = options.rctx_stages
        # Resolved JIT values override environment defaults, including zero.
        # Keep default off absent to preserve the frozen B39 artifact bytes.
        for name, env in (('physical_chunks', 'TNCA_PHYSICAL_CHUNKS'),
                          ('schedule', 'TNCA_SCHEDULE'), ('index_opt', 'TNCA_INDEX_OPT'),
                          ('rctx_spmd', 'TNCA_RCTX_SPMD'),
                          ('sparse_tensormaps', 'TNCA_SPARSE_TENSORMAPS'),
                          ('rctx_roles', 'TNCA_RCTX_ROLES'), ('rctx_role_pairs', 'TNCA_RCTX_ROLE_PAIRS'),
                          ('single_role_pipeline', 'TNCA_SINGLE_ROLE_PIPELINE'), ('bank_alloc', 'TNCA_BANK_ALLOC'),
                          ('window_reuse', 'TNCA_WINDOW_REUSE'),
                          ('packed_window_unpack', 'TNCA_PACKED_WINDOW_UNPACK'),
                          ('tiled_pointwise', 'TNCA_TILED_POINTWISE'),
                          ('ieee_div', 'TNCA_IEEE_DIV')):
            value = getattr(options, name)
            if value or int(os.environ.get(env, '0')) != value:
                compile_options[name] = value
        # The dump switches of doc/dumping_ir.md (TNCA_KERNEL_DUMP=1, written
        # under TNCA_DUMP_DIR/<kernel>/), on this entry point as on AOT --
        # and written for a kernel that FAILS to lower too, with every stage
        # it reached, which is when a dump is most wanted:
        # tools/r38/ttir_lower.py reruns the failing lowering from the .ttir.
        dump_to = _dump.requested()
        try:
            (kernel_name, params, asm_text, code, symbols, report,
             _stages) = aot._lower_ttir(module, ctx, kernel_name, params, target, prov,
                                        compile_options, stages, options.opt_level)
        except Exception as e:
            from .driver import _bench_local
            if getattr(_bench_local, 'compile_only', False):
                # triton.autotune is trying a config (driver.bench): one TNCA
                # cannot lower is a config this device cannot run, which the
                # tuner drops -- as it drops one out of GPU resources.
                from triton.runtime.errors import OutOfResources
                raise OutOfResources(0, 0, f'TNCA: {str(e).splitlines()[0][:200]}') from e
            if dump_to:
                stages.write(dump_to, metadata={'entry_point': 'triton-backend', 'profile': options.profile,
                                                'opt_level': options.opt_level,
                                                'failed': f'{type(e).__name__}: {str(e)[:500]}'})
            raise
        if dump_to:
            stages.write(dump_to, metadata={'entry_point': 'triton-backend', 'profile': options.profile,
                                            'opt_level': options.opt_level})
        # The device's code store (rtl/tnca_multicore_shell.v CODE_WORDS,
        # the runtime's region below heap_base) holds 128 KiB. A kernel
        # larger than that can never be loaded; refuse it here, as a
        # resource error, so a caller with a narrower configuration retiles
        # instead of failing at launch.
        # The size comes from the execution resource plan, the one resource
        # description admission, retile and tuning all read.
        from tnca.inductor import CODE_STORE_BYTES
        from tnca_cc.execution_resources import admission, record_decision
        measured, plan = aot.execution_plan(prov, asm_text, code, report, target,
                                            code_limit=CODE_STORE_BYTES)
        size = plan['resources']['code_bytes'] if plan is not None else measured['code_bytes']
        if size > CODE_STORE_BYTES:
            from triton.runtime.errors import OutOfResources
            raise OutOfResources(size, CODE_STORE_BYTES, 'TNCA code store bytes')
        if plan is not None:
            over = [r for r in admission(plan, code_limit=CODE_STORE_BYTES) if r[0] != 'code_bytes']
            record_decision(plan, 'code_store_admission', 'accepted',
                            f'{size} code bytes within the {CODE_STORE_BYTES}-byte code store',
                            over_profile_limits=over)
        # Aliasing. Triton lets pointer arguments overlap (PyTorch Inductor
        # passes one storage as input and output of an in-place op), and the
        # runtime refuses an overlap the manifest does not declare (ABI-05).
        # A program whose memory traffic is all LSU instructions is safe to
        # alias: tnca_lsu.v has one instruction in flight at a time and a
        # write is answered like a read, so every access happens in program
        # order and overlapping arguments behave exactly as the sequential
        # program says. The transfer engine is a second memory client that
        # can overtake an LSU store still in flight, so a program that uses
        # it declares nothing and an overlap stays refused.
        if not re.search(r'\bt\.(ld|st)\.async\b', asm_text):
            ptrs = [n for n, t in params if t == 'device_ptr64']
            compile_options['alias'] = [[a, b] for i, a in enumerate(ptrs) for b in ptrs[i + 1:]]
        fd, path = tempfile.mkstemp(suffix='.tnca')
        os.close(fd)
        try:
            aot._seal(path, kernel_name, params, asm_text, code, report, target, pinfo,
                      {}, compile_options, prov, 'triton', measured=measured)
            with open(path, 'rb') as fh:
                blob = fh.read()
        finally:
            os.unlink(path)
        metadata['name'] = kernel_name
        metadata['shared'] = 0
        metadata['tnca_profile'] = options.profile
        metadata['tnca_params'] = [n for n, _ in params]
        return blob
