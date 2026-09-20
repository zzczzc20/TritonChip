"""
The AOT compile driver -- doc/phase2_tasks_and_acceptance.md R2-03.

    source + signature + constexpr + profile + options -> artifact

One product entry point, for both inputs. `source_kind` says which adapter
produced an artifact and travels in the manifest, because the two are not the
same claim: the contract is explicit that a capability reached through the
TNCA IR input may not be written up as "Triton GEMM works" (DEP-03).

Three properties this file is responsible for.

**No literal reaches the code.** Every compile here binds parameters through
the argument block, so one artifact runs with different addresses, scalars and
data. The literals a caller passes are still required -- the compiler needs to
know each argument's SHAPE -- and are used for nothing else.

**A failure leaves nothing.** The artifact is written through
`artifact.write`, which builds the whole container in memory and then does one
`os.replace`. A compile that fails at any point before that has written no
file, and an artifact already at the path is untouched (CMP-04).

**The cache key is about the compile, not about the run.** Source, constexprs,
options, profile and the toolchain lock are in it; buffer addresses, argument
values, the grid, job IDs and seeds are not, because changing them must NOT
produce a second artifact (CACHE-01).
"""
import hashlib
import inspect as _inspect
import json
import os
import time

from . import abi, artifact, manifest as M
from .adapters import r1 as adapter
from .errors import (ArgumentError, CompileError, ProfileMismatch,
                     SignatureError, TncaError, UnsupportedError)

FORMAT_VERSION = list(M.FORMAT_VERSION)
ABI_VERSION = list(abi.ABI_VERSION)

# Options this driver understands. An unknown option is refused rather than
# ignored: a caller who misspells `opt_level` and gets silence has asked for
# something and been told nothing, and the artifact they get back is not the
# one they asked for (CMP-02).
KNOWN_OPTIONS = {
    'opt_level': 'int, 0..2; reserved, does not change emission in v1',
    'debug': 'bool; adds a compile report, never changes the code',
    'dump_dir': 'str; where to write the per-stage IR dumps',
    'check_protocol': 'bool; the channel model, which is not a safety check',
}

SOURCE_KINDS = ('triton', 'tnca-ir')

# Triton's signature spelling -> the ABI type. Triton writes `fp32` in a
# signature and `f32` in MLIR; both mean one binary32.
_TRITON_SPELLING = {'fp32': 'f32', 'fp64': 'f64', 'u1': 'bool'}


def _sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _lock_hash():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(root, 'spec', 'r1_toolchain.lock.json')
    try:
        with open(path, 'rb') as fh:
            return _sha(fh.read())
    except OSError:
        return 'unknown'


def check_options(options):
    options = dict(options or {})
    unknown = sorted(set(options) - set(KNOWN_OPTIONS))
    if unknown:
        raise CompileError(
            f"unknown compile option(s): {', '.join(unknown)}",
            detail={'known': sorted(KNOWN_OPTIONS)},
            remedy='an option this driver does not understand is refused '
                   'rather than ignored, so a misspelling cannot silently '
                   'produce a different artifact')
    return options


def cache_key(*, source_kind, source_hash, kernel_name, signature,
              constexprs, profile, config_hash, options):
    """Content address of a compile.

    What is deliberately ABSENT: buffer addresses, argument values, the grid,
    job IDs, seeds. Changing any of those must reuse the artifact, which is
    the whole point of loading parameters from the argument block.
    """
    body = json.dumps({
        'source_kind': source_kind,
        'source': source_hash,
        'kernel': kernel_name,
        'signature': list(signature),
        'constexprs': dict(sorted((constexprs or {}).items())),
        'profile': profile,
        'config_hash': config_hash,
        'options': dict(sorted(options.items())),
        'toolchain': _lock_hash(),
        'format': FORMAT_VERSION,
        'abi': ABI_VERSION,
    }, sort_keys=True, separators=(',', ':'))
    return _sha(body)


class CompileResult:
    def __init__(self, path, manifest, report, key, from_cache=False):
        self.path = path
        self.manifest = manifest
        self.report = report
        self.cache_key = key
        self.from_cache = from_cache

    def __repr__(self):
        return (f"<CompileResult {self.manifest['identity']['kernel_name']!r} "
                f"{'cached' if self.from_cache else 'compiled'} "
                f"{os.path.basename(self.path)}>")


def _signature_types(signature, param_names):
    """(name, abi_type) in the frozen declaration order.

    `signature` is Triton's: '*fp32' for a pointer, 'i32', 'f32', 'constexpr'.
    A constexpr is not an argument -- it was baked in at compile time -- so it
    is not in the block, and an ABI that included it would put every later
    parameter at the wrong offset.
    """
    out = []
    for name in param_names:
        declared = signature.get(name)
        if declared is None:
            raise SignatureError(
                f"parameter {name!r} is not in the signature",
                detail={'signature': sorted(signature)})
        if declared == 'constexpr':
            continue
        if declared.startswith('*'):
            kind = 'device_ptr64'
        elif declared in ('i32', 'u32', 'i64', 'u64', 'f32'):
            kind = declared
        # Triton spells a 32-bit float `fp32` in a signature and `f32` in
        # MLIR. Accepting only one of them refused a kernel the frontend was
        # perfectly happy with, and the error named the type rather than the
        # spelling, which is the least useful way to say it.
        elif declared in _TRITON_SPELLING:
            kind = _TRITON_SPELLING[declared]
        else:
            raise SignatureError(
                f"parameter {name!r} is declared {declared!r}",
                detail={'carried': ', '.join(sorted(abi.TYPES))},
                remedy='pass it through a buffer element')
        abi.check_type(name, kind)
        out.append((name, kind))
    return out


def _placeholders(params):
    """Shape-only values for the compile.

    The compiler needs to know that `x_ptr` is a pointer and `n` an i32; in
    dynamic mode it needs nothing else, and these never reach the code. They
    are distinct and recognisable so that a literal leaking into an artifact
    is visible rather than plausible.
    """
    values = {}
    for i, (name, kind) in enumerate(params):
        if kind == 'device_ptr64':
            values[name] = 0xDEAD0000 + 0x1000 * i
        elif kind == 'f32':
            values[name] = 0.0
        else:
            values[name] = 0
    return values


def compile_aot(source, *, kernel_name=None, signature=None, constexprs=None,
                profile='core-sim', options=None, output=None,
                source_kind='triton', param_names=None):
    """Compile one kernel into one artifact.

    `source` is a @triton.jit function (source_kind 'triton') or a callable
    taking a Builder (source_kind 'tnca-ir').
    """
    started = time.time()
    if source_kind not in SOURCE_KINDS:
        raise CompileError(
            f"unknown source kind {source_kind!r}",
            detail={'known': ', '.join(SOURCE_KINDS)})
    options = check_options(options)
    constexprs = dict(constexprs or {})

    prov = adapter.Provenance()
    target, pinfo = adapter.target_for(profile, prov)

    if source_kind == 'triton':
        result = _compile_triton(source, kernel_name, signature, constexprs,
                                 target, prov, options, param_names)
    else:
        result = _compile_tnca_ir(source, kernel_name, signature, constexprs,
                                  target, prov, options)
    kernel_name, params, asm_text, code, symbols, report = result

    key = cache_key(source_kind=source_kind,
                    source_hash=prov.entries.get('source_sha256', ''),
                    kernel_name=kernel_name,
                    signature=[f"{n}:{t}" for n, t in params],
                    constexprs=constexprs, profile=profile,
                    config_hash=pinfo['config_hash'], options=options)

    man = _manifest(kernel_name, params, target, pinfo, constexprs, options,
                    prov, source_kind, report)

    if output is None:
        output = f"{kernel_name}-{key[:12]}{artifact.SUFFIX}"
    segments = {'text': code}
    sealed = artifact.write(output, man, segments)
    report['compile_seconds'] = round(time.time() - started, 3)
    report['cache_key'] = key
    return CompileResult(output, sealed, report, key)


def _compile_triton(fn, kernel_name, signature, constexprs, target, prov,
                    options, param_names):
    """The real Triton frontend. Not a stored TTIR (CMP-01 says so)."""
    try:
        import triton                                     # noqa: F401
        from triton._C.libtriton import ir                # noqa: F401
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
    except ImportError as e:
        raise CompileError(
            "Triton is not installed in this environment",
            detail={'import_error': str(e)},
            remedy='install the package with its compile extra')

    kernel_name = kernel_name or getattr(fn, '__name__', None)
    if not kernel_name:
        raise CompileError('no kernel name, and the source has no __name__')
    if signature is None:
        raise SignatureError('a Triton compile needs a signature')

    if param_names is None:
        const = set(getattr(fn, 'constexprs', ()) or ())
        names = getattr(fn, 'arg_names', None)
        if names is None:
            raise SignatureError(
                'the source is not a @triton.jit function; pass param_names')
        param_names = [n for i, n in enumerate(names) if i not in const]

    params = _signature_types(signature, param_names)

    class _Options:
        num_warps = 1
        num_ctas = 1
        num_stages = 1
        debug = False
        sanitize_overflow = False
        arch = None
        allowed_dot_input_precisions = ()
        launch_cooperative_grid = False
        backend_name = 'tnca'

    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    try:
        module = src.make_ir(GPUTarget('tnca', 0, 32), _Options(), {}, {}, ctx)
    except Exception as e:
        raise CompileError(f"the Triton frontend refused this kernel: {e}",
                           where=f"@triton.jit {kernel_name}")

    try:
        source_text = _inspect.getsource(fn.fn if hasattr(fn, 'fn') else fn)
    except Exception:
        source_text = kernel_name
    prov.record('source_sha256', _sha(source_text))
    prov.record('source_kind', 'triton')
    prov.record('triton_version', __import__('triton').__version__)

    # dynamic_args is the adapter's default, and the reason this driver
    # exists: one artifact, any arguments. The placeholders below tell the
    # compiler each parameter's SHAPE and reach the code nowhere.
    kernel, _builder, imp = adapter.import_ttir(
        module, kernel_name, _placeholders(params),
        param_names=[n for n, _t in params], target=target, prov=prov,
        dynamic_args=True)
    asm_text, cg = adapter.emit(
        kernel, target, prov=prov,
        check_protocol=bool(options.get('check_protocol', False)))
    code, symbols = adapter.assemble(asm_text, prov=prov)
    report = dict(stages=['ttir', 'tnca-ir', 'assembly', 'machine'],
                  assembly_lines=len(asm_text.splitlines()),
                  code_bytes=len(code))
    return kernel_name, params, asm_text, code, symbols, report


def _compile_tnca_ir(build, kernel_name, signature, constexprs, target, prov,
                     options):
    """The second input: a TNCA IR program, labelled as one."""
    kernel_name = kernel_name or getattr(build, '__name__', 'kernel')
    params = _signature_types(signature or {}, list((signature or {}).keys()))
    try:
        source_text = _inspect.getsource(build)
    except Exception:
        source_text = kernel_name
    prov.record('source_sha256', _sha(source_text))
    prov.record('source_kind', 'tnca-ir')

    kernel, builder = adapter.new_kernel(kernel_name)
    try:
        build(builder, target)
    except TncaError:
        raise
    except Exception as e:
        raise CompileError(f"the TNCA IR program raised {type(e).__name__}: "
                           f"{e}", where=kernel_name)
    asm_text, cg = adapter.emit(
        kernel, target, prov=prov,
        check_protocol=bool(options.get('check_protocol', False)))
    code, symbols = adapter.assemble(asm_text, prov=prov)
    report = dict(stages=['tnca-ir', 'assembly', 'machine'],
                  assembly_lines=len(asm_text.splitlines()),
                  code_bytes=len(code))
    return kernel_name, params, asm_text, code, symbols, report


def _manifest(kernel_name, params, target, pinfo, constexprs, options, prov,
              source_kind, report):
    caps = target.capabilities if isinstance(target.capabilities, dict) \
        else dict(target.capabilities)
    needed = sorted(c for c, v in caps.items() if v)
    return {
        'identity': {
            'format_version': FORMAT_VERSION,
            'abi_version': ABI_VERSION,
            'isa_version': target.isa_version,
            'kernel_name': kernel_name,
            'source_kind': source_kind,
            'compiler_hash': prov.entries.get('assembly_sha256', ''),
            'source_hash': prov.entries.get('source_sha256', ''),
            'toolchain_lock_hash': _lock_hash(),
            'constexprs': dict(sorted(constexprs.items())),
            'options': dict(sorted(options.items())),
        },
        'target': {
            'profile': pinfo['name'],
            'config_hash': pinfo['config_hash'],
            'implementation_fingerprint': pinfo['config_hash'],
            'capabilities': needed,
        },
        'code': {'segments': [], 'entry_points': {'role0': 0},
                 'constant_segments': []},
        'parameters': {
            'signature': abi.describe(params),
            'buffer_extents': {n: 'unknown' for n, t in params
                               if t == 'device_ptr64'},
            'alias_allowed': [],
        },
        'execution': {
            'role_mask': 1,
            'program_to_role': {'0': ['role0']},
            'grid_dims_supported': [True, False, False],
            'dynamic_constraints': {},
        },
        'resources': {
            'rf_private_per_role': {'role0': target.vregs_per_rctx},
            'shared_windows': [],
            'accumulators': target.num_accumulators,
            'barriers': target.num_barriers,
            'descriptors': target.num_descriptors,
            'commands_outstanding': 0,
            'scratch_bytes': 0,
        },
        'semantics': {
            'completion_scope': 'device',
            'error_contract': 'outputs invalid on failure; no rollback',
            'cross_program_dependency': False,
            'ownership_check_version': prov.entries.get(
                'ownership_check', 'unconditional'),
            'ownership_check_result': 'clean',
        },
        'optional_metadata': {
            'provenance': prov.as_dict(),
            'report': report,
        },
    }


def inspect_artifact(path):
    """Everything an artifact says about itself, without a simulator.

    Header only: no payload, no device, no compiler. `tnca-aot inspect` is
    this function, and a caller with a runtime-only install can use it.
    """
    man, payload_at = artifact.read_header(path)
    table = artifact.segment_table(man)
    return dict(
        path=os.path.abspath(path),
        bytes=os.path.getsize(path),
        manifest=man,
        segments=[{k: s[k] for k in ('name', 'offset', 'length', 'sha256')}
                  for s in table],
        payload_offset=payload_at,
    )
