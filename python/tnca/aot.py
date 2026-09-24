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
import re
import time

from . import _data, abi, artifact, dump as _dump, manifest as M
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
    'no_ll': 'bool; at opt_level 2, switch the layout algebra off and use a legal plan instead. The ablation of phase 3.5 section 4.3, not a fallback: O2 without it never silently stops using LinearLayout',
    'opt_level': 'int, 0..2. 0 is the frozen baseline and emits what this driver has always emitted. 1 runs the pinned upstream canonicalizer and CSE over the kernel and lowers a constant to its narrowest correct form. 2 adds the layout and allocation work (phase 3.5 section 4.3)',
    'debug': 'bool; adds a compile report, never changes the code',
    'dump_dir': 'str; where to write the per-stage IR dumps',
    'check_protocol': 'bool; the channel model, which is not a safety check',
    'alias': 'list of [param, param] pairs that MAY point at the same '
             'memory. An elementwise kernel that reads and writes the same '
             'index can run in place; one that does not is a different '
             'kernel, and the compiler cannot tell which from the '
             'signature, so it is declared here or refused at launch '
             '(ABI-05)',
}

SOURCE_KINDS = ('triton', 'tnca-ir')

# Options that change the artifact, and options that only change what is
# written beside it. A dump directory in the cache key would mean turning
# a dump on hands you a different kernel than the one you were debugging,
# which defeats the purpose of looking; Triton keeps TRITON_KERNEL_DUMP
# out of its key for the same reason.
DIAGNOSTIC_OPTIONS = frozenset({'debug', 'dump_dir'})


def semantic_options(options):
    """The options that belong in the cache key."""
    return {k: v for k, v in (options or {}).items()
            if k not in DIAGNOSTIC_OPTIONS}

# Triton's signature spelling -> the ABI type. Triton writes `fp32` in a
# signature and `f32` in MLIR; both mean one binary32.
_TRITON_SPELLING = {'fp32': 'f32', 'fp64': 'f64', 'u1': 'bool'}

# Both of these live in tnca_cc.triton_options, because there is a
# second caller -- the test bridge -- and when each had its own copy,
# fixing one left `tl.dot` refused by the other.
from tnca_cc.triton_options import (CODEGEN_FNS as _CODEGEN_FNS,
                                    frontend_options as _frontend_options)


def _sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _lock_hash():
    """The toolchain lock, which is part of the cache key.

    'unknown' rather than an exception when it is absent: a runtime-only
    install has no lock to read and does not compile, and refusing over a
    missing file would break `inspect`.
    """
    try:
        with open(_data.path('spec', 'r1_toolchain.lock.json'), 'rb') as fh:
            return _sha(fh.read())
    except OSError:
        return 'unknown'



# --------------------------------------------------------------------------
# What the kernel actually uses, read off the code that was emitted.
#
# R2-12 OBS-01: "没有测量/模型支持的成本标为 unknown，不能填0冒充零开销."
# The manifest used to report the PROFILE's register count as the kernel's,
# and 0 for outstanding commands and scratch. Zero is a measurement; these
# were not measured. What can be counted from the assembly is counted, and
# what cannot is UNKNOWN by name.
# --------------------------------------------------------------------------

UNKNOWN = 'unknown'

INSTRUCTION_BYTES = 8      # ISA sec 3: every instruction is 64 bits

# reg class -> the operand spelling it appears under
_REG_RE = {
    'vector': re.compile(r'\bv(\d+)\b'),
    'scalar': re.compile(r'\bs(\d+)\b'),
    'accumulator': re.compile(r'\ba(\d+)\b'),
    'barrier': re.compile(r'\bb(\d+)\b'),
    'descriptor': re.compile(r'\bd(\d+)\b'),
    'window': re.compile(r'\bw(\d+)\b'),
}

# Instructions that put a memory command in flight. Counting the mnemonics
# gives the number ISSUED, which is an upper bound on the number OUTSTANDING
# and is reported as exactly that -- not as the peak, which needs a schedule.
_ASYNC = re.compile(r'^\s*(t\.(?:ld|st)\.async[\w.]*|g\.(?:ld|st)[\w.]*'
                    r'|td\.load[\w.]*)\b')


def _strip_asm(asm_text):
    """Assembly with comments, labels and directives removed."""
    out = []
    for line in asm_text.splitlines():
        line = line.split('//')[0].split('#')[0] if '//' in line else line
        line = line.split('//')[0]
        body = line.strip()
        if not body or body.endswith(':') or body.startswith('.org'):
            continue
        out.append(body)
    return out


def measure_resources(asm_text, target, code=b''):
    """Counted from the emitted instructions, or UNKNOWN.

    Returned alongside the profile's limits so a reader can see the headroom
    without having to look the profile up, and so OBS-01 can cross-check the
    two against each other.
    """
    body = _strip_asm(asm_text)
    used = {}
    for kind, rx in _REG_RE.items():
        # `#32` immediates and `0x` literals are not registers; the operand
        # regexes are anchored on a word boundary and the immediates all
        # carry a `#`, so they do not collide.
        seen = set()
        for line in body:
            for m in rx.finditer(line):
                seen.add(int(m.group(1)))
        used[kind] = sorted(seen)

    issued = sum(1 for line in body if _ASYNC.match(line))
    # Two different counts, and they are not the same number: `.li` expands
    # to as many as four words (movz + 3x movk, tools/asm sec 22), so the
    # assembly understates the code. Reporting the line count as "the
    # instructions" made a 27-instruction kernel out of a 30-instruction
    # one. Both are given, each saying which it is.
    machine = len(code) // INSTRUCTION_BYTES if code else UNKNOWN
    return {
        'measured_from': 'emitted assembly and machine code',
        'assembly_instructions': len(body),
        'machine_instructions': machine,
        'code_bytes': len(code) if code else UNKNOWN,
        'vector_regs_used': len(used['vector']),
        'scalar_regs_used': len(used['scalar']),
        'accumulators_used': len(used['accumulator']),
        'barriers_used': len(used['barrier']),
        'descriptors_used': len(used['descriptor']),
        'windows_used': len(used['window']),
        'async_commands_issued': issued,
        # Issued is not outstanding: the peak in flight depends on where the
        # waits fall, which needs a schedule this stage does not have.
        'commands_outstanding_peak': UNKNOWN,
        # No kernel in this stage spills, but "no spill slots were emitted"
        # is a statement about the emitter, not a measurement of the kernel.
        'scratch_bytes': UNKNOWN,
        'dma_expansion': UNKNOWN,
        'limits': {
            'vector_regs': target.vregs_per_rctx,
            'accumulators': target.num_accumulators,
            'barriers': target.num_barriers,
            'descriptors': target.num_descriptors,
        },
    }

def _alias_pairs(options, params):
    """The declared aliases, checked against the signature.

    A pair naming a parameter that does not exist, or one that is not a
    pointer, is a mistake worth refusing: it would sit in the manifest
    looking like permission for something and granting nothing.
    """
    raw = (options or {}).get('alias') or []
    ptrs = {n for n, t in params if t == 'device_ptr64'}
    pairs = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise CompileError(
                f"alias entry {pair!r} is not a pair of parameter names",
                remedy="write alias=[['x_ptr', 'o_ptr']]")
        a, b = pair
        for name in (a, b):
            if name not in ptrs:
                raise CompileError(
                    f"alias names {name!r}, which is not a pointer "
                    f"parameter of this kernel",
                    detail={'pointers': sorted(ptrs)},
                    remedy='only two buffers can alias, so both names must '
                           'be device pointers')
        if a == b:
            raise CompileError(
                f"alias pairs {a!r} with itself",
                remedy='a parameter always aliases itself; the declaration '
                       'is about two different parameters')
        pairs.append(sorted([a, b]))
    return sorted(pairs)


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


def _constexpr_metadata(values):
    """Stable JSON values without changing the objects passed to Triton."""
    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if type(value).__module__ == 'triton.language.core' and type(value).__name__ == 'dtype':
            return {'triton_dtype': str(value)}
        return value
    return encode(values or {})


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
        'constexprs': _constexpr_metadata(constexprs),
        'profile': profile,
        'config_hash': config_hash,
        'options': dict(sorted(options.items())),
        'toolchain': _lock_hash(),
        'format': FORMAT_VERSION,
        'abi': ABI_VERSION,
    }, sort_keys=True, separators=(',', ':'))
    return _sha(body)


class CompileResult:
    def __init__(self, path, manifest, report, key, from_cache=False,
                 asm=None, dumped=None):
        self.path = path
        self.manifest = manifest
        self.report = report
        self.cache_key = key
        self.from_cache = from_cache
        # `asm` is Triton's name for this: stage -> text, in memory,
        # whether or not anything was written to disk. A cached result has
        # an empty one, because nothing was compiled to produce it.
        self.asm = asm if asm is not None else {}
        # stage -> path, for the files that were written. Empty when no
        # dump was asked for.
        self.dumped = dumped or {}

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
        if declared.startswith(('*', 'tensordesc<')):
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
                source_kind='triton', param_names=None, cache=None):
    """Compile one kernel into one artifact.

    `source` is a @triton.jit function (source_kind 'triton') or a callable
    taking a Builder (source_kind 'tnca-ir').

    `cache` is a `tnca.cache.Cache`. A hit returns the stored artifact
    without invoking the compiler at all -- the point being that the compiler
    is not invoked, not that it is invoked quickly.
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

    # The key is computable before compiling, because everything in it is an
    # input. A cache that can only be consulted after the work is done is a
    # log of what it cost.
    # Asking for a dump implies a compile. A cache hit is a request not to
    # run the compiler; a dump is a request to see it run, and serving a
    # hit would write no files -- so the first run of a session would dump
    # and every later one would silently not.
    dump_to = _dump.requested(options)
    if cache is not None and (dump_to or _dump.always_compile()):
        cache = None

    if cache is not None:
        early = _key_for(source, source_kind, kernel_name, signature,
                         constexprs, profile, pinfo, options, param_names)
        hit = cache.get(early)
        if hit:
            man, _payload_at = artifact.read_header(hit)
            return CompileResult(hit, man, dict(cached=True), early,
                                 from_cache=True)

    if source_kind == 'triton':
        result = _compile_triton(source, kernel_name, signature, constexprs,
                                 target, prov, options, param_names)
    else:
        result = _compile_tnca_ir(source, kernel_name, signature, constexprs,
                                  target, prov, options)
    (kernel_name, params, asm_text, code, symbols, report,
     stages) = result

    # The same function that was used for the lookup. Computing it a second
    # way is how a cache stores under one key and looks up under another.
    key = _key_for(source, source_kind, kernel_name, signature, constexprs,
                   profile, pinfo, options, param_names)

    measured = measure_resources(asm_text, target, code=code)
    # The emitter cannot see this one: it is a requirement the frontend
    # recorded, not a count of something in the assembly.
    if 'scratch_bytes' in report:
        measured['scratch_bytes'] = report['scratch_bytes']
    report['resources'] = measured
    man = _manifest(kernel_name, params, target, pinfo, constexprs, options,
                    prov, source_kind, report, measured=measured)

    if output is None:
        output = f"{kernel_name}-{key[:12]}{artifact.SUFFIX}"
    segments = {'text': code}
    sealed = artifact.write(output, man, segments)
    report['compile_seconds'] = round(time.time() - started, 3)
    report['cache_key'] = key
    if cache is not None:
        cache.put(key, output)

    written = {}
    if dump_to:
        written = stages.write(dump_to, metadata={
            'kernel_name': kernel_name,
            'source_kind': source_kind,
            'profile': pinfo['name'],
            'cache_key': key,
            'artifact': os.path.abspath(output),
            'constexprs': _constexpr_metadata(constexprs),
            'options': dict(sorted((options or {}).items())),
            'toolchain_lock_hash': _lock_hash(),
            'resources': measured,
        })
    return CompileResult(output, sealed, report, key,
                         asm=dict(stages.stages), dumped=written)


def _source_text(source):
    """The text a compile is keyed on. One definition, used twice."""
    try:
        return _inspect.getsource(source.fn if hasattr(source, 'fn')
                                  else source)
    except Exception:
        return getattr(source, '__name__', repr(source))


def _declared_params(source, source_kind, signature, param_names):
    """The frozen (name, abi_type) list, computed WITHOUT compiling.

    Shared by the early cache lookup and the compile itself, because the two
    must produce the same key -- and when they did not, every lookup missed
    and every compile stored an entry nobody would ever find. The symptom
    was five misses and five stores where there should have been one of
    each; the cause was one side spelling a parameter `*fp32` and the other
    `device_ptr64`.
    """
    if param_names is None:
        if source_kind == 'triton':
            const = set(getattr(source, 'constexprs', ()) or ())
            names = getattr(source, 'arg_names', None) or []
            param_names = [n for i, n in enumerate(names) if i not in const]
        else:
            param_names = list((signature or {}).keys())
    return _signature_types(signature or {}, param_names)


def _key_for(source, source_kind, kernel_name, signature, constexprs,
             profile, pinfo, options, param_names):
    """The cache key, from the inputs alone.

    From the SOURCE TEXT rather than from anything the compiler produced, so
    a hit costs a hash and not a compile. That is the whole claim of
    CACHE-04: a runtime-only environment launching the same kernel
    repeatedly invokes the compiler zero times, and "it was fast" is a
    different statement from "it did not happen".
    """
    params = _declared_params(source, source_kind, signature, param_names)
    name = kernel_name or getattr(source, '__name__', 'kernel')
    return cache_key(source_kind=source_kind,
                     source_hash=_sha(_source_text(source)),
                     kernel_name=name,
                     signature=[f"{n}:{t}:{(signature or {}).get(n, t)}" for n, t in params],
                     constexprs=constexprs, profile=profile,
                     config_hash=pinfo['config_hash'],
                     options=semantic_options(options))


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
    prov.record('tensor_descriptors', {n: signature[n] for n in param_names
                                       if signature[n].startswith('tensordesc<')})

    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    try:
        module = src.make_ir(GPUTarget('tnca', 0, 32),
                             _frontend_options('tnca'), _CODEGEN_FNS, {},
                             ctx)
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
    opt_level = int((options or {}).get('opt_level', 0) or 0)
    stages = _dump.Dump(kernel_name)
    # Captured here rather than re-derived later: this is the module the
    # compile actually consumed, and anything reconstructed afterwards
    # would be a second opinion about it.
    stages.record('ttir', str(module))

    # Legalisation first, at every level: Triton's library functions
    # (`tl.exp`, `tl.sum`, `tl.max`, ...) are @triton.jit functions and
    # arrive as `tt.call` until they are inlined. No backend can lower a
    # call, so this is not an optimisation and is not gated on one --
    # an opt_level may change how good the code is, never whether it
    # compiles.
    from tnca_cc.upstream import PureValueOptimizer as _PVO, PassError as _PE
    try:
        _PVO.legalize_module(module, ctx, name=kernel_name)
    except _PE as e:
        raise CompileError(
            f"the upstream inliner could not run: {e}", where=kernel_name,
            remedy='Triton library calls cannot be lowered until they '
                   'are inlined') from e
    stages.record('ttir_inlined', str(module))

    # Phase 3.5 section 4.1: above O0 the pinned upstream canonicalizer
    # and CSE run over this module, in place, and the importer below
    # consumes the result. Optimising a copy and importing the original
    # would be the failure R35-UP-03 exists to catch.
    if opt_level >= 1:
        from tnca_cc.upstream import PureValueOptimizer, PassError
        try:
            opt = PureValueOptimizer()
            after, diff = opt.optimize_module(module, ctx, name=kernel_name)
        except PassError as e:
            raise CompileError(
                f"opt_level {opt_level} asks for the upstream optimiser "
                f"and it could not run: {e}",
                where=kernel_name,
                remedy='O1 and above do not fall back to no optimisation: '
                       'a stage that reuses upstream cannot answer a '
                       'missing upstream by quietly not reusing it') from e
        prov.record('opt_pipeline', ','.join(opt.pipeline()))
        prov.record('opt_pipeline_version', opt.PIPELINE_VERSION)
        prov.record('opt_ops_before', diff['ops_before'])
        prov.record('opt_ops_after', diff['ops_after'])
        stages.record('ttir_opt', after)

    kernel, _builder, imp = adapter.import_ttir(
        module, kernel_name, _placeholders(params),
        param_names=[n for n, _t in params], target=target, prov=prov,
        dynamic_args=True)
    stages.record('tnir', str(kernel))

    # Scratch the KERNEL asks for, which is not the same thing as spill
    # slots the emitter did not need. A kernel that builds a TensorMap
    # has to write it to memory before `td.load` can snapshot it (ISA
    # sec 9D.1), and a tile that changes shape passes through memory
    # because nothing packs or unpacks one. The buffer arrives in the
    # slot after the declared arguments, so a launch that does not
    # supply it hands the kernel a pointer to whatever was there.
    scratch_bytes = int(getattr(imp, 'scratch_used', 0))
    if scratch_bytes:
        prov.record('scratch_bytes', scratch_bytes)
        prov.record('scratch_argument_slot', len(params))

    # The TNCA IR passes run on this path too. Upstream canonicalisation
    # already worked on the TTIR above, but it does not strength-reduce a
    # multiply -- measured, on exactly this module -- and the importer
    # introduces scalar values of its own that never existed upstream.
    # Phase 3.5 section 4.3: O2 adds the layout work. `no_ll` is the
    # ablation -- the same pipeline with the layout algebra switched off
    # and a legal plan used instead, which for a view means materialising
    # it. No computation is removed either way.
    # A layout view has to be lowered at EVERY level, because a program
    # that contains one cannot be compiled otherwise. What changes with
    # the level is whether anything reasons about it: below O2, and at
    # O2 with the ablation on, each view is materialised from the bases
    # it declares. At O2 the chain is composed with upstream
    # LinearLayout first, and a composition that comes out the identity
    # costs nothing.
    from tnca_cc.layout import apply_plans
    _use_ll = opt_level >= 2 and not (options or {}).get('no_ll')
    lay = apply_plans(kernel, use_upstream=_use_ll)
    if lay['renamed'] or lay['materialised'] or lay['refused']:
        prov.record('opt_views_renamed', lay['renamed'])
        prov.record('opt_views_materialised', lay['materialised'])
        prov.record('opt_layout_upstream', _use_ll)
    if opt_level >= 1:
        from tnca_cc.optimization import (pure_scalar_cse, strength_reduce,
                                           drop_dead_pure,
                                           fold_broadcast_compare,
                                           drop_dead_load_other,
                                           narrow_lane0_store)
        fb = fold_broadcast_compare(kernel)
        dlo = drop_dead_load_other(kernel)
        n0 = narrow_lane0_store(kernel)
        prov.record('opt_lane0_stores', n0['sites'])
        prov.record('opt_immediate_compares', fb['sites'])
        prov.record('opt_unobserved_load_other', dlo['sites'])
        sr = strength_reduce(kernel)
        cse = pure_scalar_cse(kernel)
        dead = drop_dead_pure(kernel)
        prov.record('opt_strength_reduced', sr['sites'])
        prov.record('opt_pure_scalar_merged', cse['merged'])
        prov.record('opt_dead_pure_removed', dead['removed'])
        if sr['sites'] or cse['merged'] or dead['removed']:
            stages.record('tnir_opt', str(kernel))

    asm_text, cg = adapter.emit(
        kernel, target, prov=prov, opt_level=opt_level,
        check_protocol=bool(options.get('check_protocol', False)))
    stages.record('asm', asm_text)

    code, symbols = adapter.assemble(asm_text, prov=prov)
    stages.record('hex', _dump.hex_image(code))

    report = dict(stages=['ttir', 'tnca-ir', 'assembly', 'machine'],
                  assembly_lines=len(asm_text.splitlines()),
                  code_bytes=len(code),
                  scratch_bytes=scratch_bytes)
    return kernel_name, params, asm_text, code, symbols, report, stages


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

    opt_level = int((options or {}).get('opt_level', 0) or 0)
    kernel, builder = adapter.new_kernel(kernel_name)
    try:
        build(builder, target)
    except TncaError:
        raise
    except Exception as e:
        raise CompileError(f"the TNCA IR program raised {type(e).__name__}: "
                           f"{e}", where=kernel_name)
    # No .ttir for this source kind: a TNCA IR program never had one, and
    # an empty file would suggest it did.
    stages = _dump.Dump(kernel_name)
    stages.record('tnir', str(kernel))

    # Phase 3.5 section 4.1: "TNCA IR 输入也必须能到达 LL 与适用的纯值优化".
    # A TNCA IR source never had a TTIR, so the canonicalisation the
    # Triton path gets never reaches it. Above O0 its pure scalar values
    # go across the bridge and upstream says which of them coincide.
    # Phase 3.5 section 4.3: O2 adds the layout work. `no_ll` is the
    # ablation -- the same pipeline with the layout algebra switched off
    # and a legal plan used instead, which for a view means materialising
    # it. No computation is removed either way.
    # A layout view has to be lowered at EVERY level, because a program
    # that contains one cannot be compiled otherwise. What changes with
    # the level is whether anything reasons about it: below O2, and at
    # O2 with the ablation on, each view is materialised from the bases
    # it declares. At O2 the chain is composed with upstream
    # LinearLayout first, and a composition that comes out the identity
    # costs nothing.
    from tnca_cc.layout import apply_plans
    _use_ll = opt_level >= 2 and not (options or {}).get('no_ll')
    lay = apply_plans(kernel, use_upstream=_use_ll)
    if lay['renamed'] or lay['materialised'] or lay['refused']:
        prov.record('opt_views_renamed', lay['renamed'])
        prov.record('opt_views_materialised', lay['materialised'])
        prov.record('opt_layout_upstream', _use_ll)
    if opt_level >= 1:
        from tnca_cc.optimization import (pure_scalar_cse, strength_reduce,
                                           drop_dead_pure,
                                           fold_broadcast_compare,
                                           drop_dead_load_other,
                                           narrow_lane0_store)
        fb = fold_broadcast_compare(kernel)
        dlo = drop_dead_load_other(kernel)
        n0 = narrow_lane0_store(kernel)
        prov.record('opt_lane0_stores', n0['sites'])
        prov.record('opt_immediate_compares', fb['sites'])
        prov.record('opt_unobserved_load_other', dlo['sites'])
        sr = strength_reduce(kernel)
        info = pure_scalar_cse(kernel)
        dead = drop_dead_pure(kernel)
        prov.record('opt_strength_reduced', sr['sites'])
        prov.record('opt_pure_scalar_merged', info['merged'])
        prov.record('opt_dead_pure_removed', dead['removed'])
        for r in info['roles']:
            if r.get('applied'):
                prov.record('opt_pipeline', ','.join(r['pipeline']))
                prov.record('opt_bridge_protocol', r['protocol'])
        if info['merged'] or sr['sites'] or dead['removed']:
            stages.record('tnir_opt', str(kernel))

    asm_text, cg = adapter.emit(
        kernel, target, prov=prov, opt_level=opt_level,
        check_protocol=bool(options.get('check_protocol', False)))
    stages.record('asm', asm_text)

    code, symbols = adapter.assemble(asm_text, prov=prov)
    stages.record('hex', _dump.hex_image(code))

    report = dict(stages=['tnca-ir', 'assembly', 'machine'],
                  assembly_lines=len(asm_text.splitlines()),
                  code_bytes=len(code))
    return kernel_name, params, asm_text, code, symbols, report, stages


def _manifest(kernel_name, params, target, pinfo, constexprs, options, prov,
              source_kind, report, measured=None):
    measured = measured or {}
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
            'constexprs': _constexpr_metadata(constexprs),
            # Semantic options only. A dump directory is a fact about one
            # debugging session, not about the kernel: recording it put a
            # local home directory inside a distributable artifact and
            # made two artifacts with the same cache key differ.
            'options': dict(sorted(semantic_options(options).items())),
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
            'alias_allowed': _alias_pairs(options, params),
        },
        'execution': {
            'role_mask': 1,
            'program_to_role': {'0': ['role0']},
            'grid_dims_supported': [True, False, False],
            'dynamic_constraints': {},
        },
        'resources': {
            'rf_private_per_role': {
                'role0': measured.get('vector_regs_used', UNKNOWN)},
            'shared_windows': [],
            'accumulators': measured.get('accumulators_used', UNKNOWN),
            'barriers': measured.get('barriers_used', UNKNOWN),
            'descriptors': measured.get('descriptors_used', UNKNOWN),
            'commands_outstanding': measured.get(
                'commands_outstanding_peak', UNKNOWN),
            'scratch_bytes': measured.get('scratch_bytes', UNKNOWN),
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
